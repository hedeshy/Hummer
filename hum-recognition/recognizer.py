# Custom
import common

# Other
import websockets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from joblib import load
import sounddevice as sd
import librosa
import numpy as np
import asyncio
import datetime
import random

class Recognizer:

	def _callback(self, indata, frames: int, time: dict, status: int) -> None:
		data = np.frombuffer(indata, dtype=np.float32)

		# Process audio to window length
		self._segment = np.append(self._segment, data) # copies array
		sample_count: int = self._segment.shape[0]
		window_sample_count: int = common.SEGMENT_WIDTH_SEC * self._rate * self._channels

		# For now, only consider windows of full length (otherswise pca and model breaks as feature count is different than expected)
		if sample_count < window_sample_count:
			return

		# Make window
		count: int = int(min(sample_count, window_sample_count))
		self._segment = self._segment[-count:] # take latest samples as window
		y: np.ndarray = librosa.to_mono(self._segment)

		# Compute features
		fts = np.array([common.compute_feature_vector(y, self._rate)])

		# TODO: No scaling performed as the model does not do any scaling, too
		# fts = self._pca.transform(fts)
		pred = self._model.predict(fts).astype(int)
		self._humming = common.labels[pred[0]]
		print(self._humming)

	def stop(self) -> None:
		self._stream.stop()
		self._stream.close()

	def __init__(self, model, pca: PCA):

		# Initialize members
		host_info: dict = sd.query_hostapis(index=None)[0]
		device_info: dict = sd.query_devices(device=host_info['default_input_device'])
		self._channels: int = 1 # int(device_info['max_input_channels'])
		self._rate: int = 44000 # int(device_info['default_samplerate'])
		self._segment: np.array = np.empty(1)
		self._dtype: str = 'float32'
		self._model = model
		self._pca: PCA = pca
		self._humming: str = common.labels[0]

		# Create stream
		self._stream = sd.RawInputStream(
				device=host_info['default_input_device'],
				dtype=self._dtype,
				blocksize=11000,
				channels=self._channels,
				samplerate=self._rate,
				callback=self._callback)
		self._stream.start()

		print('> start recognizing ' + device_info['name'])

model = load(common.SHARED_PATH + '/model.joblib')
pca: PCA = load(common.SHARED_PATH + '/pca.joblib')
recognizer = Recognizer(model, pca)

async def send(websocket, path):
	while True:
		await websocket.send(recognizer._humming)
		await asyncio.sleep(1) # would be better to send when new computation is available

start_server = websockets.serve(send, "127.0.0.1", 5678)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()