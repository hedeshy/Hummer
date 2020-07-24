from sklearn.ensemble import RandomForestClassifier
from joblib import load
import sounddevice as sd
import librosa
import numpy as np
import asyncio
import datetime
import random
import websockets
import common

SEGMENT_WIDTH_S = 0.5 # limits the "length" of humming

class Recognizer:

	def _callback(self, indata, frames: int, time: dict, status: int) -> None:
		npdata = np.frombuffer(indata, dtype=np.int16)

		# Process audio to window length
		self._segment = np.append(self._segment, npdata)
		sample_count: int = self._segment.shape[0]
		max_sample_count: int = SEGMENT_WIDTH_S * self._rate * self._channels
		count: int = int(min(sample_count, max_sample_count))
		self._segment = self._segment[-count:]
		y: np.ndarray = librosa.to_mono(self._segment)

		# Compute features
		sr: int = self._rate
		seg = common.compute_feature_vector(y, sr)

		# TODO: No scaling performed as the model does not do any scaling, too
		
		# Classify whether humming or not
		pred = self._model.predict(np.array([seg])).astype(int)
		self._humming = bool(pred)

	def stop(self) -> None:
		self._stream.stop()
		self._stream.close()

	def __init__(self, model: RandomForestClassifier):

		# Initialize members
		host_info: dict = sd.query_hostapis(index=None)[0]
		device_info: dict = sd.query_devices(device=host_info['default_input_device'])
		self._channels: int = int(device_info['max_input_channels'])
		self._rate: int = int(device_info['default_samplerate'])
		self._segment: np.array = np.empty(1)
		self._dtype: str = 'int16'
		self._model: RandomForestClassifier = model
		self._humming: bool = False

		# Create stream
		self._stream = sd.RawInputStream(
				device=host_info['default_input_device'],
				dtype=self._dtype,
				# blocksize=32,
				channels=self._channels,
				samplerate=self._rate,
				callback=self._callback)
		self._stream.start()

		print('> start recognizing ' + device_info['name'])

model: RandomForestClassifier = load('model.joblib')
recognizer = Recognizer(model)

async def send(websocket, path):
	while True:
		await websocket.send(str(recognizer._humming))
		await asyncio.sleep(1) # would be better to send when new computation is available

start_server = websockets.serve(send, "127.0.0.1", 5678)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()