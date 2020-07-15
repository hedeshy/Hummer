print('Recorder')

# Overview about parameters in pyaudio: https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio

import pyaudio
import wave
import time
import keyboard

class Recorder:

	def _record_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> (bytes, int):
		self._recorded_frames = self._recorded_frames + in_data # TODO: this is super inefficient
		callback_flag: int = pyaudio.paContinue
		if self._stop_callback:
			callback_flag = pyaudio.paComplete
		return (None, callback_flag)

	def __init__(self):
		self._channels: int = 2
		self._rate: int = 44100
		self._format: int = pyaudio.paInt16
		self._wav_filename: str = 'file.wav'
		self._pa: pyaudio.PyAudio = pyaudio.PyAudio()
		self._recorded_frames: bytes = bytes()
		self._stop_callback: bool = False

		stream = self._pa.open(format=self._format,
				channels=self._channels,
				rate=self._rate,
				input=True,
				stream_callback=self._record_callback)

		stream.start_stream()

		while stream.is_active():
			if keyboard.is_pressed('q'):
				self._stop_callback = True
			time.sleep(0.05)

		stream.stop_stream()
		stream.close()

		with wave.open(self._wav_filename, 'wb') as w:
			w.setnchannels(self._channels)
			w.setsampwidth(self._pa.get_sample_size(self._format))
			w.setframerate(self._rate)
			w.writeframes(self._recorded_frames)
			w.close()

		self._pa.terminate()

recorder: Recorder = Recorder()