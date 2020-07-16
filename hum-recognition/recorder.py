print('Welcome to the Recorder')
print('esc: exit, 1: start, 2: stop, 3 (pressing): hum')

# Overview about parameters in pyaudio: https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio

import pyaudio
import wave
import time
from pynput import keyboard
from typing import List
import json

def get_ms() -> int:
	return time.time_ns() // 1000000 

class Recorder:
	
	def _record_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> (bytes, int):
		self._recorded_frames.append(in_data)
		return (None, pyaudio.paContinue)

	def start_hum(self) -> None:
		if self._hum_start < 0:
			self._hum_start = get_ms() - self._start_ms

	def stop_hum(self) -> None:
		if self._hum_start >= 0:
			self._humming.append((self._hum_start, get_ms() - self._start_ms))
			self._hum_start = -1


	def stop(self) -> None:

		self._stream.stop_stream()
		self._stream.close()
		self.stop_hum() # under discussion whether makes sense (maybe not)
		print('> stop recording')

		# Store audio as wave
		with wave.open(self._name + '.wav', 'wb') as w:
			w.setnchannels(self._channels)
			w.setsampwidth(self._pa.get_sample_size(self._format))
			w.setframerate(self._rate)
			w.writeframes(b''.join(self._recorded_frames))
			w.close()

		self._pa.terminate()

		# Store meta data as json
		# TODO: duration, date / time?, operation system
		meta: dict = {}
		meta['channels'] = self._channels
		meta['rate'] = self._rate
		meta['format'] = 'Int16' # take member (must decode it)
		meta['hums'] = self._humming
		with open(self._name + '.json', 'w') as w:
			json.dump(meta, w)

		# Gets the number of bytes, similar to length
		# b = b''.join(self._recorded_frames)
		# print(sys.getsizeof(b)) # time: /2 (16 bit) /2 (steore) /44100 (samples per second)

	def __init__(self, name: str):

		# Initialize members
		self._name = name
		self._pa: pyaudio.PyAudio = pyaudio.PyAudio()
		device_info: dict = self._pa.get_default_input_device_info()
		self._channels: int = int(device_info.get('maxInputChannels'))
		self._rate: int = int(device_info.get('defaultSampleRate'))
		self._format: int = pyaudio.paInt16
		self._recorded_frames: List[bytes] = []
		self._hum_start: int = -1 # -1 marks that no humming is going on
		self._humming: List[(int, int)] = [] # list of humming start / end tuples

		# Create stream
		self._stream = self._pa.open(
				format=self._format,
				channels=self._channels,
				rate=self._rate,
				input=True,
				frames_per_buffer=256, # data only stored when 256 collected (crop in the end)
				stream_callback=self._record_callback)
		self._stream.start_stream()

		print('> start recording ' + self._pa.get_default_input_device_info().get('name'))
		self._start_ms = get_ms()

# Variables
recorder: Recorder = None
count: int = 0
start: bool = False
stop: bool = False
escape: bool = False
name: str = 'raphael' # TODO: provide via parameter
humming: bool = False # indicates whether currently humming

# On key press event
def on_press(key) -> bool:
	try: # if special key, 'char' conversion does not work
		if key.char == '1':
			global start
			start = True
		if key.char == '2':
			global stop
			stop = True
		if key.char == '3':
			global humming
			humming = True
			if recorder:
				recorder.start_hum()
			print('> start humming')
	except:
		tmp = 0 # to catch hits on shift etc.
	return True # continue listening

# On release event
def on_release(key) -> bool:
	if key == keyboard.Key.esc:
		global escape
		escape = True
		return False # stop listening
	try:
		if key.char == '3':
				global humming
				humming = False
				if recorder:
					recorder.stop_hum()
				print('> stop humming')
	except:
		tmp = 0 # to catch hits on shift etc.
	return True # continue listening

# Initialize keyboard listener
listener = keyboard.Listener(
	on_press=on_press,
	on_release=on_release)
listener.start()

# Main loop
while True:

	# Stop
	if recorder and (stop or escape):
		recorder.stop()
		recorder = None

	# Escape
	if escape:
		break

	# Start
	if not recorder and start:
		recorder = Recorder(name=name + '_' + str(count))
		if humming:
			recorder.start_hum()
		count += 1

	# Reset triggers and sleep
	start = False
	stop = False
	time.sleep(0.05)