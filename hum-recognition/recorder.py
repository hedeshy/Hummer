print('Welcome to the Recorder')
print('esc: exit, 1: start, 2: stop, 3 (pressing): hum')

# TODO
# Use time from sounddevice instead of system time for timestamps

import sounddevice as sd
import wave
import time
from pynput import keyboard
from typing import List
import json

def get_ms() -> int:
	return time.time_ns() // 1000000

class Recorder:
	
	def _callback(self, indata, frame: int, time: dict, status: int) -> None:
		self._recorded_frames.append(bytes(indata)) # indata is _cffi_backend.buffer, lets just convert to bytes

	def start_hum(self) -> None:
		if self._hum_start < 0:
			self._hum_start = get_ms() - self._start_ms

	def stop_hum(self) -> None:
		if self._hum_start >= 0:
			self._humming.append((self._hum_start, get_ms() - self._start_ms))
			self._hum_start = -1

	def stop(self) -> None:

		self._stream.stop()
		self._stream.close()
		self.stop_hum() # under discussion whether makes sense (maybe not)
		print('> stop recording')

		# Store audio as wave
		with wave.open(self._name + '.wav', 'wb') as w:
			w.setnchannels(self._channels)
			w.setsampwidth(self._stream.samplesize)
			w.setframerate(self._rate)
			w.writeframes(b''.join(self._recorded_frames))
			w.close()

		# Store meta data as json
		# TODO: duration, date / time?, operation system
		meta: dict = {}
		meta['channels'] = self._channels
		meta['rate'] = self._rate
		meta['format'] = str(self._dtype)
		meta['hums'] = self._humming
		with open(self._name + '.json', 'w') as w:
			json.dump(meta, w)

		# Gets the number of bytes, similar to length
		# b = b''.join(self._recorded_frames)
		# print(sys.getsizeof(b)) # time: /2 (16 bit) /2 (steore) /44100 (samples per second)

	def __init__(self, name: str):

		# Initialize members
		self._name = name

		host_info: dict = sd.query_hostapis(index=None)[0]
		device_info: dict = sd.query_devices(device=host_info['default_input_device'])
		self._channels: int = int(device_info['max_input_channels'])
		self._rate: int = int(device_info['default_samplerate'])
		self._recorded_frames: List[bytes] = []
		self._dtype: str = 'int16'
		self._hum_start: int = -1 # -1 marks that no humming is going on
		self._humming: List[(int, int)] = [] # list of humming start / end tuples

		# Create stream
		self._stream = sd.RawInputStream(
				device=host_info['default_input_device'],
				dtype=self._dtype,
				channels=self._channels,
				samplerate=self._rate,
				callback=self._callback)
		self._stream.start()

		print('> start recording ' + device_info['name'])
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