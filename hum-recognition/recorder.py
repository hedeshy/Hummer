print('Welcome to the Recorder')

import sounddevice as sd
from pynput import keyboard
from typing import List
import wave
import time
import json
import sys

def get_ms() -> int:
	return time.time_ns() // 1000000

class Recorder:

	def _callback(self, indata, frames: int, time: dict, status: int) -> None:
		current_ms: int = get_ms()
		if self._start_ms < 0: # looks like it takes some miliseconds until this gets called the first time. thus, regard that as start
			self._start_ms: int = current_ms # - (frames / self._rate) # subtract time for collecting the first samples
		self._end_ms = current_ms
		self._recorded_frames.append(bytes(indata)) # indata is _cffi_backend.buffer, lets just convert to bytes

	def start_hum(self, event: str) -> None:
		self._humming_events.append((get_ms() - self._start_ms, event))

	def stop_hum(self) -> None:
		self._humming_events.append((get_ms() - self._start_ms, 'end'))

	def stop(self) -> None:
		self._stream.stop()
		self._stream.close()
		print('> stop recording')
		# duration: int = self._end_ms - self._start_ms

		# Store audio as wave
		# byte_count: int = 0
		with wave.open(self._name + '.wav', 'wb') as w:
			w.setnchannels(self._channels)
			w.setsampwidth(self._stream.samplesize)
			w.setframerate(self._rate)
			frames: bytes = b''.join(self._recorded_frames)
			w.writeframes(frames)
			w.close()
			# byte_count = sys.getsizeof(frames)

		# Store meta data as json
		meta: dict = {}
		meta['channels'] = self._channels
		meta['rate'] = self._rate
		meta['format'] = str(self._dtype)
		meta['hums'] = self._humming_events
		# meta['duration [ms]'] = duration
		# meta['byte_count'] = byte_count
		with open(self._name + '.json', 'w') as w:
			json.dump(meta, w)

	def __init__(self, name: str):

		# Initialize members
		self._name = name
		host_info: dict = sd.query_hostapis(index=None)[0]
		device_info: dict = sd.query_devices(device=host_info['default_input_device'])
		self._channels: int = int(device_info['max_input_channels'])
		self._rate: int = int(device_info['default_samplerate'])
		self._recorded_frames: List[bytes] = []
		self._dtype: str = 'int16'
		self._humming_events: List[(int, str)] = []

		# Create stream
		self._stream = sd.RawInputStream(
				device=host_info['default_input_device'],
				dtype=self._dtype,
				blocksize=32,
				channels=self._channels,
				samplerate=self._rate,
				callback=self._callback)
		self._stream.start()
		self._start_ms: int = -1
		self._end_ms: int = -1

		print('> start recording ' + device_info['name'])

# Variables
name: str = ''
name_entered: bool = False
recorder: Recorder = None
count: int = 0
start: bool = False
stop: bool = False
escape: bool = False
humming: bool = False # true if there is any humming

# On key press event for name entering
def on_press_name(key) -> bool:
	try:
		if key == keyboard.Key.enter:
			global name_entered
			name_entered = True
			return False
		else:
			global name
			name += key.char
			print(name)
	except:
		pass
	return True # continue listening

# Naming mode
print('Enter your nickname (ASCII-only): <ENTER> done')

# Initialize name listener
listener = keyboard.Listener(
	on_press=on_press_name,
	suppress=True)
listener.start()

while not name_entered:
	time.sleep(0.05)
listener.stop()

# On key press event
def on_press(key) -> bool:
	try: # if special key, 'char' conversion does not work
		if key.char == '1':
			global start
			start = True
		if key.char == '2':
			global stop
			stop = True
		global humming
		if not humming and key.char == '3':
			if recorder:
				recorder.start_hum('start_question')
			print('> start quest. hum')
			humming = True
		if not humming and key.char == '4':
			if recorder:
				recorder.start_hum('start_positive')
			print('> start pos. hum')
			humming = True
		if not humming and key.char == '5':
			if recorder:
				recorder.start_hum('start_negative')
			print('> start neg. hum')
			humming = True
		if not humming and key.char == '6':
			if recorder:
				recorder.start_hum('start_continuous')
			print('> start cont. hum')
			humming = True
	except:
		pass
	return True # continue listening

# On release event
def on_release(key) -> bool:
	if key == keyboard.Key.esc:
		global escape
		escape = True
		return False # stop listening
	try:
		global humming
		if humming and key.char == '3' or key.char == '4' or key.char == '5' or key.char == '6':
			humming = False
			if humming <= 0 and recorder:
				print('> stop hum')
				recorder.stop_hum()
	except:
		pass
	return True # continue listening

# Recording mode
print('Record: <ESC> exit, <1> begin record, <2> stop record')
print('<3> quest. hum, <4> pos. hum, <5> neg. hum, <6> cont. hum')

# Initialize actual keyboard listener
listener = keyboard.Listener(
	on_press=on_press,
	on_release=on_release,
	suppress=True)
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
		count += 1

	# Reset triggers and sleep
	start = False
	stop = False
	time.sleep(0.05)