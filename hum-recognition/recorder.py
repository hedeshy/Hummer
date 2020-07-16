print('Welcome to the Recorder')
print('esc: exit, 1: start, 2: stop')

# Overview about parameters in pyaudio: https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio
# TODO:
# - record and store labels

import pyaudio
import wave
import time
from pynput import keyboard
from typing import List

class Recorder:
	
	def _record_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> (bytes, int):
		self._recorded_frames.append(in_data)
		return (None, pyaudio.paContinue)

	def stop(self) -> None:

		self._stream.stop_stream()
		self._stream.close()

		# Store as wave
		with wave.open(self._name + '.wav', 'wb') as w:
			w.setnchannels(self._channels)
			w.setsampwidth(self._pa.get_sample_size(self._format))
			w.setframerate(self._rate)
			w.writeframes(b''.join(self._recorded_frames))
			w.close()

		self._pa.terminate()

		print('> stop recording')

	def __init__(self, name: str):

		# Initialize members
		self._name = name
		self._channels: int = 2
		self._rate: int = 44100
		self._format: int = pyaudio.paInt16
		self._pa: pyaudio.PyAudio = pyaudio.PyAudio()
		self._recorded_frames: List[bytes] = []
	
		# Create stream
		self._stream = self._pa.open(
				format=self._format,
				channels=self._channels,
				rate=self._rate,
				input=True,
				stream_callback=self._record_callback)
		self._stream.start_stream()

		print('> start recording')

# Variables
recorder: Recorder = None
count: int = 0
start: bool = False
stop: bool = False
escape: bool = False
name: str = 'raphael' # TODO: provide via parameter

# On key press event
def on_press(key) -> bool:
	try: # if special key, 'char' conversion does not work
		if key.char == '1':
			global start
			start = True
		if key.char == '2':
			global stop
			stop = True
	except:
		tmp = 0 # to catch hits on shift etc.
	return True # continue listening

# On release event
def on_release(key) -> bool:
	if key == keyboard.Key.esc:
		global escape
		escape = True
		return False # stop listening
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
		count += 1

	# Reset triggers and sleep
	start = False
	stop = False
	time.sleep(0.05)