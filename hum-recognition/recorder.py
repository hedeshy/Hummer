print('Recorder')

# Intead of global variables, make class: https://stackoverflow.com/questions/32705518/using-pyaudio-callback-methods-in-a-user-defined-class

import pyaudio
import wave
import time
import keyboard

CHANNELS: int = 2
RATE: int = 44100
FORMAT: int = pyaudio.paInt16
WAVE_OUTPUT_FILENAME: str = 'file.wav'

recorded_frames: bytes = bytes()
stop_callback: bool = False

p: pyaudio.PyAudio = pyaudio.PyAudio()

def callback(in_data: bytes, frame_count: int, time_info: dict, status: int) -> (bytes, int):
	global recorded_frames
	recorded_frames = recorded_frames + in_data

	global stop_callback
	if stop_callback:
		callback_flag = pyaudio.paComplete
	else:
		callback_flag = pyaudio.paContinue

	return (None, callback_flag)

stream = p.open(format=FORMAT,
				channels=CHANNELS,
				rate=RATE,
				input=True,
				stream_callback=callback)

stream.start_stream()

while stream.is_active():
	if keyboard.is_pressed('q'):
		stop_callback = True
	time.sleep(0.1)

stream.stop_stream()
stream.close()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(p.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(recorded_frames)
waveFile.close()

p.terminate()