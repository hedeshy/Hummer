print('Recorder')

import pyaudio
import wave

# List all audio devices
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
	dev = pa.get_device_info_by_index(i)
	input_chn = dev.get('maxInputChannels', 0)
	if input_chn > 0:
		name = dev.get('name')
		rate = dev.get('defaultSampleRate')
		print("Index {i}: {name} (Max Channels {input_chn}, Default @ {rate} Hz)".format(
			i=i, name=name, input_chn=input_chn, rate=int(rate)

		))

# below just takes the default input device and it just blocks the thread
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
				rate=RATE, input=True,
				frames_per_buffer=CHUNK)
print("recording...")
frames = []

for i in range(0, int(RATE * (1.0 / CHUNK) * RECORD_SECONDS)):
	data = stream.read(CHUNK)
	frames.append(data)
print("finished recording")

# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()