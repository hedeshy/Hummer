import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

# TODO: go over more than one file
# Inspiration: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

# Import metadata
meta = {}
with open('data/raphael.json') as f:
	meta = json.load(f)

print(meta)

# Test by import, make mono, export
x, sr = librosa.load('data/raphael.wav', sr=None) # no resampling
x_mono: np.ndarray = librosa.to_mono(x) # length of x_mono / sampling rate = duration in s
librosa.output.write_wav('output.wav', x_mono, sr, norm=False)

print(x_mono.shape)

# Plot wave
'''
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x_mono, sr=sr)
plt.show()
'''

# Split into pieces of certain length
length_s = x_mono.shape[0] / sr # length of clip in seconds
width_s = 1.0 # segment width
step_s = 0.1 # segment stepping
sample_count = width_s * sr

# Go over mono data and split into segments
segments = []
pos_s = 0.0
while pos_s + width_s <= length_s:
	start_idx = int(pos_s * sr)
	end_idx = int((pos_s + width_s) * sr) - 1
	y = x_mono[start_idx:end_idx]
	pos_s += step_s
	print(str(start_idx) + ', ' + str(end_idx))

	# Compute features
	rmse = librosa.feature.rms(y=y)[0]
	chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
	spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	zcr = librosa.feature.zero_crossing_rate(y)
	mfcc = librosa.feature.mfcc(y=y, sr=sr)

	# Fill feature vector
	seg = []
	seg.append(np.mean(chroma_stft))
	seg.append(np.mean(rmse))
	seg.append(np.mean(spec_cent))
	seg.append(np.mean(spec_bw))
	seg.append(np.mean(rolloff))
	seg.append(np.mean(zcr))
	for e in mfcc: # 20 features
		seg.append(np.mean(e))

	# Store feature vector and label
	segments.append((seg, True))
	print(len(seg))

	# TODO: check in meta data about humming or no humming is happening at this segment
	# three states: neutral, negative, positive about humming


# TODO
# 1. Split into segments
# 2. Compute features on each segment and store corresponding label
# 3. Train random forest or ANN

# TODO: why is the first layer taking 256 values? There are only 26 features
'''
data = pd.read_csv('dataset.csv')
data.head()# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)#Encoding the Labels
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)#Scaling the Feature columns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))#Dividing data into training and Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


classifier = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=128)

'''