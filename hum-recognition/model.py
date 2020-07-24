import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from typing import Tuple
from typing import Set
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from joblib import dump
from os import listdir
from os.path import isfile, join, splitext

# Inspiration: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

DATA_PATH = r'./data'
SEGMENT_WIDTH_S: float = 0.5
SEGMENT_STEP_S: float = 0.1
RATIO_OF_HUM: float = 0.5 # at least 50% of segment must contain humming to be labeled 'True'

def compute_overlap(a, b):
	return max(0, min(a[1], b[1]) - max(a[0], b[0]))

# Get dataset
files: Set[str] = set([splitext(f)[0] for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))])
print(files)

# Import dataset
data: List[List[float]] = []
target: List[int] = [] # right now: 0 or 1

# Go over files
for f in files:

	print('Processing: ' + f)

	# Import sound
	x, sr = librosa.load('data/'+ f + '.wav', sr=None) # no resampling
	x_mono: np.ndarray = librosa.to_mono(x) # length of x_mono / sampling rate = duration in s
	length_s: float = x_mono.shape[0] / sr # length of clip in seconds
	# librosa.output.write_wav('output.wav', x_mono, sr, norm=False)

	# print(x_mono.shape)

	# Plot wave
	# plt.figure(figsize=(14, 5))
	# librosa.display.waveplot(x_mono, sr=sr)
	# plt.show()

	# Import metadata
	meta: dict = {}
	with open('data/'+ f + '.json') as f:
		meta = json.load(f)

	# print(meta['hums'])

	# Collect hums into starts and ends
	hum_starts_ms: List[int] = []
	hum_ends_ms: List[int] = []
	for ts, state in meta['hums']:
		if state == 'start':
			hum_starts_ms.append(ts)
		else:
			hum_ends_ms.append(ts)
	hum_starts_ms.sort()
	hum_ends_ms.sort()

	# Go over hums and make pairs of start and end (to be checked by later segments for intersection)
	hums_ms: List[Tuple[int,int]] = []
	if hum_starts_ms and hum_ends_ms:

		# Check if there is a end before the first start (humming was going on at start of recording)
		if hum_ends_ms[0] < hum_starts_ms[0]:
			hum_starts_ms.insert(0,0)

		for start_ms in hum_starts_ms:
			success: bool = False
			for end_ms in hum_ends_ms:
				if end_ms > start_ms:
					success = True
					hums_ms.append((start_ms,end_ms))
					break
			if not success:
				hums_ms.append(start_ms, int(length_s*1000))

	# Go over mono data and split into segments
	pos_s = 0.0
	while pos_s + SEGMENT_WIDTH_S <= length_s:
		
		# Get audio segment
		start_idx: int = int(pos_s * sr)
		end_idx: int = int((pos_s + SEGMENT_WIDTH_S) * sr) - 1
		y: np.array = x_mono[start_idx:end_idx]

		# Compute label
		label: int = 0 # no humming
		start_ms: int = int(1000 * (start_idx / sr))
		end_ms: int = int(1000 * (end_idx / sr))
		overlap: int = 0
		for hum_ms in hums_ms:
			overlap += compute_overlap((start_ms, end_ms),hum_ms)
		if float(overlap) / (SEGMENT_WIDTH_S*1000) >= RATIO_OF_HUM:
			label = 1 # humming
		target.append(label)

		# Prepare next iteration
		pos_s += SEGMENT_STEP_S
		# print(str(start_idx) + ', ' + str(end_idx) + ': ' + str(label))
		print('.', end='', flush=True)

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
		data.append(seg)
print()

# Convert dataset to numpy arrays
data: np.array = np.array(data)
target: np.array = np.array(target)

# Split training and test data; TODO: optionally, use entire dataset to train classifier
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
sm = SMOTE(random_state=42, sampling_strategy='not majority')
X_train, y_train = sm.fit_resample(X_train, y_train)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest
forest = RandomForestClassifier(
	random_state=42,
	n_estimators=25,
	class_weight=None, # 'balanced', 'balanced_subsample'
	criterion='entropy', # 'gini'
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1)
forest.fit(X_train, y_train)

# Predict on test set and print report
y_pred = forest.predict(X_test).astype(int)
print(classification_report(y_test, y_pred, target_names=['no humming', 'humming']))

# Store model to use it at interaction
dump(forest, 'model.joblib') 