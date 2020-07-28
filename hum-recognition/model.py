import librosa
import librosa.display
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from typing import Tuple
from typing import Set
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn import decomposition
from imblearn.over_sampling import SMOTE
from joblib import dump
from os import listdir
from os.path import isfile, join, splitext
import common
from collections import defaultdict
from collections import OrderedDict

# Inspiration: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

DATA_PATH: str = r'./data'
TMP_PATH: str = r'./tmp'
SEGMENT_WIDTH_S: float = 1.0
SEGMENT_STEP_S: float = 0.1
RATIO_OF_HUM: float = 0.5 # at least 50% of segment must contain humming to be labeled as not 'none'
EVALUATION_FOLDS: int = 5

def compute_overlap(a, b):
	return max(0, min(a[1], b[1]) - max(a[0], b[0]))

# Get dataset
files: Set[str] = set([splitext(f)[0] for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))])

# Import dataset
data: List[List[float]] = []
target: List[int] = [] # right now: 0 or 1

# Go over files
for f in files:

	print('Processing: ' + f)

	# Import sound
	x, sr = librosa.load('data/'+ f + '.wav', sr=None, dtype=np.float32) # no resampling
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

	# Map from class name to class number
	class_name_to_int: OrderedDict = OrderedDict([ ('none', 0), ('question', 1), ('positive', 2), ('negative', 3), ('continuous', 4)])

	# Collect hums into starts and ends
	hum_starts_ms: List[Tuple[int,int]] = [] # tuple of start and class
	hum_ends_ms: List[int] = []
	for ts, state in meta['hums']:
		if state == 'end':
			hum_ends_ms.append(ts)
		else:
			hum_starts_ms.append((ts, class_name_to_int[state[6:]]))
	# Above assumes that states are ordered correctly in the json (should be the case)

	# Go over hums and make triples of start and end and class integer (to be checked by later segments for intersection)
	hums_ms: List[Tuple[int,int,int]] = []
	if hum_starts_ms and hum_ends_ms:

		# Check if there is an end before the first start and remove that end
		# if hum_ends_ms[0] < hum_starts_ms[0]:
		# 	hum_starts_ms.insert(0,0) # Issue: not sure which class of humming is ended here / no info in the data

		for start_ms_class in hum_starts_ms:
			success: bool = False # succes in finding an end
			for end_ms in hum_ends_ms:
				if end_ms > start_ms_class[0]:
					success = True
					hums_ms.append((start_ms_class[0],end_ms,start_ms_class[1]))
					break
			if not success:
				hums_ms.append(start_ms_class[0], int(length_s*1000), start_ms_class[1]) # just assume the end of the recording as end

	# Go over mono data and split into segments
	pos_s = 0.0
	i: int = 0
	while pos_s + SEGMENT_WIDTH_S <= length_s:
		
		# Get audio segment
		start_idx: int = int(pos_s * sr)
		end_idx: int = int((pos_s + SEGMENT_WIDTH_S) * sr) - 1
		y: np.array = x_mono[start_idx:end_idx]

		# Compute feature vector
		data.append(common.compute_feature_vector(y, sr))

		# Compute label
		label: int = 0 # no humming
		start_ms: int = int(1000 * (start_idx / sr))
		end_ms: int = int(1000 * (end_idx / sr))
		overlap: defaultdict = defaultdict(int)
		for hum_ms in hums_ms:
			overlap[str(hum_ms[2])] += compute_overlap((start_ms, end_ms),(hum_ms[0],hum_ms[1]))
		class_overlap: int = 0
		class_int: int = 0
		for key, value in overlap.items(): # get most overlapping humming class
			if value > class_overlap:
				class_overlap = value
				class_int = key
		if float(class_overlap) / (SEGMENT_WIDTH_S*1000) >= RATIO_OF_HUM:
			label = class_int # humming
		target.append(label)

		# Prepare next iteration
		pos_s += SEGMENT_STEP_S
		# print(str(start_idx) + ', ' + str(end_idx) + ': ' + str(label))
		print(str(label), end='', flush=True)
		i += 1

	print()

# Convert dataset to numpy arrays
data: np.array = np.array(data)
target: np.array = np.array(target)

# TODO: PCA to reduce dimensions (results get worse)
# pca = decomposition.PCA(n_components=5)
# pca.fit(data)
# data = pca.transform(data) # TODO: store PCA such that at recognition the same PCA can be applied

# Resample the dataset to remove imbalance TODO: reintegrate (maybe not enough sample data) and remember to set class_weight attribute in forest
# sm = SMOTE(random_state=42, sampling_strategy='not majority')
# data, target = sm.fit_resample(data, target)

# Random Forest
clf = RandomForestClassifier(
	random_state=42,
	n_estimators=100,
	class_weight='balanced', # None, 'balanced_subsample'
	criterion='entropy', # 'gini'
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1)

# Perform cross validation and report results
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(clf, data, target, scoring=scoring, cv=EVALUATION_FOLDS)
print('Recall (Macro, k=' + str(EVALUATION_FOLDS) + '): ' + str(np.mean(scores['test_recall_macro'])))
print('Precision (Macro, k=' + str(EVALUATION_FOLDS) + '): ' + str(np.mean(scores['test_precision_macro'])))

# Scale data (TODO: also scale data for the cross validation, but use a pipeline instead)
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
# Note: results without scaling look better... (scaling not required for random forest)

# Store model trained by entire data to be used for interaction
clf.fit(data, target)
dump(clf, 'model.joblib')
print(classification_report(target, clf.predict(data), target_names=class_name_to_int.keys()))

# TODO: decide which features are important (maybe apply PCA)
importances = clf.feature_importances_
print(importances)