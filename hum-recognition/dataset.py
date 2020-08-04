import common

# Audio processing
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Typing
from typing import List
from typing import Tuple
from typing import Set
from typing import NamedTuple

# Standard
from os import listdir
from os.path import isfile, join, splitext
from collections import defaultdict
from collections import namedtuple
import json
from joblib import dump

# Parameters
DATA_PATH: str = r'./data'
TMP_PATH: str = r'./tmp'
SEGMENT_WIDTH_S: float = 0.5
SEGMENT_STEP_S: float = 0.01
RATIO_OF_HUM: float = 0.5 # at least 50% of segment must contain humming to be labeled as not 'none'

# Computes overlap of two intervals
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

	# Collect hums into starts and ends
	hum_starts_ms: List[int] = []
	hum_ends_ms: List[int] = []
	hum_labels: List[int] = []
	for ts, event in meta['hums']:
		if event == 'end':
			hum_ends_ms.append(ts)
		else:
			hum_starts_ms.append(ts)
			hum_labels.append(common.label_int(event[6:]))
	# Above assumes that events are ordered correctly in the json (should be the case)

	# Go over hums and make triples of start and end and label indicator
	class Hum(NamedTuple):
		start_ms: int
		end_ms: int
		label: int

	hums: List[Hum] = []
	if hum_starts_ms and hum_ends_ms and hum_labels:

		# Check if there is an end before the first start and remove that end
		# if hum_ends_ms[0] < hum_starts_ms[0]:
		# 	hum_starts_ms.insert(0,0) # Issue: not sure which class of humming is ended here / no info in the data

		# Iterate over starts and find corresponding ends
		for i in range(len(hum_starts_ms)):
			start_ms = hum_starts_ms[i]
			success: bool = False # succes in finding an end
			for end_ms in hum_ends_ms:
				if end_ms > start_ms:
					success = True
					hums.append(Hum(start_ms, end_ms, hum_labels[i]))
					break
			if not success:
				hums.append(Hum(start_ms, int(length_s*1000), hum_labels[i])) # just assume the end of the recording as end

	# Go over mono data and split into segments
	pos_s = 0.0
	i: int = 0
	window_sample_count: int = int(SEGMENT_WIDTH_S * sr)
	while pos_s + SEGMENT_WIDTH_S <= length_s:
		
		# Get audio segment
		start_idx: int = int(pos_s * sr)
		end_idx: int = start_idx + window_sample_count
		y: np.array = x_mono[start_idx:end_idx]

		# Compute feature vector
		data.append(common.compute_feature_vector(y, sr))

		# Compute label
		label: int = 0
		start_ms: int = int(1000 * (start_idx / sr))
		end_ms: int = int(1000 * (end_idx / sr))
		overlaps: defaultdict = defaultdict(int) # start for every counting at zero
		for hum in hums:
			overlaps[str(hum.label)] += compute_overlap((start_ms, end_ms), (hum.start_ms, hum.end_ms))
		overlap: int = 0
		potential_label: int = 0
		for key, value in overlaps.items(): # get most overlapping humming label
			if value > overlap:
				overlap = value
				potential_label = int(key)
		if float(overlap) / (SEGMENT_WIDTH_S*1000) >= RATIO_OF_HUM:
			label = potential_label # humming
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

# Store dataset
dump(data, common.SHARED_PATH + '/data.joblib')
dump(target, common.SHARED_PATH + '/target.joblib')

print('Stored dataset. Shape: ' + str(data.shape))