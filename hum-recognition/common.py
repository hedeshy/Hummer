import numpy as np
import librosa
from typing import List
from collections import OrderedDict

SHARED_PATH: str = r'./shared'
BIN_COUNT: int = 5
SEGMENT_WIDTH_SEC = 0.5 # limits the "length" of humming

def compute_feature_vector(y: np.array, sr: int) -> List[float]:

	'''
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

	return seg

	'''

	# Idea: Subdivide window into bins and store freq amplitudes as 
	
	total_count = y.shape[0]
	window_count = int(total_count / BIN_COUNT)
	# Assumption: total_count % bin_count = 0

	# Go over bins and collect amplitudes and phase
	fts = np.array([], dtype=np.float32)
	for i in range(0, BIN_COUNT):
		start_idx: int = i * window_count
		end_idx: int = (i + 1) * window_count
		b: np.array = y[start_idx:end_idx]
		rmse = librosa.feature.rms(y=b)[0]
		# chroma_stft = librosa.feature.chroma_stft(y=b, sr=sr)
		# chroma_cqt = librosa.feature.chroma_cqt(y=b, sr=sr)
		spec_cent = librosa.feature.spectral_centroid(y=b, sr=sr)
		spec_bw = librosa.feature.spectral_bandwidth(y=b, sr=sr)
		rolloff = librosa.feature.spectral_rolloff(y=b, sr=sr)
		contrast = librosa.feature.spectral_contrast(y=b, sr=sr)
		flatness = librosa.feature.spectral_flatness(y=b)
		zcr = librosa.feature.zero_crossing_rate(b)
		mfcc = librosa.feature.mfcc(y=b, sr=sr)

		# Fill feature vector
		seg = []
		# seg.append(np.mean(chroma_stft))
		# seg.append(np.mean(chroma_cqt))
		seg.append(np.mean(rmse))
		seg.append(np.mean(spec_cent))
		seg.append(np.mean(spec_bw))
		seg.append(np.mean(rolloff))
		seg.append(np.mean(contrast))
		seg.append(np.mean(flatness))
		seg.append(np.mean(zcr))
		for e in mfcc: # 20 features
			seg.append(np.mean(e))
		fts = np.append(fts, seg)

		# A = np.fft.fft(signal)
		# amps = np.abs(A)
		# phas = np.angle(A)
		# fts = np.append(fts, amps)
		# fts = np.append(fts, phas)

	return list(fts)

labels: List[str] = 'none', 'positive', 'negative', 'continuous'
def label_int(label: str) -> int:
	return labels.index(label)