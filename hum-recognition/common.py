import numpy as np
import librosa
from typing import List
from collections import OrderedDict

def compute_feature_vector(y: np.array, sr: int) -> List[float]:
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

labels: List[str] = 'none', 'question', 'positive', 'negative', 'continuous'
def label_int(label: str) -> int:
	return labels.index(label)