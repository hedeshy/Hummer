# Custom
import common

# Standard
import numpy as np
from joblib import load
from joblib import dump

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Typing
from typing import List

# K-fold evaluation
EVALUATION_FOLDS: int = 5

# Load data
data: np.array = load(common.SHARED_PATH + '/data.joblib')
target: np.array = load(common.SHARED_PATH + '/target.joblib')

# Below is an attempt to automatically determine optimal hyperparameters for model
'''

# Decode on parameters to tune
tuned_parameters = [{'pca__n_components': [20, 25, 30, 35]}, {'clf__n_estimators': [75, 100, 125]}]
estimators = [
	('pca', PCA()),
	('smote', SMOTE()),
	('clf', RandomForestClassifier())]

pipe = Pipeline(estimators)
# pipe.set_params(pca__n_components=25) # tuned
pipe.set_params(smote__random_state=42)
pipe.set_params(smote__sampling_strategy='not majority')
pipe.set_params(clf__min_samples_leaf=1)
pipe.set_params(clf__random_state=42)
# pipe.set_params(clf__n_estimators=100) # tuned
pipe.set_params(clf__class_weight=None)
pipe.set_params(clf__criterion='entropy')
pipe.set_params(clf__max_depth=None)
pipe.set_params(clf__min_samples_split=2)
pipe.set_params(clf__min_samples_leaf=1)

clf = GridSearchCV(
	pipe,
	tuned_parameters,
	scoring='precision_macro',
	cv=EVALUATION_FOLDS,
#	n_jobs=-1
	)
clf.fit(data, target)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()
# y_true, y_pred = y_test, clf.predict(X_test)
# print(classification_report(y_true, y_pred))
# print()

'''

# Perform PCA on entire dataset (makes it worse atm)
'''
pca = PCA(n_components=25)
pca.fit(data)
dump(pca, common.SHARED_PATH + '/pca.joblib')
print('> PCA stored')
data = pca.transform(data)
print('> Data shape after PCA: ' + str(data.shape))
'''

# Resample the dataset to remove imbalance
sm = SMOTE(random_state=42, sampling_strategy='not majority')
data, target = sm.fit_resample(data, target)
print('> Data shape after SMOTE: ' + str(data.shape))

# Create classifier
clf = RandomForestClassifier(
	random_state=42,
	n_estimators=150,
	class_weight=None, # 'balanced', 'balanced_subsample'
	criterion='entropy', # 'gini'
	max_depth=None,
	min_samples_split=2,
	min_samples_leaf=1)

# Perform cross-validation
print('> Performing cross-validation')
scores = cross_validate(
	clf,
	data,
	target,
	scoring=['precision_macro', 'recall_macro'],
	cv=EVALUATION_FOLDS,
	# n_jobs=-1,
	verbose=True)
print('> Recall (Macro, k=' + str(EVALUATION_FOLDS) + '): ' + str(np.mean(scores['test_recall_macro'])))
print('> Precision (Macro, k=' + str(EVALUATION_FOLDS) + '): ' + str(np.mean(scores['test_precision_macro'])))

# Scale data (not required for random forest)
# scaler = StandardScaler()
# scaler.fit(data)
# data = scaler.transform(data)
# Note: results without scaling look better... (scaling not required for random forest)

# Fit classifier on the entire dataset
clf.fit(data, target)
dump(clf, common.SHARED_PATH + '/model.joblib')
print('> Model stored')
# print(classification_report(target, clf.predict(data), target_names=common.labels))

''' # Does only work without PCA applied
# Print feature importances
print("Feature importances:")

# Collect importance of each feature across bins
feature_count = clf.feature_importances_.shape[0] / common.BIN_COUNT
print('feature_count: ' + str(feature_count))
feature_count = int(feature_count)
acc_imp: List[int] = [0.0] * feature_count
for i in range(0, feature_count):
	for j in range(0, common.BIN_COUNT):
		acc_imp[i] +=  clf.feature_importances_[i + (common.BIN_COUNT * j)]
for i in range(0, len(acc_imp)):
	print(str(i) + ': ' + str(100*(acc_imp[i])))
'''