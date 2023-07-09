from typing import List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def prepare_sampled_datasets(
		data: pd.DataFrame, n_folds: int, x_features: list, y_col = 'label', scaler = None
) -> Tuple[List[Tuple[pd.DataFrame, pd.Series]], Optional[Any]]:
	if scaler is not None:
		scaler.fit(data[x_features])

	train_datasets = []
	for _ in range(n_folds):
		train_data = sample_evenly_by_label(data)
		x_train = scaler.transform(train_data[x_features])
		y_train = train_data[y_col]
		train_datasets.append((x_train, y_train))

	return train_datasets, scaler


def sample_evenly_by_label(data: pd.DataFrame) -> pd.DataFrame:
	sample_size = data['size'].value_counts().min()
	return data.groupby(['size']).sample(n = sample_size)


class EnsemleClassifier:
	def __init__(self, n_models: int, svm = True, rfc = False, mlp = False, bayes = False, name = None):
		self.n_models = n_models
		self.labels = None
		self.scaler = StandardScaler()
		self.models = []
		self.name = name

		if svm:
			for _ in range(n_models):
				self.models.append(SVC(probability = True))
		if rfc:
			for _ in range(n_models):
				self.models.append(RandomForestClassifier(n_jobs = 12))
		if mlp:
			for _ in range(n_models):
				self.models.append(MLPClassifier())
		if bayes:
			for _ in range(n_models):
				self.models.append(GaussianNB())

	def fit(self, data: pd.DataFrame, x_features: List[str], y_col = 'label'):
		self.labels = sorted(data[y_col].unique())
		train_datasets, self.scaler = prepare_sampled_datasets(
			data, n_folds = self.n_models, x_features = x_features,
			y_col = y_col, scaler = self.scaler
		)

		for i, model in enumerate(tqdm(self.models, total = len(self.models))):
			x_train, y_train = train_datasets[i % self.n_models]
			model.fit(x_train, y_train)

	def predict(self, x: pd.DataFrame, voting = 'hard') -> np.ndarray:
		if not self.labels:
			raise RuntimeError('Fit the model first!')

		x_scaled = self.scaler.transform(x)

		if voting == 'soft':
			preds = pd.DataFrame(data = np.zeros((x.shape[0], len(self.labels))), columns = self.labels)
			for model in self.models:
				model_pred = pd.DataFrame(data = model.predict_proba(x_scaled), columns = model.classes_)
				preds += model_pred
			return preds.idxmax(axis = 1).values
		else:
			preds = None
			for model in self.models:
				model_pred = model.predict(x_scaled).reshape(-1, 1)
				preds = model_pred if preds is None else np.append(preds, model_pred, axis = 1)
			ensemble_preds = np.apply_along_axis(__most_frequent_element__, axis = 1, arr = preds)
			return ensemble_preds

	# def serialize(self, path: Path):
	# 	model_paths = []
	#
	# 	for i, model in enumerate(self.models):
	# 		model_path = path.parent.joinpath(f'model_{i}')
	# 		with open(model_path, mode = 'wb') as model_file:
	# 			pickle.dump(model, model_file)
	#
	# 	with zipfile.ZipFile(path, mode = 'w') as archive:
	# 		for model_path in model_paths:
	# 			archive.write(model_path, model_path.relative_to(path.parent))
	# 			model_path.unlink()


def __most_frequent_element__(x: np.ndarray) -> np.ndarray:
	return np.array(pd.Series(x).mode()[0], dtype = '<U3')
