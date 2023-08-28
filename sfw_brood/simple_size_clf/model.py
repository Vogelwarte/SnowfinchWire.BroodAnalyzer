import pickle
from typing import List

import pandas as pd

from sfw_brood.model import SnowfinchBroodClassifier, ModelLoader, ModelType
from sfw_brood.simple_size_clf.ensemble_clf import EnsemleClassifier


class SimpleBroodSizeClassifier(SnowfinchBroodClassifier):
	def __init__(self, trained_models: List[EnsemleClassifier], model_info: dict):
		super().__init__(ModelType.SIMPLE_SIZE_CLF, model_info)
		self.models = trained_models
		self.voting = model_info['voting']
		self.x_features = model_info['features']

	# assume that feeding_data has brood_id, datetime and x_features columns
	def predict(self, feeding_data: pd.DataFrame, n_workers = 0) -> pd.DataFrame:
		out_df = feeding_data[['brood_id', 'datetime']]
		for model in self.models:
			out_df[model.name] = model.predict(feeding_data[self.x_features], voting = self.voting)
		return out_df

	def _serialize_(self, path: str):
		with open(path, mode = 'wb') as out_file:
			pickle.dump(self.models, out_file)

	@property
	def classification_modes(self) -> List[str]:
		return [model.name for model in self.models]


class SimpleClfLoader(ModelLoader):
	def __init__(self):
		super().__init__(ModelType.SIMPLE_SIZE_CLF)

	def _deserialize_model_(self, path: str, meta_data: dict) -> SnowfinchBroodClassifier:
		with open(path, mode = 'rb') as in_file:
			classifiers = pickle.load(in_file)
		return SimpleBroodSizeClassifier(classifiers, meta_data)
