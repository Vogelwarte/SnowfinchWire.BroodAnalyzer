from abc import ABC, abstractmethod

import pandas as pd


class SnowfinchBroodClassifier(ABC):
	@abstractmethod
	def predict(self, recording_paths: list[str]) -> pd.DataFrame:
		pass

	@abstractmethod
	def serialize(self, path: str):
		pass


class ModelTrainer(ABC):
	@abstractmethod
	def train_model_for_size(self, out_dir: str, validate = True):
		pass

	@abstractmethod
	def train_model_for_age(self, out_dir: str, validate = False):
		pass


class ModelValidator(ABC):
	@abstractmethod
	def validate(self, model: SnowfinchBroodClassifier, confusion_matrix_output = '') -> float:
		pass
