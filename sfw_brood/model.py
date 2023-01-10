import json
import zipfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd


class ModelType(Enum):
	CNN = 0


class SnowfinchBroodClassifier(ABC):
	def __init__(self, model_type: ModelType, model_info: Optional[dict] = None):
		self.model_type = model_type
		self.model_info = model_info

	@abstractmethod
	def predict(self, recording_paths: list[str]) -> pd.DataFrame:
		pass

	def serialize(self, path: str):
		save_path = Path(path)
		save_path.parent.mkdir(parents = True, exist_ok = True)

		meta_data = { 'type': self.model_type.name }
		if self.model_info:
			meta_data.update(self.model_info)

		meta_path = save_path.parent.joinpath('meta.json')
		with open(meta_path, mode = 'wt') as meta_file:
			json.dump(meta_data, meta_file, indent = 4)

		model_path = save_path.parent.joinpath('model')
		self._serialize_(str(model_path))

		with zipfile.ZipFile(f'{path}.zip', mode = 'w') as archive:
			archive.write(meta_path, meta_path.relative_to(save_path.parent))
			archive.write(model_path, model_path.relative_to(save_path.parent))

		meta_path.unlink()
		model_path.unlink()

	@abstractmethod
	def _serialize_(self, path: str):
		pass


class ModelLoader(ABC):
	def __init__(self, model_type: ModelType):
		self.model_type = model_type
		self.next: Optional[ModelLoader] = None

	def set_next(self, next_loader):
		self.next = next_loader

	def load_model(self, path: str) -> SnowfinchBroodClassifier:
		with zipfile.ZipFile(path) as archive:
			with archive.open('meta.json', mode = 'r') as meta_file:
				meta_data = json.load(meta_file)
				model_type = meta_data['type']
				if model_type != self.model_type.name:
					if self.next:
						return self.next.load_model(path)
					else:
						raise Exception(f'Unsupported model type: {model_type}')

			archive.extract('model')

		model = self._deserialize_model_('model', meta_data)
		Path('model').unlink()

		print(meta_data)
		return model

	@abstractmethod
	def _deserialize_model_(self, path: str, meta_data: dict) -> SnowfinchBroodClassifier:
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
