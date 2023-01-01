import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from sfw_brood.common.preprocessing.io import validate_recording_data, load_recording_data
from sfw_brood.models.model import SnowfinchBroodClassifier
from sfw_brood.preprocessing import prepare_training_data, discover_training_data


class ModelTrainer(ABC):
	def __init__(self, train_data_path: str, sample_duration_sec: float, train_work_dir: str):
		self.train_data_path = train_data_path
		self.sample_duration_sec = sample_duration_sec
		self.train_work_dir = train_work_dir

	def __enter__(self):
		print(f'Collecting train data from directory {self.train_data_path}')
		self.bs_train_data, self.ba_train_data = self.__prepare_training__(data_dir = self.train_data_path)

		print(f'Brood size training data shape: {self.bs_train_data.shape}')
		print(f'Brood age training data shape: {self.ba_train_data.shape}')

		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if Path(self.train_work_dir).exists():
			for file in os.listdir(self.train_work_dir):
				os.remove(f'{self.train_work_dir}/{file}')
			os.rmdir(self.train_work_dir)

	def train_model_for_size(self) -> SnowfinchBroodClassifier:
		return self._do_training_(self.bs_train_data)

	def train_model_for_age(self) -> SnowfinchBroodClassifier:
		return self._do_training_(self.ba_train_data)

	@abstractmethod
	def _do_training_(self, train_data: pd.DataFrame) -> SnowfinchBroodClassifier:
		pass

	def __prepare_training__(self, data_dir: str):
		train_dataset = discover_training_data(data_dir)
		bs_train_df = pd.DataFrame()
		ba_train_df = pd.DataFrame()

		for file in train_dataset.files:
			print(f'Loading recording {file.stem}')
			recording = load_recording_data(Path(file))
			validate_recording_data(recording)

			bs_df, ba_df = prepare_training_data(
				recording, train_dataset.brood_sizes, train_dataset.brood_ages,
				work_dir = self.train_work_dir, slice_duration_sec = self.sample_duration_sec
			)

			bs_train_df = pd.concat([bs_train_df, bs_df])
			ba_train_df = pd.concat([ba_train_df, ba_df])

		return bs_train_df, ba_train_df
