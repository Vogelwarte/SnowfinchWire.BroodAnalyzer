import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from sfw_brood.common.preprocessing.io import load_recording_data, validate_recording_data
from sfw_brood.preprocessing import discover_training_data, prepare_training_data


class SnowfinchBroodClassifier(ABC):
	def __init__(self, train_work_dir: str, sample_duration_sec: float):
		self.train_work_dir = train_work_dir
		self.sample_duration_sec = sample_duration_sec

	def train(self, train_data_path: str):
		bs_train_data, ba_train_data = self.__prepare_training__(data_dir = train_data_path)

		print(f'Brood size training data shape: {bs_train_data.shape}')
		print(f'Brood age training data shape: {ba_train_data.shape}')

		self._do_training_((bs_train_data, ba_train_data))

		self.__cleanup_training__()

	@abstractmethod
	def test(self, test_data_path: str) -> float:
		pass

	@abstractmethod
	def predict(self) -> int:
		pass

	@abstractmethod
	def serialize(self, path: str):
		pass

	@abstractmethod
	def _do_training_(self, train_data: tuple[pd.DataFrame, pd.DataFrame]):
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

	def __cleanup_training__(self):
		if Path(self.train_work_dir).exists():
			for file in os.listdir(self.train_work_dir):
				os.remove(f'{self.train_work_dir}/{file}')
			os.rmdir(self.train_work_dir)
