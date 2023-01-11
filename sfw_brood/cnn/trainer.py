from pathlib import Path
from typing import Optional

import pandas as pd
from opensoundscape.torch.models.cnn import CNN, load_model
from sklearn.model_selection import train_test_split

from sfw_brood.model import ModelTrainer
from sfw_brood.preprocessing import SnowfinchDataset, prepare_training
from .model import SnowfinchBroodCNN
from .util import cleanup
from .validator import CNNValidator


class CNNTrainer(ModelTrainer):
	def __init__(
			self, train_dataset: SnowfinchDataset, work_dir: str,
			sample_duration_sec: float, cnn_arch: str, n_epochs: int,
			n_workers = 12, batch_size = 100, sample_overlap_sec = 0.0,
			test_dataset: Optional[SnowfinchDataset] = None
	):
		self.work_dir = work_dir
		self.train_dataset = train_dataset
		self.test_dataset = test_dataset
		self.sample_duration_sec = sample_duration_sec
		self.sample_overlap_sec = sample_overlap_sec
		self.cnn_arch = cnn_arch
		self.n_epochs = n_epochs
		self.n_workers = n_workers
		self.batch_size = batch_size

	def train_model_for_size(self, out_dir: str):
		return self.__do_training__(self.bs_train_data, self.bs_test_data, out_dir)

	def train_model_for_age(self, out_dir: str):
		return self.__do_training__(self.ba_train_data, self.ba_test_data, out_dir)

	def __enter__(self):
		self.bs_train_data, self.ba_train_data = prepare_training(
			self.train_dataset, self.work_dir, self.sample_duration_sec, overlap_sec = self.sample_overlap_sec
		)

		print(f'Brood size training data shape: {self.bs_train_data.shape}')
		print(f'Brood age training data shape: {self.ba_train_data.shape}')

		if self.test_dataset:
			self.bs_test_data, self.ba_test_data = prepare_training(
				self.test_dataset, self.work_dir, self.sample_duration_sec, overlap_sec = self.sample_overlap_sec
			)

			print(f'Brood size test data shape: {self.bs_test_data.shape}')
			print(f'Brood age test data shape: {self.ba_test_data.shape}')
		else:
			self.bs_test_data, self.ba_test_data = None, None

		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		print('Cleaning up')
		cleanup(Path(self.work_dir))

	def __do_training__(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame], out_dir: str):
		if train_data.shape[0] == 0:
			print('No training data available')
			return

		cnn = CNN(
			architecture = self.cnn_arch,
			sample_duration = self.sample_duration_sec,
			classes = train_data.columns,
			single_target = True
		)

		print('Training CNN...')

		out_path = Path(out_dir)
		out_path.mkdir(parents = True, exist_ok = True)

		trained_model = self.__train_and_validate__(cnn, train_data, test_data, out_dir)
		trained_model.serialize(f'{out_dir}/cnn.model')

	def __train_cnn__(
			self, cnn: CNN, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame]
	) -> SnowfinchBroodCNN:
		cnn.train(
			train_data, validation_df = validation_data, epochs = self.n_epochs, batch_size = self.batch_size,
			save_path = f'{self.work_dir}/models', num_workers = self.n_workers
		)

		trained_cnn = load_model(f'{self.work_dir}/models/best.model')
		return SnowfinchBroodCNN(
			trained_cnn,
			model_info = {
				'architecture': self.cnn_arch,
				'train_epochs': trained_cnn.current_epoch,
				'train_recordings': [rec.stem for rec in self.train_dataset.files],
				'test_recordings': [rec.stem for rec in self.test_dataset.files] if self.test_dataset else [],
				'sample_duration_sec': self.sample_duration_sec,
				'sample_overlap_sec': self.sample_overlap_sec,
				'batch_size': self.batch_size
			}
		)

	def __train_and_validate__(
			self, cnn: CNN, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame], out_dir: str
	) -> SnowfinchBroodCNN:
		if test_data is None:
			return self.__train_cnn__(cnn, train_data, None)

		test_df, validation_df = train_test_split(test_data, test_size = 0.3)
		trained_model = self.__train_cnn__(cnn, train_data, validation_df)

		validator = CNNValidator(test_df)
		accuracy = validator.validate(trained_model, output = out_dir)
		print(f'CNN accuracy: {accuracy}')

		return trained_model
