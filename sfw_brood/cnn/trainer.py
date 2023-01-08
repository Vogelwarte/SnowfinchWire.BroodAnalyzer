from pathlib import Path

import pandas as pd
from opensoundscape.torch.models.cnn import CNN, load_model
from sklearn.model_selection import train_test_split

from sfw_brood.model import ModelTrainer
from sfw_brood.preprocessing import SnowfinchDataset, prepare_training
from .model import SnowfinchBroodCNN
from .util import cleanup
from .validator import CNNValidator


class CNNTrainer(ModelTrainer):
	def __init__(self, dataset: SnowfinchDataset, sample_duration_sec: float, work_dir: str):
		self.dataset = dataset
		self.sample_duration_sec = sample_duration_sec
		self.work_dir = work_dir

	def train_model_for_size(self, out_dir: str, validate = True):
		return self.__do_training__(self.bs_train_data, out_dir, validate)

	def train_model_for_age(self, out_dir: str, validate = True):
		return self.__do_training__(self.ba_train_data, out_dir, validate)

	def __enter__(self):
		self.bs_train_data, self.ba_train_data = prepare_training(
			self.dataset, self.work_dir, self.sample_duration_sec, overlap_sec = 0.0
		)

		print(f'Brood size training data shape: {self.bs_train_data.shape}')
		print(f'Brood age training data shape: {self.ba_train_data.shape}')

		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		print('Cleaning up')
		cleanup(Path(self.work_dir))

	def __do_training__(self, data: pd.DataFrame, out_dir: str, validate: bool):
		if data.shape[0] == 0:
			print('No training data available')
			return

		cnn = CNN(
			architecture = 'resnet18',
			sample_duration = self.sample_duration_sec,
			classes = data.columns,
			single_target = True
		)

		print('Training CNN...')

		out_path = Path(out_dir)
		out_path.mkdir(parents = True, exist_ok = True)

		trained_model = self.__train_and_validate__(cnn, data, out_dir) if validate else self.__train_cnn__(cnn, data)
		trained_model.serialize(f'{out_dir}/cnn.model')

	def __train_cnn__(self, cnn: CNN, train_data: pd.DataFrame) -> SnowfinchBroodCNN:
		train_df, validation_df = train_test_split(train_data, test_size = 0.15)
		cnn.train(
			train_df, validation_df, epochs = 2, batch_size = 100,
			save_path = f'{self.work_dir}/models', num_workers = 12
		)
		return SnowfinchBroodCNN(trained_cnn = load_model(f'{self.work_dir}/models/best.model'))

	def __train_and_validate__(self, cnn: CNN, data: pd.DataFrame, out_dir: str) -> SnowfinchBroodCNN:
		train_data, test_data = train_test_split(data, test_size = 0.2)
		trained_model = self.__train_cnn__(cnn, train_data)

		validator = CNNValidator(test_data)
		accuracy = validator.validate(trained_model, output = out_dir)
		print(f'CNN accuracy: {accuracy}')

		return trained_model
