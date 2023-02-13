from pathlib import Path
from typing import Optional

import pandas as pd
from opensoundscape.torch.models.cnn import CNN, InceptionV3, load_model

from sfw_brood.model import ModelTrainer
from sfw_brood.preprocessing import balance_data, group_ages
from .model import SnowfinchBroodCNN
from .util import cleanup
from .validator import CNNValidator


class CNNTrainer(ModelTrainer):
	def __init__(
			self, data_path: str, audio_path: str, work_dir: str,
			sample_duration_sec: float, rec_split: dict,
			cnn_arch: str, n_epochs: int, n_workers = 12, batch_size = 100, learn_rate = 0.001,
			target_label: Optional[str] = None, remove_silence: bool = True,
			age_groups: Optional[list[tuple[int, int]]] = None, samples_per_class = 'min'
	):
		self.cnn_arch = cnn_arch
		self.n_epochs = n_epochs
		self.n_workers = n_workers
		self.batch_size = batch_size
		self.learn_rate = learn_rate
		self.target_labels = ['feeding', 'contact'] if target_label is None else [target_label]
		self.remove_silence = remove_silence
		self.sample_duration_sec = sample_duration_sec
		self.work_dir = Path(work_dir)
		self.data_path = data_path
		self.data_split = rec_split
		self.samples_per_class = samples_per_class

		bs_data = pd.read_csv(f'{data_path}/brood-size.csv')
		bs_data = bs_data[~bs_data['is_silence'] & (bs_data['event'].isin(self.target_labels))]

		ba_data = pd.read_csv(f'{data_path}/brood-age.csv')
		ba_data = ba_data[~ba_data['is_silence'] & (ba_data['event'].isin(self.target_labels))]
		if age_groups:
			ba_data = group_ages(ba_data, groups = age_groups)

		self.bs_train_data = self.__select_recordings__(bs_data, rec_split['BS']['train'], audio_path)
		self.bs_val_data = self.__select_recordings__(bs_data, rec_split['BS']['validation'], audio_path)
		self.bs_test_data = self.__select_recordings__(bs_data, rec_split['BS']['test'], audio_path)
		print(f'\nSize data:')
		print(f'\ttrain: {self.bs_train_data.shape}')
		print(f'\tvalidation: {self.bs_val_data.shape}')
		print(f'\ttest: {self.bs_test_data.shape}')

		self.ba_train_data = self.__select_recordings__(ba_data, rec_split['BA']['train'], audio_path)
		self.ba_val_data = self.__select_recordings__(ba_data, rec_split['BA']['validation'], audio_path)
		self.ba_test_data = self.__select_recordings__(ba_data, rec_split['BA']['test'], audio_path)
		print(f'\nAge data:')
		print(f'\ttrain: {self.ba_train_data.shape}')
		print(f'\tvalidation: {self.ba_val_data.shape}')
		print(f'\ttest: {self.ba_test_data.shape}')

	def __enter__(self):
		self.work_dir.mkdir(parents = True, exist_ok = True)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		cleanup(self.work_dir)

	def train_model_for_size(self, out_dir: str):
		return self.__do_training__(
			self.bs_train_data, self.bs_test_data, self.bs_val_data, out_dir, label = 'brood size'
		)

	def train_model_for_age(self, out_dir: str):
		return self.__do_training__(
			self.ba_train_data, self.ba_test_data, self.ba_val_data, out_dir, label = 'brood age'
		)

	def __select_recordings__(self, data: pd.DataFrame, recordings: list[str], audio_path: str) -> pd.DataFrame:
		def extract_rec_name(file_name: str) -> str:
			end_idx = file_name.rindex('__')
			return file_name[:end_idx]

		selection_df = data[data['file'].apply(extract_rec_name).isin(recordings)]
		selection_df['file'] = audio_path + '/' + selection_df['file']
		selection_df = selection_df.set_index('file')

		classes = [str(cls) for cls in sorted(selection_df['class'].unique())]
		return balance_data(selection_df[classes], classes, samples_per_class = self.samples_per_class)

	def __do_training__(
			self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame],
			validation_data: Optional[pd.DataFrame], out_dir: str, label: str
	):
		if train_data.shape[0] == 0:
			print('No training data available')
			return

		print('Training CNN...')
		cnn = self.__setup_cnn__(classes = train_data.columns)

		out_path = Path(out_dir)
		out_path.mkdir(parents = True, exist_ok = True)

		trained_model = self.__train_and_validate__(cnn, train_data, test_data, validation_data, out_dir, label)
		trained_model.serialize(f'{out_dir}/cnn.model')

	def __setup_cnn__(self, classes):
		if self.cnn_arch == 'inception_v3':
			cnn = InceptionV3(classes = classes, sample_duration = self.sample_duration_sec, single_target = True)
		else:
			cnn = CNN(
				architecture = self.cnn_arch,
				sample_duration = self.sample_duration_sec,
				classes = classes,
				single_target = True
			)
		cnn.optimizer_params['lr'] = self.learn_rate
		return cnn

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
				'learning_rate': cnn.optimizer_params['lr'],
				'architecture': self.cnn_arch,
				'train_epochs': trained_cnn.current_epoch,
				'data': self.data_path,
				'data_split': self.data_split,
				'sample_duration_sec': self.sample_duration_sec,
				'batch_size': self.batch_size,
				'events': self.target_labels
			}
		)

	def __train_and_validate__(
			self, cnn: CNN, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame],
			validation_data: Optional[pd.DataFrame], out_dir: str, label: str
	) -> SnowfinchBroodCNN:
		if test_data is None:
			return self.__train_cnn__(cnn, train_data, validation_data)

		trained_model = self.__train_cnn__(cnn, train_data, validation_data)

		validator = CNNValidator(test_data, label)
		accuracy = validator.validate(trained_model, output = out_dir)
		print(f'CNN accuracy: {accuracy}')

		return trained_model
