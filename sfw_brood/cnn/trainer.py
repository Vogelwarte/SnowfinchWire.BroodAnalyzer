from pathlib import Path
from typing import Optional

import pandas as pd
from opensoundscape.torch.models.cnn import CNN, InceptionV3, load_model

from sfw_brood.model import ModelTrainer
from sfw_brood.preprocessing import balance_data, group_ages
from .model import SnowfinchBroodCNN
from .util import cleanup
from .validator import CNNValidator


def __format_data__(
		data: pd.DataFrame, audio_path: str, classes: list[str], cls_samples: Optional[str] = None
) -> pd.DataFrame:
	data['file'] = audio_path + '/' + data['file']
	data = data.set_index('file')
	if cls_samples:
		return balance_data(data[classes], classes, cls_samples)
	return data[classes]


def select_recordings(
		data: pd.DataFrame, audio_path: str, cls_samples: str, split_conf: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if 'classes' in split_conf.keys():
		classes = [str(cls) for cls in sorted(split_conf['classes'])]
		data = data[data['class'].astype('str').isin(classes)]
	else:
		classes = [str(cls) for cls in sorted(data['class'].unique())]

	selector = split_conf['selector']

	test_idx = data[selector].isin(split_conf['test'])
	test_df = __format_data__(data[test_idx], audio_path, classes)

	val_idx = data[selector].isin(split_conf['validation'])
	val_df = __format_data__(data[val_idx], audio_path, classes)

	train_df = __format_data__(data[~(test_idx | val_idx)], audio_path, classes, cls_samples)
	test_size = round(0.45 * len(train_df))
	val_size = round(0.2 * len(train_df))

	if test_size < len(test_df):
		test_df = test_df.sample(n = test_size)

	if val_size < len(val_df):
		val_df = val_df.sample(n = val_size)

	return train_df, val_df, test_df


class CNNTrainer(ModelTrainer):
	def __init__(
			self, data_path: str, audio_path: str, work_dir: str,
			sample_duration_sec: float, rec_split: dict,
			cnn_arch: str, n_epochs: int, n_workers = 12, batch_size = 100, learn_rate = 0.001,
			target_label: Optional[str] = None, remove_silence: bool = True,
			age_groups: Optional[list[tuple[float, float]]] = None, samples_per_class = 'min'
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

		bs_data = pd.read_csv(f'{data_path}/brood-size.csv', dtype = { 'is_silence': 'bool', 'class': 'int' })
		bs_data = bs_data[~bs_data['is_silence'] & (bs_data['event'].isin(self.target_labels))]

		ba_data = pd.read_csv(f'{data_path}/brood-age.csv', dtype = { 'is_silence': 'bool' })
		ba_data = ba_data[~ba_data['is_silence'] & (ba_data['event'].isin(self.target_labels))]
		if age_groups:
			ba_data = group_ages(ba_data, groups = age_groups)

		self.bs_train_data, self.bs_val_data, self.bs_test_data = select_recordings(
			bs_data, audio_path, self.samples_per_class, split_conf = rec_split['BS']
		)
		print(f'\nSize data:')
		print(f'\ttrain: {self.bs_train_data.shape}')
		print(f'\tvalidation: {self.bs_val_data.shape}')
		print(f'\ttest: {self.bs_test_data.shape}')

		self.ba_train_data, self.ba_val_data, self.ba_test_data = select_recordings(
			ba_data, audio_path, self.samples_per_class, split_conf = rec_split['BA']
		)
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

		print(f'Training done, testing model with output dir {out_dir}')

		validator = CNNValidator(test_data, label, n_workers = self.n_workers)
		accuracy = validator.validate(trained_model, output = out_dir)

		print(f'CNN accuracy: {accuracy}')

		return trained_model
