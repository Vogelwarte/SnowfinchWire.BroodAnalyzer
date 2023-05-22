from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from opensoundscape.torch.models.cnn import CNN, InceptionV3, load_model, use_resample_loss

from sfw_brood.model import ModelTrainer
from sfw_brood.preprocessing import group_ages, group_sizes
from .preprocessing import select_recordings
from .model import SnowfinchBroodCNN
from .util import cleanup
from .validator import CNNValidator


class CNNTrainer(ModelTrainer):
	def __init__(
			self, data_path: str, audio_path: str, work_dir: str,
			sample_duration_sec: float, rec_split: dict,
			cnn_arch: str, n_epochs: int, n_workers = 12, batch_size = 100, learn_rate = 0.001,
			target_label: Optional[str] = None, remove_silence: bool = True,
			samples_per_class = 'min', age_multi_target = False, age_mt_threshold = 0.7,
			age_range: Optional[Tuple[float, float]] = None
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
		self.age_multi_target = age_multi_target
		self.age_mt_threshold = age_mt_threshold
		self.age_range = age_range

		age_data = pd.read_csv(f'{data_path}/brood-age.csv', dtype = { 'is_silence': 'bool' })
		size_data = pd.read_csv(f'{data_path}/brood-size.csv', dtype = { 'is_silence': 'bool', 'class': 'int' })

		if age_range:
			low, high = age_range
			age_range_files = age_data.loc[(age_data['class_min'] >= low) & (age_data['class_max'] < high), 'file']
			size_data = size_data.set_index('file').loc[age_range_files].reset_index()

		size_data = size_data[~size_data['is_silence'] & (size_data['event'].isin(self.target_labels))]
		if 'groups' in rec_split['size']:
			size_data, size_classes = group_sizes(size_data, groups = rec_split['size']['groups'])
		else:
			size_classes = None

		self.bs_train_data, self.bs_val_data, self.bs_test_data = select_recordings(
			size_data, audio_path, self.samples_per_class, split_conf = rec_split['size'], classes = size_classes
		)
		print(f'\nSize data:')
		print(f'\ttrain: {self.bs_train_data.shape}')
		print(f'\tvalidation: {None if self.bs_val_data is None else self.bs_val_data.shape}')
		print(f'\ttest: {None if self.bs_test_data is None else self.bs_test_data.shape}')

		age_data = age_data[~age_data['is_silence'] & (age_data['event'].isin(self.target_labels))]
		if 'groups' in rec_split['age']:
			age_data, age_classes = group_ages(
				age_data, groups = rec_split['age']['groups'], multi_target = age_multi_target
			)
		else:
			age_classes = None

		self.ba_train_data, self.ba_val_data, self.ba_test_data = select_recordings(
			age_data, audio_path, self.samples_per_class, split_conf = rec_split['age'], classes = age_classes
		)
		print(f'\nAge data:')
		print(f'\ttrain: {self.ba_train_data.shape}')
		print(f'\tvalidation: {None if self.ba_val_data is None else self.ba_val_data.shape}')
		print(f'\ttest: {None if self.ba_test_data is None else self.ba_test_data.shape}')

	def __enter__(self):
		self.work_dir.mkdir(parents = True, exist_ok = True)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		cleanup(self.work_dir)

	def train_model_for_size(self, out_dir: str):
		return self.__do_training__(
			self.bs_train_data, self.bs_test_data, self.bs_val_data, out_dir,
			target = 'size', multi_target = False
		)

	def train_model_for_age(self, out_dir: str):
		return self.__do_training__(
			self.ba_train_data, self.ba_test_data, self.ba_val_data, out_dir,
			target = 'age', multi_target = self.age_multi_target
		)

	def __do_training__(
			self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame],
			validation_data: Optional[pd.DataFrame], out_dir: str, target: str, multi_target: bool
	):
		if train_data.shape[0] == 0:
			print('No training data available')
			return

		print('Training CNN...')
		cnn = self.__setup_cnn__(classes = train_data.columns, multi_target = multi_target)

		out_path = Path(out_dir)
		out_path.mkdir(parents = True, exist_ok = True)

		trained_model = self.__train_and_validate__(
			cnn, train_data, test_data, validation_data, out_dir, target, multi_target
		)
		trained_model.serialize(f'{out_dir}/cnn.model')

	def __setup_cnn__(self, classes, multi_target: bool):
		if self.cnn_arch == 'inception_v3':
			cnn = InceptionV3(
				classes = classes, sample_duration = self.sample_duration_sec,
				single_target = not multi_target
			)
		else:
			cnn = CNN(
				architecture = self.cnn_arch,
				sample_duration = self.sample_duration_sec,
				classes = classes,
				single_target = not multi_target
			)
			if multi_target:
				use_resample_loss(cnn)

		cnn.optimizer_params['lr'] = self.learn_rate
		return cnn

	def __train_cnn__(
			self, cnn: CNN, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame],
			target: str, multi_target: bool
	) -> SnowfinchBroodCNN:
		cnn.train(
			train_data, validation_df = validation_data, epochs = self.n_epochs, batch_size = self.batch_size,
			save_path = f'{self.work_dir}/models', num_workers = self.n_workers
		)

		trained_cnn = load_model(f'{self.work_dir}/models/best.model')
		return SnowfinchBroodCNN(
			trained_cnn,
			model_info = {
				'target': target,
				'architecture': self.cnn_arch,
				'learning_rate': self.learn_rate,
				'batch_size': self.batch_size,
				'train_epochs': trained_cnn.current_epoch,
				'data': self.data_path,
				'data_config': self.data_split[target],
				'sample_duration_sec': self.sample_duration_sec,
				'events': self.target_labels,
				'multi_target': multi_target,
				'mt_thredholsd': self.age_mt_threshold,
				'age_range': self.age_range
			}
		)

	def __train_and_validate__(
			self, cnn: CNN, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame],
			validation_data: Optional[pd.DataFrame], out_dir: str, target: str, multi_target: bool
	) -> SnowfinchBroodCNN:
		trained_model = self.__train_cnn__(cnn, train_data, validation_data, target, multi_target)

		if test_data is not None:
			self.__validate__(
				trained_model, test_data, out_dir, label = f'brood {target}', multi_target = multi_target
			)
		elif validation_data is not None:
			print('Running test step with validation data')
			self.__validate__(
				trained_model, validation_data, out_dir, label = f'brood {target}', multi_target = multi_target
			)
		else:
			print('No test nor validation data, skipping validation')

		return trained_model

	def __validate__(self, model: SnowfinchBroodCNN, test_data: pd.DataFrame, out: str, label: str, multi_target: bool):
		print(f'Testing model with output dir {out}')
		validator = CNNValidator(test_data, label, n_workers = self.n_workers)
		test_result = validator.validate(model, output = out, multi_target = multi_target)
		print(f'CNN test result: {test_result}')
