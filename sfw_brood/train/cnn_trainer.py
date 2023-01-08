from pathlib import Path

import pandas as pd
from opensoundscape.torch.models.cnn import CNN, load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sfw_brood.models.cnn import SnowfinchBroodCNN
from sfw_brood.preprocessing import SnowfinchDataset, prepare_training
from sfw_brood.train.trainer import ModelTrainer, TrainResult


def __cleanup__(path: Path):
	if not path.exists():
		return

	if not path.is_dir():
		path.unlink()
		return

	for file in path.iterdir():
		__cleanup__(file)
	path.rmdir()


def validate_cnn(model: SnowfinchBroodCNN, test_data: pd.DataFrame) -> float:
	rec_files = list(test_data.index)
	classes = list(test_data.columns)

	pred_df = model.predict(rec_files)

	y_pred = pred_df[classes].idxmax(axis = 1)
	y_true = test_data.loc[pred_df.reset_index().file][classes].idxmax(axis = 1)

	return accuracy_score(y_true, y_pred)


class CNNTrainer(ModelTrainer):
	def __init__(self, dataset: SnowfinchDataset, sample_duration_sec: float, work_dir: str):
		self.dataset = dataset
		self.sample_duration_sec = sample_duration_sec
		self.work_dir = work_dir

	def train_model_for_size(self, validate = False) -> TrainResult:
		return self.__do_training__(self.bs_train_data, validate)

	def train_model_for_age(self, validate = False) -> TrainResult:
		return self.__do_training__(self.ba_train_data, validate)

	def __enter__(self):
		self.bs_train_data, self.ba_train_data = prepare_training(
			self.dataset, self.work_dir, self.sample_duration_sec, overlap_sec = 0.0
		)

		print(f'Brood size training data shape: {self.bs_train_data.shape}')
		print(f'Brood age training data shape: {self.ba_train_data.shape}')

		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		print('Cleaning up')
		__cleanup__(Path(self.work_dir))

	def __do_training__(self, train_data: pd.DataFrame, validate: bool) -> TrainResult:
		if train_data.shape[0] == 0:
			print('No training data available')
			return TrainResult(model = None, validation_score = 0.0)

		cnn = CNN(
			architecture = 'resnet18',
			sample_duration = self.sample_duration_sec,
			classes = train_data.columns,
			single_target = True
		)

		print('Training CNN...')

		if validate:
			train_data, test_data = train_test_split(train_data, test_size = 0.2)
			trained_model = self.__train_cnn__(cnn, train_data)
			return TrainResult(
				model = trained_model,
				validation_score = validate_cnn(trained_model, test_data)
			)

		trained_model = self.__train_cnn__(cnn, train_data)
		return TrainResult(model = trained_model, validation_score = 0.0)

	def __train_cnn__(self, cnn: CNN, train_data: pd.DataFrame) -> SnowfinchBroodCNN:
		train_df, validation_df = train_test_split(train_data, test_size = 0.15)
		cnn.train(
			train_df, validation_df, epochs = 5, batch_size = 100,
			save_path = f'{self.work_dir}/models', num_workers = 12
		)
		return SnowfinchBroodCNN(trained_cnn = load_model(f'{self.work_dir}/models/best.model'))
