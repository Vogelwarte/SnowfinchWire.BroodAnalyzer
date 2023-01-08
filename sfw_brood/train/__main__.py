import os
from datetime import datetime

from sfw_brood.train.cnn_trainer import CNNTrainer
from sfw_brood.preprocessing import discover_training_data
from sfw_brood.train.trainer import TrainResult


def evaluate_and_save_model(train_result: TrainResult, label: str):
	if train_result.model:
		print(f'{label} trained model accuracy: {train_result.validation_score}')
		time_str = datetime.now().isoformat()[:19].replace(':', '-')
		train_result.model.serialize(f'_out/{label}.{time_str}.cnn')


if __name__ == '__main__':
	train_data_path = os.getenv('DATA_PATH', default = '_data')
	sample_duration = 2.0
	train_work_dir = '_training'

	print(f'Collecting train data from directory {train_data_path}')
	train_dataset = discover_training_data(train_data_path)

	with CNNTrainer(train_dataset, sample_duration, train_work_dir) as trainer:
		bs_train_result = trainer.train_model_for_size(validate = True)
		evaluate_and_save_model(bs_train_result, 'BS')

		ba_train_result = trainer.train_model_for_age(validate = True)
		evaluate_and_save_model(ba_train_result, 'BA')
