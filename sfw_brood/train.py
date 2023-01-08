import os
from datetime import datetime

from sfw_brood.cnn.trainer import CNNTrainer
from sfw_brood.preprocessing import discover_training_data


def make_time_str() -> str:
	return datetime.now().isoformat()[:19].replace(':', '-')


if __name__ == '__main__':
	train_data_path = os.getenv('DATA_PATH', default = '_data')
	sample_duration = 2.0
	train_work_dir = '_training'

	print(f'Collecting train data from directory {train_data_path}')
	train_dataset = discover_training_data(train_data_path)

	with CNNTrainer(train_dataset, sample_duration, train_work_dir) as trainer:
		trainer.train_model_for_size(out_dir = f'_out/BS__{make_time_str()}', validate = True)
		trainer.train_model_for_age(out_dir = f'_out/BA__{make_time_str()}', validate = True)
