import os
from datetime import datetime

from sfw_brood.train.cnn_trainer import CNNTrainer
from sfw_brood.preprocessing import discover_training_data

if __name__ == '__main__':
	train_data_path = os.getenv('DATA_PATH', default = '_data')
	sample_duration = 2.0
	train_work_dir = '_training'

	print(f'Collecting train data from directory {train_data_path}')
	train_dataset = discover_training_data(train_data_path)

	with CNNTrainer(train_dataset, sample_duration, train_work_dir) as trainer:
		train_result = trainer.train_model_for_size(validate = True)
		if train_result.model:
			print(f'Trained model accuracy: {train_result.validation_score}')
			# time_str = datetime.now().isoformat()[:19].replace(':', '-')
			train_result.model.serialize('_out/model.bs')
