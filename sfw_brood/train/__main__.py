import os
from datetime import datetime

from sfw_brood.train.cnn_trainer import CNNTrainer

if __name__ == '__main__':
	train_data_path = os.getenv('DATA_PATH', default = '_data')
	sample_duration = 2.0
	train_work_dir = '_training'

	with CNNTrainer(train_data_path, sample_duration, train_work_dir) as trainer:
		model = trainer.train_model_for_size()
		# time_str = datetime.now().isoformat()[:19].replace(':', '-')
		model.serialize('_out/trained_model')
