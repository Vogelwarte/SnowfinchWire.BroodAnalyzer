import argparse
import os
from datetime import datetime

from sfw_brood.cnn.trainer import CNNTrainer
from sfw_brood.preprocessing import discover_training_data


def make_time_str() -> str:
	return datetime.now().isoformat()[:19].replace(':', '-')


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-a', '--arch', type = str, default = 'resnet18')
	arg_parser.add_argument('-d', '--sample-duration', type = float, default = 2.0)
	arg_parser.add_argument('-n', '--n-epochs', type = int, default = 10)
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 12)
	arg_parser.add_argument('-b', '--batch-size', type = int, default = 100)
	arg_parser.add_argument('-o', '--sample-overlap', type = float, default = 0.0)
	arg_parser.add_argument('-t', '--target', type = str, choices = ['size', 'age', 'all'], default = 'all')
	args = arg_parser.parse_args()

	train_data_path = os.getenv('DATA_PATH', default = '_data.test')
	sample_duration = 2.0
	train_work_dir = '_training'

	print(f'Collecting train data from directory {train_data_path}')
	train_dataset = discover_training_data(train_data_path)
	# test_dataset = discover_training_data('_data.test')

	trainer = CNNTrainer(
		train_dataset = train_dataset,
		# test_dataset = test_dataset,
		work_dir = train_work_dir,
		sample_duration_sec = args.sample_duration,
		sample_overlap_sec = args.sample_overlap,
		cnn_arch = args.arch,
		n_epochs = args.n_epochs,
		n_workers = args.n_workers,
		batch_size = args.batch_size
	)
	with trainer:
		try:
			if args.target == 'size':
				trainer.train_model_for_size(out_dir = f'_out/BS__{make_time_str()}')
			elif args.target == 'age':
				trainer.train_model_for_age(out_dir = f'_out/BA__{make_time_str()}')
			else:
				trainer.train_model_for_size(out_dir = f'_out/BS__{make_time_str()}')
				trainer.train_model_for_age(out_dir = f'_out/BA__{make_time_str()}')
		except Exception as e:
			print(e)
