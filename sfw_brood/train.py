import argparse
import json
from datetime import datetime

from sfw_brood.cnn.trainer import CNNTrainer


def make_time_str() -> str:
	return datetime.now().isoformat()[:19].replace(':', '-')


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data_path', type = str)
	arg_parser.add_argument('audio_path', type = str)
	arg_parser.add_argument('-a', '--arch', type = str, default = 'resnet18')
	arg_parser.add_argument('-d', '--sample-duration', type = float, default = 2.0)
	arg_parser.add_argument('-n', '--n-epochs', type = int, default = 10)
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 12)
	arg_parser.add_argument('-b', '--batch-size', type = int, default = 100)
	arg_parser.add_argument('-l', '--learning-rate', type = float, default = 0.001)
	arg_parser.add_argument('-e', '--event', type = str, choices = ['feeding', 'contact', 'all'], default = 'all')
	arg_parser.add_argument('-t', '--target', type = str, choices = ['size', 'age', 'all'], default = 'all')
	arg_parser.add_argument('-c', '--split-config-path', type = str, default = 'data-split.json')
	args = arg_parser.parse_args()

	with open(args.split_config_path, mode = 'rt') as split_file:
		data_split_config = json.load(split_file)

	trainer = CNNTrainer(
		data_path = args.data_path,
		audio_path = args.audio_path,
		train_recordings = data_split_config['train'],
		validation_recordings = data_split_config['validation'],
		test_recordings = data_split_config['test'],
		work_dir = '_training',
		sample_duration_sec = args.sample_duration,
		cnn_arch = args.arch,
		n_epochs = args.n_epochs,
		n_workers = args.n_workers,
		batch_size = args.batch_size,
		learn_rate = args.learning_rate,
		target_label = None if args.event == 'all' else args.event
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
