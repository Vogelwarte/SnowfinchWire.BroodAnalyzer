import argparse
import json
import warnings
from datetime import datetime
from typing import Optional

from pandas.errors import SettingWithCopyWarning

from sfw_brood.cnn.trainer import CNNTrainer

warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = SettingWithCopyWarning)


def make_time_str() -> str:
	return datetime.now().isoformat()[:19].replace(':', '-')


def parse_age_groups(age_groups: str) -> Optional[list[tuple[float, float]]]:
	if not age_groups:
		return None

	out_groups = []

	for age_group in age_groups.split(','):
		try:
			age_range = age_group.split('-')
			out_groups.append((float(age_range[0]), float(age_range[1])))
		except Exception as parse_error:
			print(f'Invalid age groups: {parse_error}')
			exit(1)

	return out_groups


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
	arg_parser.add_argument('-c', '--split-config-path', type = str)
	arg_parser.add_argument('--group-ages', type = str, default = '')
	arg_parser.add_argument('--samples-per-class', type = str, default = 'min')
	args = arg_parser.parse_args()

	with open(args.split_config_path, mode = 'rt') as split_file:
		data_split_config = json.load(split_file)

	trainer = CNNTrainer(
		data_path = args.data_path,
		audio_path = args.audio_path,
		rec_split = data_split_config,
		work_dir = '_training',
		sample_duration_sec = args.sample_duration,
		cnn_arch = args.arch,
		n_epochs = args.n_epochs,
		n_workers = args.n_workers,
		batch_size = args.batch_size,
		learn_rate = args.learning_rate,
		target_label = None if args.event == 'all' else args.event,
		age_groups = parse_age_groups(args.group_ages),
		samples_per_class = args.samples_per_class
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
