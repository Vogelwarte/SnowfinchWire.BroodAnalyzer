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


# def parse_cls_groups(cls_groups: str) -> Optional[list[tuple[float, float]]]:
# 	if cls_groups == 'none':
# 		return None
#
# 	out_groups = []
#
# 	for cls_group in cls_groups.split(','):
# 		try:
# 			out_groups.append(parse_range_str(cls_group, throw_error = True))
# 		except Exception as parse_error:
# 			print(f'Invalid age groups: {parse_error}')
# 			exit(1)
#
# 	return out_groups


def parse_range_str(range_str: str, throw_error = False) -> Optional[tuple[float, float]]:
	if range_str == 'none':
		return None

	try:
		low, high = range_str.split('-')
		return float(low), float(high)
	except ValueError as error:
		if throw_error:
			raise error
		else:
			print('Warning: invalid range format, ignoring range argument')
			return None


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
	# arg_parser.add_argument('--group-ages', type = str, default = 'none')
	# arg_parser.add_argument('--group-sizes', type = str, default = 'none')
	arg_parser.add_argument('--age-mode', type = str, default = 'single')
	arg_parser.add_argument('--samples-per-class', type = str, default = 'min')
	arg_parser.add_argument('--age-range', type = str, default = 'none')
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
		# age_groups = parse_cls_groups(args.group_ages),
		# size_groups = parse_cls_groups(args.group_sizes),
		samples_per_class = args.samples_per_class,
		age_multi_target = 'multi' in args.age_mode,
		age_mt_threshold = float(args.age_mode.split('-')[1]) if 'multi' in args.age_mode else 'single',
		age_range = parse_range_str(args.age_range)
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
