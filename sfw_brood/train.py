import argparse
import json
import warnings
from datetime import datetime
from typing import Optional, Tuple

from pandas.errors import SettingWithCopyWarning

from sfw_brood.cnn.trainer import CNNTrainer
# from sfw_brood.nemo.trainer import MatchboxNetTrainer
from sfw_brood.simple_size_clf.trainer import SimpleClfTrainer

warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = SettingWithCopyWarning)


def make_time_str() -> str:
	return datetime.now().isoformat()[:19].replace(':', '-')


def parse_range_str(range_str: str, throw_error = False) -> Optional[Tuple[float, float]]:
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


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data_path', type = str)
	arg_parser.add_argument('--audio-path', type = str, default = None)
	arg_parser.add_argument(
		'-a', '--arch', type = str, default = 'resnet18', choices = [
			'resnet18', 'resnet50', 'resnet101', 'resnet152', 'vgg11_bn',
			'densenet121', 'inception_v3', 'matchboxnet', 'simple-ensemble'
		]
	)
	arg_parser.add_argument('-d', '--sample-duration', type = float, default = 2.0)
	arg_parser.add_argument('-n', '--n-epochs', type = int, default = 10)
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 12)
	arg_parser.add_argument('-b', '--batch-size', type = int, default = 100)
	arg_parser.add_argument('-l', '--learning-rate', type = float, default = 0.001)
	arg_parser.add_argument('-e', '--event', type = str, choices = ['feeding', 'contact', 'all'], default = 'all')
	arg_parser.add_argument('-t', '--target', type = str, choices = ['size', 'age', 'all'], default = 'all')
	arg_parser.add_argument('-c', '--split-config-path', type = str, required = True)
	arg_parser.add_argument('--age-mode', type = str, default = 'single')
	arg_parser.add_argument('--samples-per-class', type = str, default = 'min')
	arg_parser.add_argument('--age-range', type = str, default = 'none')
	arg_parser.add_argument('--out', type = str, default = '_out')
	arg_parser.add_argument('--n-simple-models', type = int, default = 20)
	arg_parser.add_argument('--ensemble-voting', type = str, choices = ['soft', 'hard'], default = 'soft')
	arg_parser.add_argument('--ensemble-svm', action = 'store_true')
	arg_parser.add_argument('--ensemble-mlp', action = 'store_true')
	arg_parser.add_argument('--ensemble-rfc', action = 'store_true')
	arg_parser.add_argument('--ensemble-bayes', action = 'store_true')
	args = arg_parser.parse_args()

	with open(args.split_config_path, mode = 'rt') as split_file:
		data_split_config = json.load(split_file)

	if args.arch == 'matchboxnet':
		print('MatchboxNet is not supported in the current version')
		exit(1)
		# trainer = MatchboxNetTrainer(
		# 	dataset_path = args.data_path,
		# 	audio_path = args.audio_path,
		# 	data_config = data_split_config,
		# 	sample_duration = args.sample_duration,
		# 	n_epochs = args.n_epochs,
		# 	n_workers = args.n_workers,
		# 	batch_size = args.batch_size,
		# 	learn_rate = args.learning_rate,
		# 	samples_per_class = args.samples_per_class
		# )
	elif args.arch == 'simple-ensemble':
		trainer = SimpleClfTrainer(
			data_path = args.data_path,
			data_config = data_split_config,
			n_models = args.n_simple_models,
			voting = args.ensemble_voting,
			svm = args.ensemble_svm,
			rfc = args.ensemble_rfc,
			mlp = args.ensemble_mlp,
			bayes = args.ensemble_bayes
		)
	else:
		if args.audio_path is None:
			print('Audio path not specified, cannot train CNN model')
			exit(1)

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
			samples_per_class = args.samples_per_class,
			age_multi_target = 'multi' in args.age_mode,
			age_mt_threshold = float(args.age_mode.split('-')[1]) if 'multi' in args.age_mode else 'single',
			age_range = parse_range_str(args.age_range)
		)

	with trainer:
		try:
			if args.arch == 'simple-ensemble':
				trainer.train_model_for_size(out_dir = f'{args.out}/BS__{make_time_str()}')
			else:
				if args.target == 'size':
					trainer.train_model_for_size(out_dir = f'{args.out}/BS__{make_time_str()}')
				elif args.target == 'age':
					trainer.train_model_for_age(out_dir = f'{args.out}/BA__{make_time_str()}')
				else:
					trainer.train_model_for_size(out_dir = f'{args.out}/BS__{make_time_str()}')
					trainer.train_model_for_age(out_dir = f'{args.out}/BA__{make_time_str()}')
		except Exception as e:
			print(e)


if __name__ == '__main__':
	main()
