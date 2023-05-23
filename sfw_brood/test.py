import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd

from sfw_brood.cnn.model import CNNLoader
from sfw_brood.inference import Inference, BroodAgeInferenceValidator, BroodSizeInferenceValidator
from sfw_brood.nemo.model import MatchboxNetLoader

warnings.simplefilter(action = 'ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('test_data', type = str)
	arg_parser.add_argument('-d', '--data-root', type = str)
	arg_parser.add_argument('-c', '--data-config', type = str)
	arg_parser.add_argument('-m', '--model', type = str)
	arg_parser.add_argument('-o', '--out', type = str)
	arg_parser.add_argument('-t', '--target', type = str, choices = ['age', 'size'])
	arg_parser.add_argument('-p', '--period-hours', type = int, default = 2)
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 10)
	arg_parser.add_argument('--overlap-hours', type = int, default = 0)
	arg_parser.add_argument('--mt-threshold', type = float, default = 0.5)
	args = arg_parser.parse_args()

	print(f'Inference test script launched with args: {args}')

	model_loader = CNNLoader()
	model_loader.set_next(MatchboxNetLoader())

	model = model_loader.load_model(args.model)
	print(f'Loaded trained model: {model.model_info}')
	inference = Inference(model)

	with open(args.data_config) as data_conf_file:
		data_config = json.load(data_conf_file)[args.target]

	if args.target == 'age':
		validator = BroodAgeInferenceValidator(
			period_hours = args.period_hours,
			overlap_hours = args.overlap_hours,
			age_groups = data_config['groups'],
			multi_target_threshold = args.mt_threshold
		)
	else:
		validator = BroodSizeInferenceValidator(
			period_hours = args.period_hours,
			overlap_hours = args.overlap_hours,
			size_groups = data_config['groups'] if 'groups' in data_config.keys() else None
		)

	test_data = pd.read_csv(args.test_data)
	test_broods = data_config['test'] if 'test' in data_config.keys() else data_config['validation']
	test_data = test_data[test_data[data_config['selector']].isin(test_broods)]
	print(f'Running inference tests for broods {test_broods}')

	validator.validate_inference(
		inference, test_data,
		data_root = Path(args.data_root),
		output = args.out,
		n_workers = args.n_workers
	)
