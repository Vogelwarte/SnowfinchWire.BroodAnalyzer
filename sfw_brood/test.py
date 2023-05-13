import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd

from sfw_brood.cnn.model import CNNLoader
from sfw_brood.inference import Inference, BroodAgeInferenceValidator, BroodSizeInferenceValidator

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
	arg_parser.add_argument('-p', '--period-days', type = int, default = 2)
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 10)
	args = arg_parser.parse_args()

	print(f'Inference test script launched with args: {args}')

	cnn_loader = CNNLoader()
	model = cnn_loader.load_model(args.model)
	print(f'Loaded trained model: {model.model_info}')
	inference = Inference(model)

	with open(args.data_config) as data_conf_file:
		data_config = json.load(data_conf_file)

	if args.target == 'age':
		data_config = data_config['BA']
		validator = BroodAgeInferenceValidator(
			period_days = args.period_days,
			age_groups = [(0, 5.5), (6, 8.5), (9, 11.5), (12, 14.5), (15, 30)],
			multi_target_threshold = 0.5
		)
	else:
		data_config = data_config['BS']
		validator = BroodSizeInferenceValidator(period_days = args.period_days)

	test_data = pd.read_csv(args.test_data)
	test_broods = data_config['test'] if 'test' in data_config.keys() else data_config['validation']
	test_data = test_data[test_data[data_config['selector']].isin(test_broods)]
	print(f'Running inference tests for broods {data_config["test"]}')

	validator.validate_inference(
		inference, test_data,
		data_root = Path(args.data_root),
		output = args.out,
		multi_target = args.target == 'age',
		n_workers = args.n_workers
	)
