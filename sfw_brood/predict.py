import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from sfw_brood.model import ModelType
from sfw_brood.simple_size_clf.inference import SimpleSizeInference
from sfw_brood.simple_size_clf.model import SimpleClfLoader


def detect_feeding(path: str, model_type: str, model_path: str, rec_path: str, out_path: str, extra_args: dict):
	out_path = Path(out_path).absolute()
	model_path = Path(model_path).absolute()
	rec_path = Path(rec_path).absolute()

	args = [
		sys.executable, '-Wignore', 'classify.py', model_type,
		f'--model-path={model_path.as_posix()}', f'--input-directory={rec_path.as_posix()}',
		f'--output-directory={out_path.as_posix()}', '--create-plots'
	]

	for key, val in extra_args.items():
		args.append(f'--{key}={val}')

	print('Detecting feeding calls...')
	print(' '.join(args))
	process = subprocess.run(args, cwd = path)
	print('Feeding detection done')

	if process.stdout:
		print(process.stdout)
	if process.stderr:
		print(process.stderr)

	if process.returncode:
		print('Feeding detection failed')
		exit(process.returncode)


def load_config(path: str) -> dict:
	with open(path) as config_file:
		config = yaml.safe_load(config_file)

	try:
		args = {
			'input_path': config['data']['recordings'],
			'label_path': config['data'].get('labels', None),
			'model_path': config['model']['path'],
			'n_workers': config['model'].get('n_workers', 10),
			'output_dir': config.get('output_dir', '_out'),
			'period_hours': config['aggregation'].get('period_hours', 48),
			'overlap_hours': config['aggregation'].get('overlap_hours', 0)
		}
		if 'feeding_detector' in config.keys():
			args['feeding_detector_type'] = config['feeding_detector'].get('type', 'fe')
			args['feeding_detector_path'] = config['feeding_detector'].get('path', '.')
			fd_args = config['feeding_detector']['args']
			args['feeding_detector_model'] = fd_args.pop('model_path')
			args['feeding_detector_extra_args'] = fd_args
		return args

	except KeyError:
		print('Invalid config file')
		exit(1)


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
		'-i', '--input-path', type = str, default = None,
		help = 'Path to input data: audio file, directory with audio files or directory with feeding stats'
	)
	arg_parser.add_argument('-c', '--config-path', type = str, help = 'Path to config file', default = None)
	arg_parser.add_argument('-l', '--label-path', type = str, default = None)
	arg_parser.add_argument('-m', '--model-path', type = str, help = 'Path to serialized model', default = None)
	arg_parser.add_argument('-o', '--output-dir', type = str, default = '_out')
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 10)
	arg_parser.add_argument('-p', '--period-hours', type = int, default = 48)
	arg_parser.add_argument('--overlap-hours', type = int, default = 0)
	arg_parser.add_argument('--mt-threshold', type = float, default = 0.5)
	arg_parser.add_argument('--feeding-detector-type', type = str, choices = ['fe', 'oss'], default = 'fe')
	arg_parser.add_argument('--feeding-detector-path', type = str, default = '.')
	arg_parser.add_argument('--feeding-detector-model', type = str, default = None)
	args = arg_parser.parse_args()

	feeding_detector_args = { }
	if args.config_path is not None:
		config = load_config(args.config_path)
		print(f'Loaded config from file: {config}')
		args.__dict__.update(config)
		feeding_detector_args.update(config.get('feeding_detector_extra_args', { }))

	if args.input_path is None or args.model_path is None:
		print('Missing required arguments: input_path or model')
		exit(1)

	label_paths = None if args.label_path is None else [Path(args.label_path)]
	input_path = Path(args.input_path)
	feeding_stats_path = input_path.joinpath('feeding-stats.csv')
	age_pred_path = input_path.joinpath('brood-age.csv')

	if args.feeding_detector_model is not None:
		labels_path = '.sfw-labels' if args.label_path is None else args.label_path
		feeding_stats_path = Path(labels_path).joinpath('summary.csv')
		feeding_stats_path.unlink(missing_ok = True)  # otherwise BCA does not write CSV header ...

		detect_feeding(
			path = args.feeding_detector_path,
			model_type = args.feeding_detector_type,
			model_path = args.feeding_detector_model,
			rec_path = args.input_path,
			out_path = labels_path,
			extra_args = feeding_detector_args
		)
		label_paths = [Path(labels_path)]

	time_str = datetime.now().isoformat()[:19].replace(':', '-')
	out_path = Path(args.output_dir).joinpath(f'result__{time_str}')

	error_log = open('.error.log', mode = 'w')
	sys.stderr = error_log

	from sfw_brood.inference.core import Inference
	from sfw_brood.cnn.model import CNNLoader
	# from sfw_brood.nemo.model import MatchboxNetLoader

	model_loader = CNNLoader()
	model_loader \
		.set_next(SimpleClfLoader()) \
		# .set_next(MatchboxNetLoader())

	model_path = Path(args.model_path)
	if model_path.is_dir():
		model_files = list(model_path.rglob('*.zip'))
	else:
		model_files = [model_path]

	# make sure that size-stat-model is used after all the others,
	# so that age pred results can be produced for it to use
	model_files.sort(key = lambda path: 1 if 'size-stat' in path.stem else 0)

	try:
		for model_file in model_files:
			print(f'Loading model {model_file.as_posix()}')
			model = model_loader.load_model(model_file.as_posix())
			model_out_path = out_path.joinpath(model_file.stem)

			if model.model_type == ModelType.SIMPLE_SIZE_CLF:
				inference = SimpleSizeInference(model)
				preds = inference.predict(
					feeding_stats_path, age_pred_path,
					period_hours = args.period_hours, overlap_hours = args.overlap_hours
				)
				for pred in preds:
					pred.save(model_out_path.joinpath(pred.model_name))
			else:
				inference = Inference(model)
				pred_result = inference.predict(
					paths = [input_path], n_workers = args.n_workers,
					agg_period_hours = args.period_hours, overlap_hours = args.overlap_hours,
					multi_target_threshold = args.mt_threshold, label_paths = label_paths
				)
				pred_result.save(model_out_path)
				if model.model_info['target'] == 'age':
					age_pred_path = model_out_path.joinpath('brood-period-preds.csv')

		print(f'Prediction results saved to directory {out_path}')

	except Exception as error:
		print(error)
		exit(1)

	error_log.close()


if __name__ == '__main__':
	main()
