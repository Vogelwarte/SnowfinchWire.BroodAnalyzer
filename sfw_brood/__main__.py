import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def detect_feeding(path: str, model_type: str, model_path: str, rec_path: str, out_path: str):
	out_path = Path(out_path).absolute()
	model_path = Path(model_path).absolute()
	rec_path = Path(rec_path).absolute()

	args = [
		'/home/gardzielb/SnowfinchWire.BeggingCallsAnalyzer/venv/bin/python',
		'-m', 'beggingcallsanalyzer', f'predict-{model_type}',
		f'--model-path={model_path.as_posix()}', f'--input-directory={rec_path.as_posix()}',
		f'--output-directory={out_path.as_posix()}', '--extension=WAV'
	]
	print('Detecting feeding calls...')
	process = subprocess.run(args, cwd = path)
	print('Feeding detection done')

	if process.stdout:
		print(process.stdout)
	if process.stderr:
		print(process.stderr)

	if process.returncode:
		print('Feeding detection failed')
		exit(process.returncode)


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('recording_path', type = str, help = 'Path to audio file or directory')
	arg_parser.add_argument('-l', '--label-path', type = str, default = None)
	arg_parser.add_argument('-m', '--model', type = str, help = 'Path to serialized model')
	arg_parser.add_argument('-o', '--output-dir', type = str, default = '_out')
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 10)
	arg_parser.add_argument('-p', '--period-hours', type = int, default = 2)
	arg_parser.add_argument('--overlap-hours', type = int, default = 0)
	arg_parser.add_argument('--mt-threshold', type = float, default = 0.5)
	arg_parser.add_argument('--feeding-detector-type', type = str, choices = ['fe', 'oss'], default = 'fe')
	arg_parser.add_argument('--feeding-detector-path', type = str, default = '.')
	arg_parser.add_argument('--feeding-detector-model', type = str, default = None)
	args = arg_parser.parse_args()

	label_paths = None if args.label_path is None else [Path(args.label_path)]
	if args.feeding_detector_model is not None:
		labels_path = '.sfw-labels' if args.label_path is None else args.label_path
		label_paths = [Path(labels_path)]
		detect_feeding(
			path = args.feeding_detector_path,
			model_type = args.feeding_detector_type,
			model_path = args.feeding_detector_model,
			rec_path = args.recording_path,
			out_path = labels_path
		)

	time_str = datetime.now().isoformat()[:19].replace(':', '-')
	out_path = Path(args.output_dir).joinpath(f'result__{time_str}')

	dev_null = open(os.devnull, mode = 'w')
	sys.stderr = dev_null

	from sfw_brood.inference import Inference
	from sfw_brood.cnn.model import CNNLoader
	from sfw_brood.nemo.model import MatchboxNetLoader

	model_loader = CNNLoader()
	model_loader.set_next(MatchboxNetLoader())

	model_path = Path(args.model)
	if model_path.is_dir():
		model_files = list(model_path.rglob('*.zip'))
	else:
		model_files = [model_path]

	try:
		for model_file in model_files:
			print(f'Loading model {model_file.as_posix()}')

			model = model_loader.load_model(model_file.as_posix())
			inference = Inference(model)

			pred_result = inference.predict(
				paths = [Path(args.recording_path)], n_workers = args.n_workers,
				agg_period_hours = args.period_hours, overlap_hours = args.overlap_hours,
				multi_target_threshold = args.mt_threshold, label_paths = label_paths
			)

			pred_result.save(out_path.joinpath(model_file.stem))

		print(f'Prediction results saved to directory {out_path}')

	except Exception as error:
		print(error)
		exit(1)


if __name__ == '__main__':
	main()
