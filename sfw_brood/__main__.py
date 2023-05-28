# import warnings
# warnings.simplefilter(action = 'ignore')

import os
import sys

dev_null = open(os.devnull, mode = 'w')
sys.stderr = dev_null

import argparse
from datetime import datetime
from pathlib import Path

from sfw_brood.cnn.model import CNNLoader
from sfw_brood.inference import Inference
from sfw_brood.nemo.model import MatchboxNetLoader


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
	args = arg_parser.parse_args()

	# with warnings.catch_warnings():
	# 	warnings.simplefilter('ignore')

	model_loader = CNNLoader()
	model_loader.set_next(MatchboxNetLoader())
	model = model_loader.load_model(args.model)
	inference = Inference(model)

	pred_result = inference.predict(
		paths = [Path(args.recording_path)], n_workers = args.n_workers,
		agg_period_hours = args.period_hours, overlap_hours = args.overlap_hours,
		multi_target_threshold = args.mt_threshold, label_paths = [Path(args.label_path)]
	)

	# print('\nPrediction result:')
	# print(pred_result.brood_periods_results[['brood_id']])

	time_str = datetime.now().isoformat()[:19].replace(':', '-')
	out_path = Path(args.output_dir).joinpath(f'result__{time_str}')
	pred_result.save(out_path)
	print(f'Prediction results saved to directory {out_path}')


if __name__ == '__main__':
	main()
