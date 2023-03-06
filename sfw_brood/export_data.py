import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from sfw_brood.preprocessing import discover_training_data, prepare_training

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data_path', type = str)
	arg_parser.add_argument('out_path', type = str)
	arg_parser.add_argument('-w', '--work-dir', type = str)
	arg_parser.add_argument('-o', '--overlap', type = float, default = 0.0)
	arg_parser.add_argument('-s', '--sample-len', type = float, default = 2.0)
	arg_parser.add_argument('-j', '--n-jobs', type = int, default = 1)
	arg_parser.add_argument('--rec-data', type = str, default = None)
	arg_parser.add_argument('--skip-if-present', action = 'store_true')
	args = arg_parser.parse_args()

	start_time = datetime.now()

	out_dir = f's{args.sample_len}-o{args.overlap}'
	out_audio_path = Path(args.work_dir).joinpath(out_dir)
	out_csv_path = Path(args.out_path).joinpath(out_dir)
	out_bs_path = out_csv_path.joinpath('brood-size.csv')
	out_ba_path = out_csv_path.joinpath('brood-age.csv')

	if args.skip_if_present:
		results_present = \
			out_audio_path.exists() and out_audio_path.is_dir() and any(out_audio_path.iterdir()) \
			and out_bs_path.exists() and out_bs_path.stat().st_size > 0 \
			and out_ba_path.exists() and out_ba_path.stat().st_size > 0

		if results_present:
			print('Results are already present and skip flag is true, nothing to do.')
			exit(0)

	rec_df = None if args.rec_data is None else pd.read_csv(args.rec_data)
	dataset = discover_training_data(args.data_path, rec_df = rec_df)

	bs_df, ba_df = prepare_training(
		dataset, work_dir = out_audio_path, slice_duration_sec = args.sample_len, overlap_sec = args.overlap,
		n_workers = args.n_jobs, rec_df = rec_df
	)

	out_csv_path.mkdir(exist_ok = True, parents = True)
	bs_df.to_csv(out_bs_path)
	ba_df.to_csv(out_ba_path)

	task_duration = datetime.now() - start_time
	print(f'Task completed in {task_duration}')
