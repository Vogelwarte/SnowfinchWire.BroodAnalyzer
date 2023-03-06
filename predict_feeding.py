import argparse
import subprocess
from pathlib import Path

import pandas as pd

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data_path', type = str)
	arg_parser.add_argument('-m', '--model-path', type = str)
	arg_parser.add_argument('-r', '--rec-data', type = str)
	args = arg_parser.parse_args()

	rec_df = pd.read_csv(args.rec_data)
	brood_ids = list(rec_df['brood_id'].unique())
	data_path = Path(args.data_path)
	rec_dirs = [path for path in data_path.rglob('*') if path.stem in brood_ids]

	processes = []
	for rec_dir in rec_dirs:
		cmd = ['./sfw-bca', 'predict', args.model_path, rec_dir.as_posix(), '--extension', 'WAV']
		processes.append(subprocess.Popen(cmd))

	for process in processes:
		exit_code = process.wait()
		if exit_code:
			print(f'Process {process} exited with code {exit_code}')
