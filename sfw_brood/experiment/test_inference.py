import argparse
import json
from pathlib import Path

from sfw_brood.experiment.experiment_lib import parse_test_args, run_subprocess


def test_inference(
		root_dir: Path, n_workers: int, targets: list, agg_period: int,
		agg_overlap: int = 0, dry_run = False
):
	rec_dir = Path('/home/gardzielb/snowfinch_recordings')
	brood_data_path = Path('/home/gardzielb/snowfinch-recordings.csv')
	# rec_dir = Path(r'D:\MINI\magisterka\Snowfinch\data')
	# brood_data_path = Path(r'D:\MINI\magisterka\Snowfinch\local\snowfinch-recordings.csv')

	error_log = open('experiment-inference-errors.log', mode = 'w')

	for seed_dir in root_dir.rglob('seed-*'):
		setup_path = seed_dir.parent.joinpath('experiment.json')
		if not setup_path.is_file():
			continue

		with open(setup_path) as setup_file:
			experiment_setup = json.load(setup_file)

		if experiment_setup['target'] not in targets:
			continue

		test_args = parse_test_args(
			experiment_setup, seed_dir, rec_dir, brood_data_path,
			n_workers, agg_period, agg_overlap
		)

		print()
		print(' '.join(test_args))

		if not dry_run:
			if run_subprocess(test_args, 'test', seed_dir, error_log):
				print('OK')

	error_log.close()


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('root_dir', type = str)
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 24)
	arg_parser.add_argument('-p', '--agg-period', type = int, default = 48)
	arg_parser.add_argument('-o', '--agg-overlap', type = int, default = 24)
	arg_parser.add_argument('-t', '--target', type = str, choices = ['age', 'size', 'all'], required = True)
	arg_parser.add_argument('-d', '--dry-run', action = 'store_true')
	args = arg_parser.parse_args()

	test_inference(
		root_dir = Path(args.root_dir),
		n_workers = args.n_workers,
		targets = ['age', 'size'] if args.target == 'all' else [args.target],
		agg_period = args.agg_period,
		agg_overlap = args.agg_overlap,
		dry_run = args.dry_run
	)
