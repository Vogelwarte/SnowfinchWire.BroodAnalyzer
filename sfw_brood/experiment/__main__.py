import json
import time
from pathlib import Path

from sfw_brood.experiment.experiment_lib import time_str, hash_experiment, parse_train_args, run_subprocess, \
	parse_test_args


def run():
	work_dir = Path('/home/gardzielb/sfw-brood-work')
	rec_dir = Path('/home/gardzielb/snowfinch_recordings')
	brood_data_path = Path('/home/gardzielb/snowfinch-recordings.csv')
	# work_dir = Path(r'D:\MINI\magisterka\Snowfinch\local\work')
	# rec_dir = Path(r'D:\MINI\magisterka\Snowfinch\data')
	# brood_data_path = Path(r'D:\MINI\magisterka\Snowfinch\local\snowfinch-recordings.csv')
	out_path = work_dir.joinpath('final-results').joinpath(f'oss-{time_str()}')

	with open('sfw_brood/experiment/experiment.json') as exp_file:
		experiment = json.load(exp_file)

	error_log = open('/home/gardzielb/sfw-brood-work/experiment-errors.log', mode = 'w')

	for setup in experiment['oss']:
		# out = out_path.joinpath(time_str())
		out = out_path.joinpath(hash_experiment(setup))
		out.mkdir(parents = True, exist_ok = True)

		setup['durations'] = {}

		for rng_seed in experiment['rng_seeds']:
			start_ts = time.time()
			seed_out = out.joinpath(f'seed-{rng_seed}')

			train_args = parse_train_args(work_dir, setup, seed_out, rng_seed, experiment['n_workers'])
			print(' '.join(train_args))
			if not run_subprocess(train_args, 'train', seed_out, error_log):
				continue

			test_args = parse_test_args(setup, seed_out, rec_dir, brood_data_path, experiment['n_workers'])
			print(' '.join(test_args))
			run_subprocess(test_args, 'test', seed_out, error_log)

			setup['durations'][rng_seed] = time.time() - start_ts

		with open(out.joinpath('experiment.json'), mode = 'wt') as setup_file:
			json.dump(setup, setup_file)

	error_log.close()


if __name__ == '__main__':
	run()
