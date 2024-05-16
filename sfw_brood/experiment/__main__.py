import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_test_args(
		setup: dict, out_dir: Path,
		data_path: Path, brood_data_path: Path
) -> list[str]:
	target = setup['target']

	out_dir_pattern = 'BS__*' if target == 'size' else 'BA__*'
	train_out_dirs = sorted(out_dir.glob(out_dir_pattern))
	if not train_out_dirs:
		print('Training output not found')
		exit(1)

	train_out_dir = train_out_dirs[-1]

	args = [
		sys.executable, '-m', 'sfw_brood.test',
		'-d', data_path.as_posix(),
		'-c', Path('config').joinpath(setup['data_config']).as_posix(),
		'-m', train_out_dir.joinpath(f'{target}-oss-model.zip').as_posix(),
		'-o', train_out_dir.joinpath('inference').as_posix(),
		'-t', target, '-w', '12'
	]

	if target == 'age':
		args.extend(['-p', '12'])
	elif target == 'size':
		args.extend(['-p', '48', '--overlap-hours', '24'])

	args.append(brood_data_path.as_posix())
	return args


def parse_train_args(work_dir: Path, setup: dict, out: Path) -> list[str]:
	sample_duration = setup['sample_duration']
	target = setup['target']

	args = [
		sys.executable, 'train_model.py', '-w', '12',
		'-n', str(setup['max_epochs']), '-a', setup['architecture'],
		'-b', str(setup['bs']), '-l', str(setup['lr']), '-d', str(sample_duration),
		'-t', target, '-e', 'feeding', '-c', f'config/{setup["data_config"]}',
		'--samples-per-class', str(setup['samples_per_class']),
		'--out', out.as_posix(),
		'--audio-path', work_dir.joinpath(f's{sample_duration}/audio').as_posix()
	]

	if target == 'age':
		args += age_args(setup)
	elif target == 'size':
		args += size_args(setup)

	args.append(work_dir.joinpath(f'out/s{sample_duration}').as_posix())

	return args


def size_args(setup: dict) -> list[str]:
	if 'age_range' in setup:
		return ['--age-range', setup['age_range']]
	return []


def age_args(setup: dict) -> list[str]:
	if 'mt_thresh' in setup:
		age_mode = f'multi-{setup["mt_thresh"]}'
	else:
		age_mode = 'single'

	return ['--age-mode', age_mode]


def time_str() -> str:
	return datetime.now().isoformat()[:19].replace(':', '-')


def run_subprocess(args: list[str], name: str, out_dir: Path, error_log) -> bool:
	process = subprocess.run(args, text = True, stderr = error_log)

	if process.stdout:
		with open(out_dir.joinpath(f'{name}-stdout.txt'), mode = 'wt') as stdout_file:
			stdout_file.write(process.stdout)
	if process.stderr:
		with open(out_dir.joinpath(f'{name}-stdout.txt'), mode = 'wt') as stdout_file:
			stdout_file.write(process.stdout)

	if process.returncode:
		print(f'{name.capitalize()} process failed: {process.returncode}')
		return False

	return True


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
		out = out_path.joinpath(time_str())
		out.mkdir(parents = True, exist_ok = True)

		train_args = parse_train_args(work_dir, setup, out)
		# print(' '.join(train_args))
		# train_result_dir = ('BA' if setup['target'] == 'age' else 'BS') + '__dupa'
		# out.joinpath(train_result_dir).mkdir()
		if not run_subprocess(train_args, 'train', out, error_log):
			continue

		test_args = parse_test_args(setup, out, rec_dir, brood_data_path)
		# print(' '.join(test_args))
		run_subprocess(test_args, 'test', out, error_log)

	error_log.close()


if __name__ == '__main__':
	run()
