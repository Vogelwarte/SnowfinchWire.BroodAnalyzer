import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_test_args(
		setup: dict, out_dir: Path,
		data_path: Path, brood_data_path: Path,
		n_workers: int, agg_period = None, agg_overlap = None
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
		'-t', target, '-w', str(n_workers)
	]

	out_dir = 'inference'

	if agg_period is not None:
		args.extend(['-p', str(agg_period)])
		out_dir += f'-p{agg_period}'
		if agg_overlap is not None:
			args.extend(['--overlap-hours', str(agg_overlap)])
			out_dir += f'-o{agg_overlap}'
	elif target == 'age':
		args.extend(['-p', '12'])
		out_dir += '-p12'
	elif target == 'size':
		args.extend(['-p', '48', '--overlap-hours', '24'])
		out_dir += '-p48-o24'

	args.extend([
		'-o', train_out_dir.joinpath(out_dir).as_posix(),
		brood_data_path.as_posix()
	])

	return args


def parse_train_args(work_dir: Path, setup: dict, out: Path, rng_seed: int, n_workers: int) -> list[str]:
	sample_duration = setup['sample_duration']
	target = setup['target']

	args = [
		sys.executable, 'train_model.py', '-w', str(n_workers),
		'-n', str(setup['max_epochs']), '-a', setup['architecture'],
		'-b', str(setup['bs']), '-l', str(setup['lr']), '-d', str(sample_duration),
		'-t', target, '-e', 'feeding', '-c', f'config/{setup["data_config"]}',
		'--samples-per-class', str(setup['samples_per_class']),
		'--out', out.as_posix(),
		'--audio-path', work_dir.joinpath(f's{sample_duration}/audio').as_posix(),
		'--rng-seed', str(rng_seed)
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


def hash_experiment(experiment_setup: dict) -> str:
	return hashlib.md5(json.dumps(experiment_setup).encode()).hexdigest()
