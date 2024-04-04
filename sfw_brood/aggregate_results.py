import argparse
from pathlib import Path

from .validation import aggregate_results

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('result_dirs', type = str, nargs = '+')
	arg_parser.add_argument('-s', '--subdirs', action = 'store_true')
	arg_parser.add_argument('-o', '--out', type = str)
	args = arg_parser.parse_args()

	result_dirs = args.result_dirs
	if args.subdirs:
		sub_dirs = []
		for path in result_dirs:
			for sub_dir in Path(path).glob('*'):
				if sub_dir.is_dir():
					sub_dirs.append(sub_dir.as_posix())
		result_dirs = sub_dirs

	aggregate_results(result_dirs, out_dir = args.out)
