import argparse

from .common.preprocessing.io import load_recording_data, validate_recording_data

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('recording_title', type = str)
	arg_parser.add_argument('-d', '--data-dir', type = str, default = '.')

	args = arg_parser.parse_args()

	try:
		recording_data = load_recording_data(data_path = args.data_dir, recording_title = args.recording_title)
		validate_recording_data(recording_data)
		print('Audio and labels OK')
	except FileNotFoundError as error:
		print(f'ERROR: Recording {args.recording_title} not found in directory {args.data_dir}: {error}')
	except ValueError as error:
		print(f'ERROR: Data validation failed: {error}')
