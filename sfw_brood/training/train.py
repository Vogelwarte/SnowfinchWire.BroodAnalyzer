import os
from pathlib import Path

import pandas as pd

from sfw_brood.common.preprocessing.io import load_recording_data, validate_recording_data
from sfw_brood.preprocessing import discover_training_data, prepare_training_data


def preprocess_data(data_dir: str, training_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
	train_dataset = discover_training_data(data_dir)
	bs_train_df = pd.DataFrame()
	ba_train_df = pd.DataFrame()

	for file in train_dataset.files:
		print(f'Loading recording {file.stem}')
		recording = load_recording_data(Path(file))
		validate_recording_data(recording)

		bs_df, ba_df = prepare_training_data(
			recording, train_dataset.brood_sizes, train_dataset.brood_ages,
			work_dir = training_dir, slice_duration_sec = 2.0
		)

		bs_train_df = pd.concat([bs_train_df, bs_df])
		ba_train_df = pd.concat([ba_train_df, ba_df])

	return bs_train_df, ba_train_df


def cleanup(training_dir: str):
	if Path(training_dir).exists():
		for file in os.listdir(training_dir):
			os.remove(f'{training_dir}/{file}')
		os.rmdir(training_dir)


if __name__ == '__main__':
	training_dir = '../../_training'
	bs_train_data, ba_train_data = preprocess_data(
		data_dir = os.getenv('DATA_PATH', default = '../../_data'),
		training_dir = training_dir
	)

	print(f'Brood size training data shape: {bs_train_data.shape}')
	print(f'Brood age training data shape: {ba_train_data.shape}')

	cleanup(training_dir)
