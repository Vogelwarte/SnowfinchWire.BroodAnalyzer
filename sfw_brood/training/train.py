import os
from pathlib import Path

import pandas as pd

from sfw_brood.common.preprocessing.io import load_recording_data, validate_recording_data
from sfw_brood.preprocessing import discover_training_data, prepare_training_data

if __name__ == '__main__':
	data_dir = os.getenv('DATA_PATH', default = '../../_data')

	train_dataset = discover_training_data(data_dir)
	bs_train_df = pd.DataFrame()
	ba_train_df = pd.DataFrame()

	for file in train_dataset.files:
		print(f'Loading recording {file.stem}')
		recording = load_recording_data(Path(file))
		# validate_recording_data(recording)

		bs_df, ba_df = prepare_training_data(
			recording, train_dataset.brood_sizes, train_dataset.brood_ages,
			work_dir = '../../_data/train', slice_duration_sec = 2.0
		)

		bs_train_df = pd.concat([bs_train_df, bs_df])
		ba_train_df = pd.concat([ba_train_df, ba_df])

	print(f'Brood size training data shape: {bs_train_df.shape}')
	print(f'Brood age training data shape: {ba_train_df.shape}')
