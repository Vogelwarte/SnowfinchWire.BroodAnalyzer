from typing import Optional

import pandas as pd
import numpy as np
from opensoundscape.data_selection import resample


def balance_data(data: pd.DataFrame, classes: list[str], samples_per_class: str) -> pd.DataFrame:
	class_samples = [np.count_nonzero(data[cls]) for cls in classes]

	if samples_per_class == 'min':
		return resample(data, n_samples_per_class = np.min(class_samples))
	elif samples_per_class == 'max':
		return resample(data, n_samples_per_class = np.max(class_samples))
	elif samples_per_class == 'mean':
		return resample(data, n_samples_per_class = round(np.mean(class_samples)))
	else:
		return resample(data, n_samples_per_class = int(samples_per_class))


def __format_data__(
		data: pd.DataFrame, audio_path: str, classes: list[str], cls_samples: Optional[str] = None
) -> pd.DataFrame:
	data['file'] = audio_path + '/' + data['file']
	data = data.set_index('file')
	if cls_samples:
		return balance_data(data[classes], classes, cls_samples)
	return data[classes]


def select_recordings(
		data: pd.DataFrame, audio_path: str, cls_samples: str, split_conf: dict,
		classes: Optional[list[str]] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	if classes is None:
		if 'classes' in split_conf.keys() and 'class' in data.columns:
			classes = [str(cls) for cls in split_conf['classes']]
			data = data[data['class'].astype('str').isin(classes)]
		else:
			classes = set()
			for cls_col in [col for col in data.columns if 'class' in col]:
				classes.update(data[cls_col].unique())

	classes = sorted(classes)
	selector = split_conf['selector']
	train_idx = np.full(len(data), True)

	if 'test' in split_conf:
		test_idx = data[selector].isin(split_conf['test'])
		test_df = __format_data__(data[test_idx], audio_path, classes)
		train_idx &= ~test_idx
	else:
		test_df = None

	if 'validation' in split_conf:
		val_idx = data[selector].isin(split_conf['validation'])
		val_df = __format_data__(data[val_idx], audio_path, classes)
		train_idx &= ~val_idx
	else:
		val_df = None

	train_df = __format_data__(data[train_idx], audio_path, classes, cls_samples)
	test_size = round(0.45 * len(train_df))
	val_size = round(0.2 * len(train_df))

	if test_df is not None and test_size < len(test_df):
		test_df = test_df.sample(n = test_size)

	if val_df is not None and val_size < len(val_df):
		val_df = val_df.sample(n = val_size)

	return train_df, val_df, test_df
