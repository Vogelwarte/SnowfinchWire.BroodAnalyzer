from datetime import date
from pathlib import Path
from typing import Tuple, Optional, List, Union, Dict

import numpy as np
import pandas as pd


def get_daily_agg_vals(brood_id, day, feeding_stats_daily: pd.DataFrame) -> Tuple[float, float]:
	rows = feeding_stats_daily[(feeding_stats_daily['brood_id'] == brood_id) & (feeding_stats_daily['day'] == day)]
	if len(rows) > 0:
		row = rows.iloc[0]
		return row['duration'], row['feeding_count']
	return np.NaN, np.NaN


def find_brood_metrics(brood_id: str, day: date, brood_df: pd.DataFrame) -> Tuple[int, float, float]:
	rows = brood_df[(brood_df['brood_id'] == brood_id) & (brood_df['day'] == day)]
	if len(rows) > 0:
		row = rows.iloc[0]
		return row['age_min'], row['age_max'], int(row['brood_size']) if 'brood_size' in row else np.NaN
	return np.NaN, np.NaN, np.NaN


def prepare_feeding_data(feeding_stats_path: Path, brood_data_path: Path) -> pd.DataFrame:
	feeding_stats = pd.read_csv(feeding_stats_path)
	feeding_stats['datetime'] = pd.to_datetime(feeding_stats['datetime'])
	feeding_stats['day'] = feeding_stats['datetime'].apply(lambda dt: dt.date())

	brood_df = pd.read_csv(brood_data_path)
	brood_df['datetime'] = pd.to_datetime(brood_df['datetime'])
	brood_df['day'] = brood_df['datetime'].apply(lambda dt: dt.date())

	feeding_stats[['age_min', 'age_max', 'size']] = feeding_stats.apply(
		lambda row: find_brood_metrics(row['brood_id'], row['day'], brood_df),
		axis = 1, result_type = 'expand'
	)
	if feeding_stats['size'].isna().all():
		feeding_stats.drop(columns = 'size', inplace = True)
	feeding_stats = feeding_stats.dropna()
	feeding_stats = feeding_stats[(feeding_stats['size'] >= 2) & (feeding_stats['size'] <= 5)]

	feeding_stats_daily = feeding_stats[['brood_id', 'day', 'duration', 'feeding_count']] \
		.groupby(['brood_id', 'day']) \
		.agg({ 'duration': 'mean', 'feeding_count': 'sum' }) \
		.reset_index()

	feeding_stats[['agg_duration', 'agg_feeding_count']] = feeding_stats.apply(
		lambda row: get_daily_agg_vals(row['brood_id'], row['day'], feeding_stats_daily),
		axis = 1, result_type = 'expand'
	)

	feeding_stats['minute'] = feeding_stats['datetime'].apply(lambda dt: dt.hour * 60 + dt.minute)
	max_minute = feeding_stats['minute'].max()
	feeding_stats['min_sin'] = np.sin((2 * np.pi * feeding_stats['minute']) / max_minute)
	feeding_stats['min_cos'] = np.cos((2 * np.pi * feeding_stats['minute']) / max_minute)

	return feeding_stats


def split_dataset(data: pd.DataFrame, test_broods: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
	test_idx = data['brood_id'].isin(test_broods)
	return data[~test_idx], data[test_idx]


# modifies input data frame
def generate_size_labels(data: pd.DataFrame, class_configs: List[List[Union[int, Tuple[int, int]]]]) -> List[str]:
	label_columns = []
	for classes in class_configs:
		col_name, cls_map = make_size_class_map(classes)
		label_columns.append(col_name)
		data[col_name] = data['size'].apply(lambda bs: cls_map[bs])
	return label_columns


def make_size_class_map(classes: List[Union[int, Tuple[int, int]]]) -> Tuple[str, Dict[int, str]]:
	out_map = { }

	for size in range(2, 6):
		for cls in classes:
			if type(cls) == tuple:
				low, high = cls
				if low <= size <= high:
					label = f'{int(low)}-{int(high)}'
					out_map[int(size)] = label
					break
			elif size == cls:
				out_map[int(size)] = str(int(cls))
				break

	y_col = '__'.join(sorted(set(out_map.values())))

	return y_col, out_map
