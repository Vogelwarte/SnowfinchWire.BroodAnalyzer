from datetime import date
from pathlib import Path
from typing import Union, Tuple, List, Dict

import numpy as np
import pandas as pd

from sfw_brood.model import ModelTrainer
from sfw_brood.simple_size_clf.model import SimpleBroodSizeClassifier
from sfw_brood.simple_size_clf.ensemble_clf import EnsemleClassifier
from sfw_brood.simple_size_clf.validator import SimpleClfValidator


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
		return int(row['brood_size']), row['age_min'], row['age_max']
	return np.NaN, np.NaN, np.NaN


def prepare_feeding_data(feeding_stats_path: Union[str, Path], brood_data_path: Union[str, Path]) -> pd.DataFrame:
	feeding_stats = pd.read_csv(feeding_stats_path)
	feeding_stats['datetime'] = pd.to_datetime(feeding_stats['datetime'])
	feeding_stats['day'] = feeding_stats['datetime'].apply(lambda dt: dt.date())

	brood_df = pd.read_csv(brood_data_path)
	brood_df['datetime'] = pd.to_datetime(brood_df['datetime'])
	brood_df['day'] = brood_df['datetime'].apply(lambda dt: dt.date())

	feeding_stats[['size', 'age_min', 'age_max']] = feeding_stats.apply(
		lambda row: find_brood_metrics(row['brood_id'], row['day'], brood_df),
		axis = 1, result_type = 'expand'
	)
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


class SimpleClfTrainer(ModelTrainer):
	def __init__(self, data_path: str, data_config: dict, n_models: int, voting: str):
		self.data_path = Path(data_path)
		self.data_config = data_config['size']
		self.n_models = n_models
		self.voting = voting
		self.x_features = [
			'duration', 'feeding_count', 'age_min', 'age_max',
			'min_sin', 'min_cos', 'agg_duration', 'agg_feeding_count'
		]

		feeding_data = prepare_feeding_data(
			feeding_stats_path = self.data_path.joinpath('feeding-stats.csv'),
			brood_data_path = self.data_path.joinpath('snowfinch-broods.csv')
		)

		self.label_columns = generate_size_labels(
			feeding_data, class_configs = [
				[2, 3, 4, 5],
				[(2, 3), (4, 5)],
				[(2, 3), 4, 5],
				[2, (3, 4), 5],
				[2, 3, (4, 5)]
			]
		)

		self.train_data, self.test_data = split_dataset(feeding_data, test_broods = self.data_config['test'])

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		pass

	def train_model_for_size(self, out_dir: str):
		classifiers = []

		for label_col in self.label_columns:
			clf = EnsemleClassifier(n_models = self.n_models, svm = True, bayes = True, name = label_col)
			print(f'Training simple size CLF for label group {label_col}')
			clf.fit(self.train_data, self.x_features, y_col = label_col)
			classifiers.append(clf)

		trained_model = SimpleBroodSizeClassifier(
			classifiers,
			model_info = {
				'target': 'size',
				'n_models': self.n_models,
				'voting': self.voting,
				'classifiers': ['SVM', 'Bayessian'],
				'dataset': self.data_path.absolute().as_posix(),
				'data_config': self.data_config,
				'labels': self.label_columns,
				'features': self.x_features
			}
		)

		trained_model.serialize(Path(out_dir).joinpath('model').as_posix())

		model_validator = SimpleClfValidator(self.test_data, self.label_columns)
		model_validator.validate(trained_model, output = out_dir)

		return trained_model

	def train_model_for_age(self, out_dir: str):
		raise AssertionError('Simple size classifier does not support age classification')
