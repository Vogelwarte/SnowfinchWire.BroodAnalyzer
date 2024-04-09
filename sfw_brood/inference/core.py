import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from sfw_brood.common.preprocessing.io import read_audacity_labels
from sfw_brood.inference.util import assign_recording_periods
from sfw_brood.model import SnowfinchBroodClassifier, classes_from_data_config


def __search_by_period__(df: pd.DataFrame, period_start: datetime, column: str):
	target_df = df.loc[df['period_start'] == period_start, column]
	if len(target_df) == 1:
		return target_df.iloc[0]
	else:
		return np.NaN


@dataclass
class SnowfinchBroodPrediction:
	model_name: str
	target: str
	classes: List[str]
	sample_results: pd.DataFrame
	rec_results: Optional[pd.DataFrame]
	brood_periods_results: pd.DataFrame

	def save(self, out: Union[Path, str], brood_period_truth: Optional[pd.DataFrame] = None):
		out = Path(out)
		out.mkdir(parents = True, exist_ok = True)
		self.sample_results.to_csv(out.joinpath('sample-preds.csv'), index = False)
		self.brood_periods_results.sort_values(by = ['brood_id', 'period_start']).to_csv(
			out.joinpath('brood-period-preds.csv'), index = False
		)

		if self.rec_results is not None:
			self.rec_results \
				.sort_values(by = ['brood_id', 'datetime']) \
				.to_csv(out.joinpath('rec-preds.csv'), index = False)

		for brood in self.brood_periods_results['brood_id'].unique():
			brood_truth = None
			if brood_period_truth is not None:
				brood_truth = brood_period_truth[brood_period_truth['brood_id'] == brood]
			self.plot_brood_results(brood, out, brood_truth)

	def plot_brood_results(self, brood_id: str, out: Path, brood_truth: Optional[pd.DataFrame] = None):
		result_df = self.brood_periods_results[self.brood_periods_results['brood_id'] == brood_id]
		result_df['period_start'] = pd.to_datetime(result_df['period_start'])
		min_day = result_df['period_start'].min()
		n_days = (result_df['period_start'].max() - min_day).days + 1
		graph_dates = [min_day + timedelta(days = i) for i in range(n_days)]
		period_dfs = []

		for day in graph_dates:
			period_row = result_df[result_df['period_start'] == day]
			if len(period_row) == 0:
				cls_scores = [0] * len(self.classes)
			else:
				period_row = period_row.reset_index().iloc[0]
				period_n_samples = period_row['n_samples']
				if period_n_samples == 0:
					cls_scores = [0] * len(self.classes)
				else:
					cls_scores = [period_row[f'{cls}_n_samples'] / period_n_samples for cls in self.classes]

			period_df = pd.DataFrame(data = {
				'day': [day] * len(self.classes),
				'class': self.classes,
				'score': cls_scores
			})

			if brood_truth is not None:
				period_df['true_class_1'] = [__search_by_period__(brood_truth, day, 'class')] * len(self.classes)
				if 'class_2' in brood_truth.columns:
					period_df['true_class_2'] = [__search_by_period__(brood_truth, day, 'class_2')] * len(self.classes)

			period_dfs.append(period_df)

		graph_df = pd.concat(period_dfs).reset_index().drop(columns = 'index')
		fig, axes = plt.subplots()
		sns.scatterplot(graph_df, x = 'day', y = 'class', size = 'score', sizes = (0, 100), legend = False, ax = axes)
		if brood_truth is not None:
			sns.lineplot(graph_df, x = 'day', y = 'true_class_1', color = 'r', ax = axes)
			if 'true_class_2' in graph_df.columns:
				sns.lineplot(graph_df, x = 'day', y = 'true_class_2', color = 'r', ax = axes)

		plt.xticks(rotation = 90)
		plt.title(f'{self.target.capitalize()} of brood {brood_id} predicted by {self.model_name} model')
		axes.invert_yaxis()
		fig.tight_layout()
		plt.savefig(out.joinpath(f'{brood_id}.png'))


# if self.target == 'age':
# 	def parse_age_range(age_range: str) -> Tuple[float, float, float]:
# 		low, high = age_range.split('-')
# 		low = float(low)
# 		high = float(high)
# 		return low, (low + high) / 2, high
#
# 	result_df['pred_cls'] = result_df[self.classes].idxmax(axis = 1)
# 	result_df[['age_min', 'age_mean', 'age_max']] = result_df.apply(
# 		lambda row: parse_age_range(row['pred_cls']), result_type = 'expand', axis = 'columns'
# 	)
# 	fig, axes = plt.subplots()
# 	axes.plot(result_df['age_min'])
# 	axes.plot(result_df['age_mean'])
# 	axes.plot(result_df['age_max'])
# 	plt.xticks(np.arange(0, len(result_df)), labels = result_df['period_start'], rotation = 90)
# 	fig.tight_layout()
# 	plt.savefig(out.joinpath(f'{brood_id}-lines.png'))


def aggregate_by_brood_periods(
		pred_df: pd.DataFrame, classes: list, period_hours: int, overlap_hours = 0,
		multi_target = False, multi_target_threshold = 0.7, period_map = None
) -> pd.DataFrame:
	pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
	pred_df, _ = assign_recording_periods(
		pred_df, period_hours, overlap_hours = overlap_hours, period_map = period_map
	)

	# print(pred_df[['rec_path', 'brood_id', 'datetime', 'period_start', 'n_samples']])

	agg_map = { }
	test_cols = ['brood_id', 'period_start']
	if 'rec_path' in pred_df.columns:
		agg_map['rec_path'] = 'count'
		test_cols.append('rec_path')

	for col in pred_df.columns:
		if 'n_samples' in col:
			test_cols.append(col)
			agg_map[col] = 'sum'

	pred_agg_df = pred_df[test_cols].groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
	if 'rec_path' in pred_agg_df.columns:
		pred_agg_df.rename(columns = { 'rec_path': 'rec_count' }, inplace = True)

	if multi_target:
		for cls in classes:
			pred_agg_df[cls] = np.where(
				pred_agg_df[f'{cls}_n_samples'] / pred_agg_df['n_samples'] > multi_target_threshold, 1, 0
			)
	else:
		for cls in classes:
			pred_agg_df[cls] = pred_agg_df[f'{cls}_n_samples'] / pred_agg_df['n_samples']

		cls_max = pred_agg_df[classes].idxmax(axis = 1)
		for cls in classes:
			pred_agg_df[cls] = np.where(cls_max == cls, 1, 0)

	pred_agg_df['class'] = pred_agg_df[classes].idxmax(axis = 1)

	return pred_agg_df


class Inference:
	def __init__(self, model: SnowfinchBroodClassifier):
		self.model = model
		self.classes = classes_from_data_config(self.model.model_info['data_config'])

	def predict(
			self, paths: List[Path], n_workers: int, agg_period_hours: int,
			label_paths: Optional[List[Path]] = None, overlap_hours = 0,
			multi_target_threshold = 0.7, period_map = None
	) -> SnowfinchBroodPrediction:
		samples_df = self.__prepare_data__(paths, label_paths)
		print(f'Running predictions for {len(samples_df)} feeding samples')
		# print(samples_df)
		pred_df = self.model.predict(samples_df, n_workers = n_workers)
		# pred_df.to_csv('_inference-pred.csv')
		pred_df, agg_df = self.__format_predictions__(pred_df)
		brood_period_agg_df = self.__aggregate_by_brood_periods__(
			agg_df, agg_period_hours, overlap_hours, multi_target_threshold, period_map
		)
		return SnowfinchBroodPrediction(
			model_name = self.model.model_type.name,
			target = self.model.model_info['target'],
			classes = self.classes,
			sample_results = pred_df,
			rec_results = agg_df,
			brood_periods_results = brood_period_agg_df
		)

	def __aggregate_by_brood_periods__(
			self, pred_df: pd.DataFrame, period_hours: int, overlap_hours = 0,
			multi_target_threshold = 0.7, period_map = None
	) -> pd.DataFrame:
		return aggregate_by_brood_periods(
			pred_df, self.classes, period_hours, overlap_hours,
			self.model.model_info['multi_target'], multi_target_threshold, period_map
		)

	def __label_path_for_rec__(self, rec_path: Path) -> Path:
		return rec_path.parent.joinpath(f'{rec_path.stem}.txt')
		# return rec_path.parent.joinpath(f'predicted_{rec_path.stem}.txt')

	def __prepare_data__(self, audio_paths: List[Path], label_paths: Optional[List[Path]]) -> pd.DataFrame:
		if label_paths is None:
			label_paths = [path if path.is_dir() else self.__label_path_for_rec__(path) for path in audio_paths]
		if len(label_paths) != len(audio_paths):
			raise RuntimeError('There must be label path specified for every audio path')

		rec_paths = []
		rec_label_paths = []
		for audio_path, label_path in zip(audio_paths, label_paths):
			if audio_path.is_dir():
				print(f'Discovering recordings from {audio_path.as_posix()} directory')
				for fmt in ['wav', 'flac', 'WAV']:
					for file in audio_path.rglob(f'*.{fmt}'):
						rec_paths.append(file)
						label_file = self.__label_path_for_rec__(file.relative_to(audio_path))
						rec_label_paths.append(label_path.joinpath(label_file))
			else:
				rec_paths.append(audio_path)
				rec_label_paths.append(label_path)

		print(f'Extracting feeding samples from {len(rec_paths)} recordings')
		samples_df = pd.DataFrame()

		for rec_path, label_path in tqdm(zip(rec_paths, rec_label_paths), total = len(rec_paths), file = sys.stdout):
			# print(rec_path, label_path)
			# labels_file = next(Path(rec_path.parent).glob(f'predicted_{rec_path.stem}*.txt'), None)
			# if not labels_file:
			# 	continue
			if not label_path.exists():
				continue

			labels_list = read_audacity_labels(label_path)
			labels_df = pd.DataFrame(labels_list).convert_dtypes()
			if labels_df.empty:
				continue

			rec_df = labels_df[labels_df['label'] == 'feeding'].rename(
				columns = { 'start': 'start_time', 'end': 'end_time' }
			)
			rec_df['file'] = rec_path
			rec_df['start_time'] = rec_df['start_time'].astype(float)
			rec_df['end_time'] = rec_df['end_time'].astype(float)
			rec_df = rec_df.set_index(['file', 'start_time', 'end_time'])
			samples_df = pd.concat([samples_df, rec_df])

		return samples_df

	def __extract_brood_id__(self, rec_path: str) -> str:
		return Path(rec_path).parent.parent.stem

	def __extract_datetime__(self, rec_path: str) -> datetime:
		rec_name = Path(rec_path).stem
		return datetime(
			int(rec_name[:4]), int(rec_name[4:6]), int(rec_name[6:8]),
			int(rec_name[9:11]), int(rec_name[11:13]), int(rec_name[13:15])
		)

	def __format_predictions__(self, pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
		classes = [col for col in pred_df.columns if col not in ['start_time', 'end_time', 'file']]
		pred_df['class'] = pred_df[classes].idxmax(axis = 1)
		pred_df['duration'] = pred_df['end_time'] - pred_df['start_time']
		pred_df = pred_df.reset_index().rename(columns = { 'file': 'rec_path' })
		pred_df['brood_id'] = pred_df['rec_path'].apply(self.__extract_brood_id__)

		agg_map = {
			'index': 'count',
			'duration': 'sum'
		}

		for cls in classes:
			agg_map[cls] = 'sum'

		agg_cols = ['rec_path', 'brood_id'] + list(agg_map.keys())
		agg_df = pred_df[agg_cols].groupby(['rec_path', 'brood_id']).agg(agg_map)
		agg_df = agg_df.reset_index().rename(columns = { 'index': 'n_samples' })
		agg_df['datetime'] = agg_df['rec_path'].apply(self.__extract_datetime__)
		for cls in classes:
			agg_df[f'{cls}_score'] = agg_df[cls] / agg_df['n_samples']
			agg_df.rename(inplace = True, columns = { cls: f'{cls}_n_samples' })

		pred_df = pred_df[['rec_path', 'brood_id', 'start_time', 'end_time', 'class']]

		return pred_df, agg_df
