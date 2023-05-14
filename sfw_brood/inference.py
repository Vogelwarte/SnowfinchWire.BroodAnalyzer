from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sfw_brood.common.preprocessing.io import read_audacity_labels
from sfw_brood.model import SnowfinchBroodClassifier
from sfw_brood.preprocessing import group_ages
from sfw_brood.validation import generate_validation_results


@dataclass
class SnowfinchBroodPrediction:
	raw: pd.DataFrame
	by_rec: pd.DataFrame
	by_brood_periods: pd.DataFrame

	def save(self, out: Union[Path, str]):
		out = Path(out)
		out.mkdir(parents = True, exist_ok = True)
		self.raw.to_csv(out.joinpath('sample-preds.csv'), index = False)
		self.by_rec.to_csv(out.joinpath('rec-preds.csv'), index = False)
		self.by_brood_periods.to_csv(out.joinpath('brood-period-preds.csv'), index = False)


class Inference:
	def __init__(self, model: SnowfinchBroodClassifier):
		self.model = model

	def predict(
			self, paths: List[Path], n_workers: int, agg_period_days: int,
			multi_target_threshold = 0.7, period_map = None
	) -> SnowfinchBroodPrediction:
		samples_df = self.__prepare_data__(paths)
		print(f'Running predictions for {len(samples_df)} samples:')
		print(samples_df)
		pred_df = self.model.predict(samples_df, n_workers = n_workers)
		pred_df.to_csv('_inference-pred.csv')
		pred_df, agg_df = self.__format_predictions__(pred_df)
		brood_period_agg_df = self.__aggregate_by_brood_periods__(
			agg_df, agg_period_days, multi_target_threshold, period_map
		)
		return SnowfinchBroodPrediction(pred_df, agg_df, brood_period_agg_df)

	@property
	def classes(self):
		try:
			model_data_config = self.model.model_info['data_config']
			if 'groups' in model_data_config.keys():
				return model_data_config['groups']
			return model_data_config['classes']
		except KeyError:
			raise RuntimeError('Invalid model data info: no information about classes')

	def __aggregate_by_brood_periods__(
			self, pred_df: pd.DataFrame, period_days: int, multi_target_threshold = 0.7, period_map = None
	) -> pd.DataFrame:
		pred_df['datetime'] = pd.to_datetime(pred_df['datetime'])
		pred_df, _ = assign_recording_periods(pred_df, period_days = period_days, period_map = period_map)

		agg_map = { 'rec_path': 'count' }
		test_cols = ['rec_path', 'brood_id', 'period_start']

		for col in pred_df.columns:
			if 'n_samples' in col:
				test_cols.append(col)
				agg_map[col] = 'sum'

		pred_agg_df = pred_df[test_cols].groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
		pred_agg_df = pred_agg_df.rename(columns = { 'rec_path': 'rec_count' })

		classes = self.classes
		if self.model.model_info['target'] == 'size':
			for bs in classes:
				pred_agg_df[bs] = pred_agg_df[f'{bs}_n_samples'] / pred_agg_df['n_samples']

			bs_max = pred_agg_df[classes].idxmax(axis = 1)
			for bs in classes:
				pred_agg_df[bs] = np.where(bs_max == bs, 1, 0)
		else:
			for age_group in classes:
				pred_agg_df[age_group] = np.where(
					pred_agg_df[f'{age_group}_n_samples'] / pred_agg_df['n_samples'] > multi_target_threshold, 1, 0
				)

		return pred_agg_df

	def __prepare_data__(self, audio_paths: List[Path]) -> pd.DataFrame:
		rec_paths = []

		for audio_path in audio_paths:
			if audio_path.is_dir():
				print(f'Inference: discovering recordings from {audio_path.as_posix()} directory')
				for fmt in ['wav', 'flac', 'WAV']:
					for file in audio_path.rglob(f'*.{fmt}'):
						rec_paths.append(file)
			else:
				rec_paths.append(audio_path)

		print(f'Inference: extracting audio samples from {len(rec_paths)} recordings')
		samples_df = pd.DataFrame()

		for rec_path in tqdm(rec_paths):
			labels_file = next(Path(rec_path.parent).glob(f'*{rec_path.stem}*.txt'), None)
			if not labels_file:
				continue

			labels_list = read_audacity_labels(labels_file)
			labels_df = pd.DataFrame(labels_list).convert_dtypes()
			if labels_df.empty:
				continue

			rec_df = labels_df[labels_df['label'] == 'feeding'].rename(
				columns = { 'start': 'start_time', 'end': 'end_time' }
			)
			rec_df['file'] = rec_path
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

	def __format_predictions__(self, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
		classes = [col for col in pred_df.columns if col not in ['start_time', 'end_time', 'file']]
		pred_df['predicted_class'] = pred_df[classes].idxmax(axis = 1)
		pred_df['duration'] = pred_df['end_time'] - pred_df['start_time']
		pred_df = pred_df.reset_index().rename(columns = { 'file': 'rec_path' })

		agg_map = {
			'index': 'count',
			'duration': 'sum'
		}

		for cls in classes:
			agg_map[cls] = 'sum'

		agg_cols = ['rec_path'] + list(agg_map.keys())
		agg_df = pred_df[agg_cols].groupby('rec_path').agg(agg_map)
		agg_df = agg_df.reset_index().rename(columns = { 'index': 'n_samples' })
		agg_df['brood_id'] = agg_df['rec_path'].apply(self.__extract_brood_id__)
		agg_df['datetime'] = agg_df['rec_path'].apply(self.__extract_datetime__)
		for cls in classes:
			agg_df[f'{cls}_score'] = agg_df[cls] / agg_df['n_samples']
			agg_df.rename(inplace = True, columns = { cls: f'{cls}_n_samples' })

		pred_df = pred_df[['rec_path', 'start_time', 'end_time', 'predicted_class']]

		return pred_df, agg_df


def assign_recording_periods(
		rec_df: pd.DataFrame, period_days: int, period_map: Optional[dict] = None
) -> tuple[pd.DataFrame, dict]:
	def calculate_period_start(rec_time, min_date):
		period_offset = (rec_time.date() - min_date).days // period_days
		return min_date + timedelta(days = period_days * period_offset)

	period_df = pd.DataFrame()
	period_map_out = { }

	for brood in rec_df['brood_id'].unique():
		brood_df = rec_df[rec_df['brood_id'] == brood]

		if period_map and brood in period_map.keys():
			period_start = period_map[brood]
		else:
			period_start = brood_df['datetime'].min().date()

		period_map_out[brood] = period_start
		brood_df['period_start'] = brood_df['datetime'].apply(lambda dt: calculate_period_start(dt, period_start))
		period_df = pd.concat([period_df, brood_df])

	return period_df, period_map_out


class InferenceValidator(ABC):
	def __init__(self, period_days: int, target: str, multi_target_threshold: float):
		self.period_days = period_days
		self.target = target
		self.multi_target_threshold = multi_target_threshold

	def validate_inference(
			self, inference: Inference, test_data: pd.DataFrame, data_root: Path,
			output = '', multi_target = False, n_workers = 10
	) -> dict:
		audio_paths = [data_root.joinpath(path) for path in test_data['rec_path']]

		test_data['datetime'] = pd.to_datetime(test_data['datetime'])
		test_data, period_map = assign_recording_periods(test_data, period_days = self.period_days)
		test_data = self._aggregate_test_data_(test_data, inference.classes)

		pred = inference.predict(
			audio_paths, n_workers, agg_period_days = self.period_days,
			period_map = period_map, multi_target_threshold = self.multi_target_threshold
		)
		pred_df = pred.by_brood_periods
		# pred.by_rec['datetime'] = pd.to_datetime(pred.by_rec['datetime'])
		# pred_df, _ = assign_recording_periods(pred.by_rec, period_days = self.period_days, period_map = period_map)
		# pred_df = self._aggregate_predictions_(pred_df).set_index(['brood_id', 'period_start'])

		if output:
			out_path = Path(output)
			out_path.mkdir(parents = True, exist_ok = True)
			test_data.to_csv(out_path.joinpath('test.csv'))
			pred_df.to_csv(out_path.joinpath('pred.csv'))
			pred.save(out_path.joinpath('inference-pred'))

		classes = inference.classes
		return generate_validation_results(
			test_df = test_data.set_index(['brood_id', 'period_start']).loc[pred_df.index, classes],
			pred_df = pred_df[classes],
			classes = classes,
			target_label = f'brood {self.target}',
			output = output,
			multi_target = multi_target
		)

	# @abstractmethod
	# def _classes_(self) -> list:
	# 	pass

	@abstractmethod
	def _aggregate_test_data_(self, test_data: pd.DataFrame, classes: list) -> pd.DataFrame:
		pass

# @abstractmethod
# def _aggregate_predictions_(self, pred_df: pd.DataFrame) -> pd.DataFrame:
# 	pass


class BroodSizeInferenceValidator(InferenceValidator):
	def __init__(self, period_days: int):
		super().__init__(period_days, target = 'size', multi_target_threshold = 0.0)

	# def _classes_(self) -> list:
	# 	return self.__classes__

	def _aggregate_test_data_(self, test_data: pd.DataFrame, classes: list) -> pd.DataFrame:
		size_test_df = test_data.drop(columns = ['age_min', 'age_max', 'datetime'])

		size_test_df['brood_size'] = size_test_df['brood_size'].astype('category')
		size_test_df['brood_size'] = size_test_df['brood_size'].cat.set_categories(classes)

		size_1hot = pd.get_dummies(size_test_df['brood_size'])
		size_test_df = pd.concat([size_test_df.drop(columns = 'brood_size'), size_1hot.astype('int')], axis = 1)

		agg_map = { 'rec_path': 'count' }
		for bs in classes:
			agg_map[bs] = 'sum'

		size_test_agg = size_test_df.groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
		size_test_agg = size_test_agg.rename(columns = { 'rec_path': 'rec_count' })

		for bs in classes:
			size_test_agg[bs] = np.where(size_test_agg[bs] / size_test_agg['rec_count'] > 0.8, 1, 0)

		return size_test_agg

# def _aggregate_predictions_(self, pred_df: pd.DataFrame) -> pd.DataFrame:
# 	agg_map = { 'rec_path': 'count' }
# 	test_cols = ['rec_path', 'brood_id', 'period_start']
#
# 	for col in pred_df.columns:
# 		if 'n_samples' in col:
# 			test_cols.append(col)
# 			agg_map[col] = 'sum'
#
# 	pred_agg_df = pred_df[test_cols].groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
# 	pred_agg_df = pred_agg_df.rename(columns = { 'rec_path': 'rec_count' })
#
# 	classes = self._classes_()
# 	for bs in classes:
# 		pred_agg_df[bs] = pred_agg_df[f'{bs}_n_samples'] / pred_agg_df['n_samples']
#
# 	bs_max = pred_agg_df[classes].idxmax(axis = 1)
# 	for bs in classes:
# 		pred_agg_df[bs] = np.where(bs_max == bs, 1, 0)
#
# 	return pred_agg_df


class BroodAgeInferenceValidator(InferenceValidator):
	def __init__(self, period_days: int, age_groups: list[tuple[float, float]], multi_target_threshold = 0.3):
		super().__init__(period_days, target = 'age', multi_target_threshold = multi_target_threshold)
		self.age_groups = age_groups
		# self.__classes__ = label_age_groups(age_groups)

	# def _classes_(self) -> list:
	# 	return self.__classes__

	def _aggregate_test_data_(self, test_data: pd.DataFrame, classes: list) -> pd.DataFrame:
		age_test_df, _ = group_ages(
			test_data.rename(columns = { 'age_min': 'class_min', 'age_max': 'class_max' }),
			groups = self.age_groups, multi_target = True
		)
		age_test_df = age_test_df.drop(columns = ['datetime', 'age_min', 'age_max'])
		agg_map = { 'rec_path': 'count' }
		for age_group in classes:
			agg_map[age_group] = 'sum'

		age_test_agg = age_test_df.groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
		age_test_agg = age_test_agg.rename(columns = { 'rec_path': 'rec_count' })

		for age_group in classes:
			age_test_agg[age_group] = np.where(age_test_agg[age_group] / age_test_agg['rec_count'] > 0.3, 1, 0)

		return age_test_agg

# def _aggregate_predictions_(self, pred_df: pd.DataFrame) -> pd.DataFrame:
# 	agg_map = { 'rec_path': 'count' }
# 	test_cols = ['rec_path', 'brood_id', 'period_start']
#
# 	for col in pred_df.columns:
# 		if 'n_samples' in col:
# 			test_cols.append(col)
# 			agg_map[col] = 'sum'
#
# 	pred_agg_df = pred_df[test_cols].groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
# 	pred_agg_df = pred_agg_df.rename(columns = { 'rec_path': 'rec_count' })
#
# 	for age_group in self._classes_():
# 		pred_agg_df[age_group] = np.where(
# 			pred_agg_df[f'{age_group}_n_samples'] / pred_agg_df['n_samples'] > self.multi_target_threshold, 1, 0
# 		)
#
# 	return pred_agg_df
