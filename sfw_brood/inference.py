import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from sfw_brood.cnn.util import cleanup
from sfw_brood.common.preprocessing.io import load_recording_data
from sfw_brood.model import SnowfinchBroodClassifier
from sfw_brood.preprocessing import filter_recording, group_ages, label_age_groups
from sfw_brood.validation import generate_validation_results


@dataclass
class SnowfinchBroodPrediction:
	raw: pd.DataFrame
	agg: pd.DataFrame

	def save(self, out: Union[Path, str]):
		out = Path(out)
		out.mkdir(parents = True, exist_ok = True)
		self.raw.to_csv(out.joinpath('raw.csv'), index = False)
		self.agg.to_csv(out.joinpath('agg.csv'), index = False)


class Inference:
	def __init__(self, model: SnowfinchBroodClassifier, work_dir: Path):
		self.model = model
		self.work_dir = work_dir

	def __enter__(self):
		self.work_dir.mkdir(parents = True, exist_ok = True)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		cleanup(self.work_dir)

	def predict(self, paths: List[Path], n_workers: int) -> SnowfinchBroodPrediction:
		sample_paths = self.__prepare_data__(paths, n_workers)
		print(f'Running predictions for {len(sample_paths)} samples')
		pred_df = self.model.predict(sample_paths, n_workers = n_workers)
		pred_df.to_csv('_inference-pred.csv')
		pred_df, agg_df = self.__format_predictions__(pred_df)
		return SnowfinchBroodPrediction(pred_df, agg_df)

	def __prepare_data__(self, audio_paths: List[Path], n_workers: int) -> list[str]:
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

		sample_paths = []
		with multiprocessing.Pool(n_workers) as proc_pool:
			for samples in tqdm(proc_pool.imap_unordered(self.__extract_samples__, rec_paths), total = len(rec_paths)):
				sample_paths.append(samples)

			# for rec_path in recordings:
			# 	try:
			# 		recording = load_recording_data(rec_path, include_brood_info = False)
			# 		audio_samples = filter_recording(recording, target_labels = ['feeding'])
			#
			# 		rec_path_rel = rec_path.relative_to(rec_path.root) if rec_path.is_absolute() else rec_path
			# 		work_dir = self.work_dir.joinpath(rec_path_rel.parent).joinpath(rec_path.stem)
			# 		work_dir.mkdir(exist_ok = True, parents = True)
			#
			# 		for i, (sample, _) in enumerate(audio_samples):
			# 			sample_path = work_dir.joinpath(f'{i}.wav')
			# 			sf.write(sample_path, sample, samplerate = recording.audio_sample_rate)
			# 			sample_paths.append(sample_path.as_posix())
			#
			# 	except FileNotFoundError:
			# 		print(f'Warning: failed to load recording {rec_path}')
			# 		continue

		return sample_paths

	def __extract_samples__(self, rec_path: Path) -> list[str]:
		try:
			recording = load_recording_data(rec_path, include_brood_info = False)
			audio_samples = filter_recording(recording, target_labels = ['feeding'])

			rec_path_rel = rec_path.relative_to(rec_path.root) if rec_path.is_absolute() else rec_path
			work_dir = self.work_dir.joinpath(rec_path_rel.parent).joinpath(rec_path.stem)
			work_dir.mkdir(exist_ok = True, parents = True)

			sample_paths = []
			for i, (sample, _) in enumerate(audio_samples):
				sample_path = work_dir.joinpath(f'{i}.wav')
				sf.write(sample_path, sample, samplerate = recording.audio_sample_rate)
				sample_paths.append(sample_path.as_posix())

			return sample_paths

		except FileNotFoundError:
			print(f'Warning: failed to load recording {rec_path}')
			return []

	def __extract_rec_path__(self, sample_path: str) -> str:
		sample_path = Path(sample_path)
		return sample_path.parent.relative_to(self.work_dir).as_posix()

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
		pred_df['rec_path'] = pred_df['file'].apply(self.__extract_rec_path__)

		agg_map = {
			'file': 'count',
			'duration': 'sum'
		}

		for cls in classes:
			agg_map[cls] = 'sum'

		agg_cols = ['rec_path'] + list(agg_map.keys())
		agg_df = pred_df[agg_cols].groupby('rec_path').agg(agg_map)
		agg_df = agg_df.reset_index().rename(columns = { 'file': 'n_samples' })
		agg_df['brood_id'] = agg_df['rec_path'].apply(self.__extract_brood_id__)
		agg_df['datetime'] = agg_df['rec_path'].apply(self.__extract_datetime__)
		for cls in classes:
			agg_df[f'{cls}_score'] = agg_df[cls] / agg_df['n_samples']
			agg_df.rename(inplace = True, columns = { cls: f'{cls}_n_samples' })

		pred_df = pred_df[['file', 'start_time', 'end_time', 'predicted_class']]

		return pred_df, agg_df


def assign_recording_periods(
		rec_df: pd.DataFrame, period_days: int, period_map: Optional[dict] = None
) -> tuple[pd.DataFrame, dict]:
	def calculate_period_start(rec_time, min_date):
		period_offset = (rec_time.date() - min_date).days // period_days
		return min_date + timedelta(days = period_days * period_offset)

	period_df = pd.DataFrame()
	period_map_out = {}

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
	def __init__(self, period_days: int, label: str):
		self.period_days = period_days
		self.label = label

	def validate_inference(
			self, inference: Inference, test_data: pd.DataFrame, data_root: Path,
			output = '', multi_target = False, n_workers = 10
	) -> dict:
		audio_paths = [data_root.joinpath(path) for path in test_data['rec_path']]

		test_data['datetime'] = pd.to_datetime(test_data['datetime'])
		test_data, period_map = assign_recording_periods(test_data, period_days = self.period_days)
		test_data = self._aggregate_test_data_(test_data)

		pred = inference.predict(audio_paths, n_workers)
		pred.agg['datetime'] = pd.to_datetime(pred.agg['datetime'])
		pred_df, _ = assign_recording_periods(pred.agg, period_days = self.period_days, period_map = period_map)
		pred_df = self._aggregate_predictions_(pred_df).set_index(['brood_id', 'period_start'])

		if output:
			out_path = Path(output)
			out_path.mkdir(parents = True, exist_ok = True)
			test_data.to_csv(out_path.joinpath('test.csv'))
			pred_df.to_csv(out_path.joinpath('pred.csv'))
			pred.save(out_path.joinpath('inference-pred'))

		classes = self._classes_()
		return generate_validation_results(
			test_df = test_data.set_index(['brood_id', 'period_start']).loc[pred_df.index, classes],
			pred_df = pred_df[classes],
			classes = classes,
			target_label = self.label,
			output = output,
			multi_target = multi_target
		)

	@abstractmethod
	def _classes_(self) -> list:
		pass

	@abstractmethod
	def _aggregate_test_data_(self, test_data: pd.DataFrame) -> pd.DataFrame:
		pass

	@abstractmethod
	def _aggregate_predictions_(self, pred_df: pd.DataFrame) -> pd.DataFrame:
		pass


class BroodSizeInferenceValidator(InferenceValidator):
	def __init__(self, period_days: int):
		super().__init__(period_days, label = 'brood size')

	def _classes_(self) -> list:
		return [2, 3, 4, 5]

	def _aggregate_test_data_(self, test_data: pd.DataFrame) -> pd.DataFrame:
		classes = self._classes_()
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

	def _aggregate_predictions_(self, pred_df: pd.DataFrame) -> pd.DataFrame:
		agg_map = { 'rec_path': 'count' }
		test_cols = ['rec_path', 'brood_id', 'period_start']

		for col in pred_df.columns:
			if 'n_samples' in col:
				test_cols.append(col)
				agg_map[col] = 'sum'

		pred_agg_df = pred_df[test_cols].groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
		pred_agg_df = pred_agg_df.rename(columns = { 'rec_path': 'rec_count' })

		classes = self._classes_()
		for bs in classes:
			pred_agg_df[bs] = pred_df[f'{bs}_n_samples'] / pred_df['n_samples']

		for bs in classes:
			bs_max = pred_agg_df[classes].idxmax(axis = 1)
			pred_agg_df[bs] = np.where(bs_max == bs, 1, 0)

		return pred_agg_df


class BroodAgeInferenceValidator(InferenceValidator):
	def __init__(self, period_days: int, age_groups: list[tuple[float, float]], multi_target_threshold = 0.3):
		super().__init__(period_days, label = 'brood age')
		self.age_groups = age_groups
		self.multi_target_threshold = multi_target_threshold
		self.__classes__ = label_age_groups(age_groups)

	def _classes_(self) -> list:
		return self.__classes__

	def _aggregate_test_data_(self, test_data: pd.DataFrame) -> pd.DataFrame:
		age_test_df, _ = group_ages(
			test_data.rename(columns = { 'age_min': 'class_min', 'age_max': 'class_max' }),
			groups = self.age_groups, multi_target = True
		)
		age_test_df = age_test_df.drop(columns = ['datetime', 'age_min', 'age_max'])
		agg_map = { 'rec_path': 'count' }
		for age_group in self.__classes__:
			agg_map[age_group] = 'sum'

		age_test_agg = age_test_df.groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
		age_test_agg = age_test_agg.rename(columns = { 'rec_path': 'rec_count' })

		for age_group in self.__classes__:
			age_test_agg[age_group] = np.where(age_test_agg[age_group] / age_test_agg['rec_count'] > 0.3, 1, 0)

		return age_test_agg

	def _aggregate_predictions_(self, pred_df: pd.DataFrame) -> pd.DataFrame:
		agg_map = { 'rec_path': 'count' }
		test_cols = ['rec_path', 'brood_id', 'period_start']

		for col in pred_df.columns:
			if 'n_samples' in col:
				test_cols.append(col)
				agg_map[col] = 'sum'

		pred_agg_df = pred_df[test_cols].groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
		pred_agg_df = pred_agg_df.rename(columns = { 'rec_path': 'rec_count' })

		for age_group in self._classes_():
			pred_agg_df[age_group] = np.where(
				pred_agg_df[f'{age_group}_n_samples'] / pred_agg_df['n_samples'] > self.multi_target_threshold, 1, 0
			)

		return pred_agg_df
