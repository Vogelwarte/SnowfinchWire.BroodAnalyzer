from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sfw_brood.inference.core import Inference, SnowfinchBroodPrediction
from sfw_brood.inference.util import assign_recording_periods, reject_underrepresented_samples
from sfw_brood.preprocessing import group_ages, group_sizes
from sfw_brood.validation import generate_validation_results, check_accuracy_per_brood


def generate_scores_per_brood(
		pred: SnowfinchBroodPrediction, brood_period_truth: pd.DataFrame,
		samples_truth: pd.DataFrame, data_root: Union[Path, str], out_path: Path
):
	out_path.mkdir(parents = True, exist_ok = True)

	check_accuracy_per_brood(
		pred.brood_periods_results, brood_period_truth['class'],
		out_path = out_path.joinpath('from-periods.csv')
	)

	samples_pred_df = pred.sample_results
	samples_pred_df['rec_path'] = samples_pred_df['rec_path'].apply(
		lambda p: Path(p).relative_to(data_root).as_posix()
	)

	samples_true_df = samples_truth[~samples_truth['rec_path'].duplicated(keep = 'first')] \
		.set_index('rec_path').reindex(samples_pred_df['rec_path']).dropna().reset_index()

	samples_true_df.to_csv(out_path.joinpath('samples-true-df.csv'))
	samples_pred_df.to_csv(out_path.joinpath('samples-pred-df.csv'))

	check_accuracy_per_brood(
		pred_df = samples_pred_df[samples_pred_df['rec_path'].isin(samples_true_df['rec_path'])].reset_index(),
		true_values = samples_true_df['class'],
		out_path = out_path.joinpath('from-samples.csv')
	)


class InferenceValidator(ABC):
	def __init__(self, period_hours: int, overlap_hours: int, target: str, multi_target_threshold: float):
		self.period_hours = period_hours
		self.overlap_hours = overlap_hours
		self.target = target
		self.multi_target_threshold = multi_target_threshold

	def validate_inference(
			self, inference: Inference, test_data: pd.DataFrame, data_root: Path, output: str, n_workers = 10
	) -> dict:
		classes = inference.classes
		is_multi_target = inference.model.model_info['multi_target']
		audio_paths = [data_root.joinpath(path) for path in test_data['rec_path']]

		test_data['datetime'] = pd.to_datetime(test_data['datetime'])
		test_data, period_map = assign_recording_periods(
			test_data, period_hours = self.period_hours, overlap_hours = self.overlap_hours
		)
		samples_test_data, test_data = self._aggregate_test_data_(test_data, classes, is_multi_target)
		test_data['class'] = test_data[classes].idxmax(axis = 1)

		pred = inference.predict(
			audio_paths, n_workers, agg_period_hours = self.period_hours, overlap_hours = self.overlap_hours,
			period_map = period_map, multi_target_threshold = self.multi_target_threshold
		)

		out_path = Path(output)
		out_path.mkdir(parents = True, exist_ok = True)
		# test_data.sort_values(by = ['brood_id', 'period_start']).to_csv(out_path.joinpath('test.csv'))
		# pred_df.sort_values(by = ['brood_id', 'period_start']).to_csv(out_path.joinpath('pred.csv'))
		pred.save(out_path.joinpath('inference-pred'), brood_period_truth = test_data)

		pred.brood_periods_results = reject_underrepresented_samples(pred.brood_periods_results)
		brood_period_preds = pred.brood_periods_results.set_index(['brood_id', 'period_start'])
		test_data = test_data.set_index(['brood_id', 'period_start']).loc[brood_period_preds.index]

		merge_df = test_data.reset_index().sort_values(by = ['brood_id', 'period_start'])
		merge_df.columns = [f'test_{col}' for col in merge_df.columns]
		merge_df = pd.concat(
			[brood_period_preds[classes].reset_index().sort_values(by = ['brood_id', 'period_start']), merge_df],
			axis = 1
		)
		merge_df.to_csv(Path(output).joinpath('pred-test.csv'))

		generate_scores_per_brood(
			pred, brood_period_truth = test_data.reset_index(), samples_truth = samples_test_data,
			data_root = data_root, out_path = out_path.joinpath('brood-scores')
		)

		return generate_validation_results(
			test_df = test_data[classes],
			pred_df = brood_period_preds[classes],
			classes = classes,
			target_label = f'brood {self.target}',
			output = output,
			multi_target = is_multi_target
		)

	@abstractmethod
	def _aggregate_test_data_(
			self, test_data: pd.DataFrame, classes: list, multi_target = False
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
		pass


class BroodSizeInferenceValidator(InferenceValidator):
	def __init__(self, period_hours: int, overlap_hours: int, size_groups: Optional[List[Tuple[float, float]]] = None):
		super().__init__(period_hours, overlap_hours, target = 'size', multi_target_threshold = 0.0)
		self.size_groups = size_groups

	def _aggregate_test_data_(
			self, test_data: pd.DataFrame, classes: list, multi_target = False
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
		size_test_df = test_data \
			.drop(columns = ['age_min', 'age_max', 'datetime']) \
			.rename(columns = { 'brood_size': 'class' })

		if self.size_groups is None:
			size_classes = [int(cls) for cls in classes]
			size_test_df['class'] = size_test_df['class'].astype('category')
			size_test_df['class'] = size_test_df['class'].cat.set_categories(size_classes)

			size_1hot = pd.get_dummies(size_test_df['class'])
			size_test_df = pd.concat([size_test_df, size_1hot.astype('int')], axis = 1)
			size_test_df.columns = [str(col) for col in size_test_df.columns]
		else:
			size_test_df, _ = group_sizes(test_data, groups = self.size_groups)

		agg_map = { 'rec_path': 'count' }
		for bs in classes:
			agg_map[bs] = 'sum'

		agg_cols = ['brood_id', 'period_start'] + list(agg_map.keys())
		size_test_agg = size_test_df[agg_cols].groupby(['brood_id', 'period_start']).agg(agg_map).reset_index()
		size_test_agg = size_test_agg.rename(columns = { 'rec_path': 'rec_count' })

		for bs in classes:
			size_test_agg[bs] = np.where(size_test_agg[bs] / size_test_agg['rec_count'] > 0.8, 1, 0)

		return size_test_df, size_test_agg


class BroodAgeInferenceValidator(InferenceValidator):
	def __init__(
			self, period_hours: int, overlap_hours: int,
			age_groups: List[Tuple[float, float]], multi_target_threshold = 0.3
	):
		super().__init__(period_hours, overlap_hours, target = 'age', multi_target_threshold = multi_target_threshold)
		self.age_groups = age_groups

	def _aggregate_test_data_(
			self, test_data: pd.DataFrame, classes: list, multi_target = False
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

		if multi_target:
			for age_group in classes:
				age_test_agg[age_group] = np.where(
					age_test_agg[age_group] / age_test_agg['rec_count'] > self.multi_target_threshold, 1, 0
				)
		else:
			cls_max = age_test_agg[classes].idxmax(axis = 1)
			for age_group in classes:
				age_test_agg[age_group] = np.where(cls_max == age_group, 1, 0)

			age_test_df['n_classes'] = age_test_df[classes].sum(axis = 1)
			age_test_df = age_test_df[age_test_df['n_classes'] == 1]
			age_test_df['class'] = age_test_df[classes].idxmax(axis = 1)

		return age_test_df, age_test_agg
