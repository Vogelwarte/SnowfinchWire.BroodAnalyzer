from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from sfw_brood.inference.core import SnowfinchBroodPrediction, aggregate_by_brood_periods
from sfw_brood.inference.util import assign_recording_periods
from sfw_brood.preprocessing import classes_to_1hot, map_class_to_group
from sfw_brood.simple_size_clf.model import SimpleBroodSizeClassifier
from sfw_brood.simple_size_clf.preprocessing import prepare_feeding_data
from sfw_brood.validation import normalize_confusion_matrix, display_confusion_matrix, save_confusion_matrix, \
	save_test_summary, check_accuracy_per_brood


def __parse_age_class__(age_class: str) -> Tuple[float, float]:
	low, high = age_class.split('-')
	return float(low), float(high)


def prepare_age_data(age_data_path: Path) -> pd.DataFrame:
	age_data = pd.read_csv(age_data_path)
	age_data[['age_min', 'age_max']] = age_data.apply(
		lambda row: __parse_age_class__(row['class']),
		axis = 1, result_type = 'expand'
	)
	return age_data[['brood_id', 'period_start', 'age_min', 'age_max']] \
		.rename(columns = { 'period_start': 'datetime' })


class SimpleSizeInference:
	def __init__(self, model: SimpleBroodSizeClassifier):
		self.model = model

	def predict(
			self, feeding_stats_path: Path, age_pred_path: Path, period_hours: int,
			overlap_hours = 0, period_map = None
	) -> List[SnowfinchBroodPrediction]:
		age_data = prepare_age_data(age_pred_path)
		feeding_data = prepare_feeding_data(feeding_stats_path, brood_data = age_data)
		hourly_preds = self.model.predict(feeding_data, n_workers = 0)
		out_preds = []

		for clf_name in self.model.classification_modes:
			clf_preds = hourly_preds[['brood_id', 'datetime', clf_name]].rename(columns = { clf_name: 'class' })
			classes = list(clf_preds['class'].unique())
			clf_preds = classes_to_1hot(clf_preds)

			clf_preds['n_samples'] = 1
			for cls in classes:
				clf_preds.rename(columns = { cls: f'{cls}_n_samples' }, inplace = True)

			agg_preds = aggregate_by_brood_periods(
				clf_preds, classes, period_hours, overlap_hours, period_map = period_map
			)
			out_pred = SnowfinchBroodPrediction(
				model_name = f'SimpleSize__{clf_name}', target = 'size', classes = classes,
				sample_results = clf_preds, rec_results = None, brood_periods_results = agg_preds
			)

			out_preds.append(out_pred)

		return out_preds


def __classes_to_groups__(classes: list[str]) -> List[Tuple[float, float]]:
	groups = []
	for cls in classes:
		bounds = cls.split('-')
		if len(bounds) == 1:
			groups.append((int(bounds[0]), int(bounds[0])))
		else:
			groups.append((int(bounds[0]), int(bounds[1])))
	return groups


def __find_reference_info__(brood, dt, true_df):
	time_delta = abs(true_df['datetime'] - dt)
	time_delta[true_df['brood_id'] != brood] = np.NaN
	return true_df.iloc[time_delta.argmin()][['brood_size', 'period_start']]


def aggregate_test_data(test_df: pd.DataFrame, pred: SnowfinchBroodPrediction) -> Tuple[pd.DataFrame, pd.DataFrame]:
	size_test_df = pred.sample_results[['brood_id', 'datetime']]
	size_test_df[['size', 'period_start']] = size_test_df.apply(
		lambda row: __find_reference_info__(row['brood_id'], row['datetime'], test_df),
		axis = 'columns', result_type = 'expand'
	)

	size_groups = __classes_to_groups__(pred.classes)

	def map_size(size: float) -> str:
		return map_class_to_group(size, size_groups, group_labels = pred.classes)

	size_test_df['class'] = size_test_df['size'].apply(map_size)
	size_test_df = classes_to_1hot(size_test_df)
	for cls in pred.classes:
		if cls not in size_test_df.columns:
			size_test_df[cls] = 0

	agg_map = { 'datetime': 'count' }
	for bs in pred.classes:
		agg_map[bs] = 'sum'

	agg_cols = ['brood_id', 'period_start'] + list(agg_map.keys())
	size_test_agg = size_test_df[agg_cols].groupby(['brood_id', 'period_start']).agg(agg_map)
	idx = pred.brood_periods_results.set_index(['brood_id', 'period_start']).index
	size_test_agg = size_test_agg.loc[idx].reset_index()
	size_test_agg = size_test_agg.rename(columns = { 'datetime': 'sample_count' })
	size_test_agg['class'] = size_test_agg[pred.classes].idxmax(axis = 1)

	return size_test_df, size_test_agg


def __make_sample_id__(row):
	return f'{row["brood_id"]}_{row["datetime"]}'


# Output:
# - test-result.json with complex summary
# - for each class configuration:
# 	- confusion matrix
# 	- inference-pred
# 	- brood-scores
class SimpleSizeInferenceValidator:
	def __init__(self, period_hours: int, overlap_hours: int):
		self.period_hours = period_hours
		self.overlap_hours = overlap_hours

	def validate_inference(
			self, inference: SimpleSizeInference,
			feeding_stats_path: Path, age_pred_path: Path, brood_info_path: Path, out_path: Path
	):
		true_brood_data = pd.read_csv(brood_info_path)

		true_brood_data['datetime'] = pd.to_datetime(true_brood_data['datetime'])
		true_brood_data, period_map = assign_recording_periods(
			true_brood_data, period_hours = self.period_hours, overlap_hours = self.overlap_hours
		)

		preds = inference.predict(
			feeding_stats_path, age_pred_path,
			self.period_hours, self.overlap_hours, period_map
		)
		scores = { }

		for pred in preds:
			pred_out_dir = out_path.joinpath(pred.model_name)

			samples_test_data, test_data = aggregate_test_data(true_brood_data, pred)
			y_true = test_data['class']
			y_pred = pred.brood_periods_results['class']

			scores[pred.model_name] = accuracy_score(y_true, y_pred)

			cm = normalize_confusion_matrix(confusion_matrix(y_true, y_pred, labels = pred.classes))
			display_confusion_matrix(cm, title = 'size', classes = pred.classes)
			save_confusion_matrix(cm, out_dir = pred_out_dir)

			pred.save(pred_out_dir.joinpath('pred'), brood_period_truth = test_data)

			truth_out_dir = pred_out_dir.joinpath('truth')
			truth_out_dir.mkdir(parents = True, exist_ok = True)
			samples_test_data.to_csv(truth_out_dir.joinpath('samples-truth.csv'), index = False)
			test_data.to_csv(truth_out_dir.joinpath('prediods-truth.csv'), index = False)

			merge_df = test_data.sort_values(by = ['brood_id', 'period_start'])
			merge_df.columns = [
				col if col in ['brood_id', 'period_start'] else f'true_{col}' for col in merge_df.columns
			]
			preds_to_merge = pred.brood_periods_results.sort_values(by = ['brood_id', 'period_start'])
			merge_df = pd.concat(
				[merge_df, preds_to_merge.drop(columns = ['brood_id', 'period_start'])],
				axis = 1
			)
			merge_df.to_csv(pred_out_dir.joinpath('pred-test.csv'), index = False)

			brood_scores_path = pred_out_dir.joinpath('brood-scores')
			brood_scores_path.mkdir(parents = True, exist_ok = True)

			check_accuracy_per_brood(
				pred.brood_periods_results, test_data['class'],
				out_path = brood_scores_path.joinpath('from-periods.csv')
			)

			check_accuracy_per_brood(
				pred_df = pred.sample_results,
				true_values = samples_test_data['class'],
				out_path = brood_scores_path.joinpath('from-samples.csv')
			)

		save_test_summary(
			target = 'size', classes = [pred.model_name for pred in preds],
			scores = scores, out_dir = out_path,
		)
