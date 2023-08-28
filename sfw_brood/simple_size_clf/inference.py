from pathlib import Path
from typing import Tuple, List

import pandas as pd

from sfw_brood.inference.core import SnowfinchBroodPrediction, aggregate_by_brood_periods
from sfw_brood.inference.util import assign_recording_periods
from sfw_brood.preprocessing import classes_to_1hot
from sfw_brood.simple_size_clf.model import SimpleBroodSizeClassifier
from sfw_brood.simple_size_clf.preprocessing import prepare_feeding_data


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
			self, inference: SimpleSizeInference, feeding_stats_path: Path,
			age_pred_path: Path, brood_info_path: Path, out_path: Path
	):
		test_data = pd.read_csv(brood_info_path)

		test_data['datetime'] = pd.to_datetime(test_data['datetime'])
		test_data, period_map = assign_recording_periods(
			test_data, period_hours = self.period_hours, overlap_hours = self.overlap_hours
		)
		# samples_test_data, test_data = self._aggregate_test_data_(test_data, classes, is_multi_target)
		# test_data['class'] = test_data[classes].idxmax(axis = 1)

		preds = inference.predict(feeding_stats_path, age_pred_path, self.period_hours, self.overlap_hours)

