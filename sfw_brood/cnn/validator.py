from pathlib import Path

import pandas as pd

from sfw_brood.model import ModelValidator, SnowfinchBroodClassifier
from sfw_brood.validation import generate_validation_results


class CNNValidator(ModelValidator):
	def __init__(self, test_data: pd.DataFrame, label: str, n_workers: int):
		self.test_data = test_data
		self.label = label
		self.n_workers = n_workers

	def validate(self, model: SnowfinchBroodClassifier, output = '', multi_target = False) -> dict:
		print(f'Performing CNN validation, multi target = {multi_target}')

		rec_files = list(self.test_data.index)
		classes = list(self.test_data.columns)

		print(f'Running test prediction for {len(rec_files)} samples')
		pred_df = model.predict(rec_files, n_workers = self.n_workers)
		pred_classes = sorted(set(classes).intersection(set(pred_df.columns)))
		print(f'Classes present in prediction output: {pred_classes}')

		if output:
			out_path = Path(output)
			out_path.mkdir(parents = True, exist_ok = True)
			self.test_data.to_csv(out_path.joinpath('true.csv'))
			pred_df.to_csv(out_path.joinpath('pred.csv'))

		return generate_validation_results(
			test_df = self.test_data.loc[pred_df.file, pred_classes],
			pred_df = pred_df[pred_classes],
			classes = pred_classes,
			target_label = self.label,
			output = output,
			multi_target = multi_target
		)
