from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from sfw_brood.model import ModelValidator, SnowfinchBroodClassifier
from sfw_brood.validation import display_confusion_matrix, save_confusion_matrix, save_test_summary


class SimpleClfValidator(ModelValidator):
	def __init__(self, test_data: pd.DataFrame, label_columns: List[str]):
		self.test_data = test_data
		self.label_columns = label_columns

	def validate(self, model: SnowfinchBroodClassifier, output = '', multi_target = False) -> dict:
		print(f'Performing simple size CLF validation')

		pred_df = model.predict(self.test_data, n_workers = 0)
		out_dir = Path(output)
		clf_scores = { }

		for label_col in self.label_columns:
			pred_labels = pred_df[label_col]
			true_labels = self.test_data[label_col]
			classes = list(true_labels.unique())

			accuracy = accuracy_score(true_labels, pred_labels)
			print(f'Simple CLF accuracy for labels {label_col}: {accuracy}')
			clf_scores[label_col] = accuracy

			cm = confusion_matrix(true_labels, pred_labels)
			display_confusion_matrix(cm, title = 'size', classes = classes)
			save_confusion_matrix(cm, out_dir.joinpath(label_col))

		save_test_summary(target = 'size', classes = self.label_columns, scores = clf_scores, out_dir = out_dir)
		pred_df.to_csv(out_dir.joinpath('pred.csv'))

		return clf_scores
