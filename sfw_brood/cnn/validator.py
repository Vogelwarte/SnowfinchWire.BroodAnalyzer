import json
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, \
	multilabel_confusion_matrix, label_ranking_average_precision_score

from sfw_brood.model import ModelValidator, SnowfinchBroodClassifier
from sfw_brood.validation import generate_validation_results


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
	div = cm.sum(axis = 1, keepdims = True)
	div = np.where(div > 0, div, 1)
	return cm / div


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

		return generate_validation_results(
			test_df = self.test_data.loc[pred_df.file, pred_classes],
			pred_df = pred_df[pred_classes],
			classes = pred_classes,
			target_label = self.label,
			output = output,
			multi_target = multi_target
		)

	# if multi_target:
	# 	y_pred = pred_df[pred_classes]
	# 	y_true = self.test_data.loc[pred_df.file, pred_classes]
	# 	result = {
	# 		'subset_accuracy': accuracy_score(y_true, y_pred),
	# 		'label_ranking_precision': label_ranking_average_precision_score(y_true = y_true, y_score = y_pred)
	# 	}
	# else:
	# 	y_pred = pred_df[pred_classes].idxmax(axis = 1)
	# 	y_true = self.test_data.loc[pred_df.file, classes].idxmax(axis = 1)
	# 	result = {
	# 		'accuracy': accuracy_score(y_true, y_pred)
	# 	}
	#
	# if output:
	# 	print('Generating classification report and confusion matrix')
	#
	# 	if multi_target:
	# 		multi_confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
	#
	# 		n_classes = len(pred_classes)
	# 		n_cm_cols = min(4, n_classes)
	# 		n_cm_rows = ceil(n_classes / n_cm_cols)
	# 		fig, ax = plt.subplots(n_cm_rows, n_cm_cols, figsize = (2 * n_cm_cols, 3 * n_cm_rows))
	#
	# 		for axes, cm, label in zip(ax.flatten(), multi_confusion_matrix, pred_classes):
	# 			cm_disp = ConfusionMatrixDisplay(normalize_confusion_matrix(cm))
	# 			cm_disp.plot(ax = axes, colorbar = False)
	# 			axes.set_title(label)
	#
	# 		fig.tight_layout()
	# 	else:
	# 		fix, axes = plt.subplots()
	# 		cm = confusion_matrix(y_true, y_pred)
	# 		cm_disp = ConfusionMatrixDisplay(normalize_confusion_matrix(cm))
	# 		cm_disp.plot(ax = axes, colorbar = False)
	# 		plt.xlabel(f'Predicted {self.label}')
	# 		plt.ylabel(f'True {self.label}')
	# 		fix.tight_layout()
	#
	# 	if output == 'show':
	# 		plt.show()
	# 		print(result)
	# 	else:
	# 		plt.savefig(f'{output}/confusion-matrix.png')
	# 		with open(f'{output}/test-result.json', mode = 'wt') as result_file:
	# 			json.dump(result, result_file, indent = 4)
	#
	# 	print(f'Classification report and confusion matrix saved to {output}')
	#
	# return result
