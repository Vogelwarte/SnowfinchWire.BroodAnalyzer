import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, label_ranking_average_precision_score, multilabel_confusion_matrix, \
	ConfusionMatrixDisplay, confusion_matrix


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
	div = cm.sum(axis = 1, keepdims = True)
	div = np.where(div > 0, div, 1)
	return cm / div


def generate_validation_results(
		test_df: pd.DataFrame, pred_df: pd.DataFrame, classes: list, target_label: str,
		output = '', multi_target = False
) -> dict:
	if multi_target:
		y_pred = pred_df
		y_true = test_df
		result = {
			'subset_accuracy': accuracy_score(y_true, y_pred),
			'label_ranking_precision': label_ranking_average_precision_score(y_true = y_true, y_score = y_pred)
		}
	else:
		y_pred = pred_df.idxmax(axis = 1)
		y_true = test_df.idxmax(axis = 1)
		print(f'y_true = {y_true}')
		print(f'y_pred = {y_pred}')
		result = {
			'accuracy': accuracy_score(y_true, y_pred)
		}

	if output:
		print('Generating classification report and confusion matrix')

		if multi_target:
			multi_confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

			n_classes = len(classes)
			n_cm_cols = min(4, n_classes)
			n_cm_rows = ceil(n_classes / n_cm_cols)
			fig, ax = plt.subplots(n_cm_rows, n_cm_cols, figsize = (2 * n_cm_cols, 3 * n_cm_rows))

			for axes, cm, label in zip(ax.flatten(), multi_confusion_matrix, classes):
				cm_disp = ConfusionMatrixDisplay(normalize_confusion_matrix(cm))
				cm_disp.plot(ax = axes, colorbar = False)
				axes.set_title(label)

			fig.tight_layout()
		else:
			fix, axes = plt.subplots()
			cm = confusion_matrix(y_true, y_pred, labels = classes)
			cm_disp = ConfusionMatrixDisplay(normalize_confusion_matrix(cm), display_labels = classes)
			cm_disp.plot(ax = axes, colorbar = False)
			plt.xlabel(f'Predicted {target_label}')
			plt.ylabel(f'True {target_label}')
			fix.tight_layout()

		if output == 'show':
			plt.show()
			print(result)
		else:
			Path(output).mkdir(parents = True, exist_ok = True)
			plt.savefig(f'{output}/confusion-matrix.png')
			with open(f'{output}/test-result.json', mode = 'wt') as result_file:
				json.dump(result, result_file, indent = 4)

		print(f'Classification report and confusion matrix saved to {output}')

	return result
