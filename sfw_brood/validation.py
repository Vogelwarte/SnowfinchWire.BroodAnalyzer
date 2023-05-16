import json
from math import ceil
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, label_ranking_average_precision_score, multilabel_confusion_matrix, \
	ConfusionMatrixDisplay, confusion_matrix


def display_confusion_matrix(cm: np.ndarray, title: str, classes: list[str], multi_cols = 3):
	if cm.ndim == 3:
		n_classes = len(classes)
		n_cm_cols = min(multi_cols, n_classes)
		n_cm_rows = ceil(n_classes / n_cm_cols)
		fig, ax = plt.subplots(n_cm_rows, n_cm_cols, figsize = (2 * n_cm_cols, 3 * n_cm_rows))

		for sub_cm, label, axes in zip(cm, classes, ax.flatten()):
			cm_disp = ConfusionMatrixDisplay(normalize_confusion_matrix(sub_cm))
			cm_disp.plot(ax = axes, colorbar = False)
			axes.set_title(label)

		for axes in ax.flatten()[cm.shape[0]:]:
			axes.axis('off')

		fig.tight_layout()
	else:
		fix, axes = plt.subplots()
		cm_disp = ConfusionMatrixDisplay(normalize_confusion_matrix(cm), display_labels = classes)
		cm_disp.plot(ax = axes, colorbar = False)
		plt.xlabel(f'Predicted {title}')
		plt.ylabel(f'True {title}')
		fix.tight_layout()


def __aggregate_results__(result_dirs: list[Path], out_dir: Path):
	clf_title = None
	classes = None
	cm_agg = None
	score_map = { }

	for path in result_dirs:
		path = Path(path)

		cm_path = path.joinpath('confusion-matrix.npy')
		if cm_path.exists():
			if cm_agg is None:
				cm_agg = np.load(cm_path.as_posix())
			else:
				cm_agg += np.load(cm_path.as_posix())

		summary_path = path.joinpath('test-result.json')
		if summary_path.exists():
			with open(summary_path) as summary_file:
				summary = json.load(summary_file)

			if clf_title is None:
				clf_title = summary['target']

			if classes is None:
				classes = summary['classes']

			for key in summary['result'].keys():
				if key not in score_map.keys():
					score_map[key] = []
				score_map[key].append(summary['result'][key])

	cm_agg /= len(result_dirs)
	score_agg = { }
	for key in score_map.keys():
		score_agg[f'{key}_mean'] = np.mean(score_map[key])
		score_agg[f'{key}_std'] = np.std(score_map[key])

	display_confusion_matrix(cm_agg, title = clf_title, classes = classes)
	save_results(
		clf_title, classes, score_agg, cm_agg, out_dir,
		extra_info = {
			'result_paths': [rd.as_posix() for rd in result_dirs]
		}
	)


def save_results(
		target: str, classes: list[str], scores: dict, cm: np.ndarray, out_dir: Union[Path, str],
		extra_info: Optional[dict] = None
):
	out_path = Path(out_dir)

	out_path.mkdir(parents = True, exist_ok = True)
	np.save(out_path.joinpath('confusion-matrix.npy').as_posix(), cm)
	plt.savefig(out_path.joinpath('confusion-matrix.png'))

	summary = {
		'target': target,
		'classes': classes,
		'result': scores
	}

	if extra_info is not None:
		summary.update(extra_info)

	with open(out_path.joinpath('test-result.json'), mode = 'wt') as result_file:
		json.dump(summary, result_file, indent = 4)


def aggregate_results(result_dirs: list[Union[str, Path]], out_dir: Union[str, Path]):
	out_dir = Path(out_dir)
	result_dirs = [Path(path) for path in result_dirs]

	__aggregate_results__(result_dirs, out_dir)

	inference_dirs = []
	for path in result_dirs:
		inference_path = path.joinpath('inference')
		if inference_path.is_dir():
			inference_dirs.append(inference_path)

	__aggregate_results__(inference_dirs, out_dir = out_dir.joinpath('inference'))


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
			cm = multilabel_confusion_matrix(y_true, y_pred).astype(float)
			for i in range(cm.shape[0]):
				cm[i] = normalize_confusion_matrix(cm[i])
		else:
			cm = normalize_confusion_matrix(confusion_matrix(y_true, y_pred, labels = classes))
		display_confusion_matrix(cm, title = target_label, classes = classes)

		if output == 'show':
			plt.show()
			print(result)
		else:
			save_results(target_label, classes, result, cm, out_dir = output)
			print(f'Classification report and confusion matrix saved to {output}')

	return result
