import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, \
	multilabel_confusion_matrix, label_ranking_average_precision_score

from sfw_brood.model import ModelValidator, SnowfinchBroodClassifier


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

		# y_pred = pred_df[pred_classes].idxmax(axis = 1)
		y_pred = pred_df[pred_classes]
		# y_true = self.test_data.loc[pred_df.file, classes].idxmax(axis = 1)
		y_true = self.test_data.loc[pred_df.file, pred_classes]

		y_pred.to_csv(f'{output}/y-pred.csv')
		y_true.to_csv(f'{output}/y-true.csv')

		if output:
			print('Generating classification report and confusion matrix')

			if multi_target:
				report_df = None
				multi_confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)

				n_classes = len(pred_classes)
				fig, ax = plt.subplots(1, n_classes, figsize = (6, 6))
				for axes, cm, label in zip(ax.flatten(), multi_confusion_matrix, pred_classes):
					cm_disp = ConfusionMatrixDisplay(cm)
					cm_disp.plot(xticks_rotation = 'vertical', ax = axes, colorbar = False, values_format = 'd')
					axes.set_title(label)

				fig.tight_layout()
			else:
				report = classification_report(y_true, y_pred, output_dict = True)
				report_df = pd.DataFrame(report).transpose()

				ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
				plt.xlabel(f'Predicted {self.label}')
				plt.ylabel(f'True {self.label}')

			if output == 'show':
				plt.show()
				if report_df is not None:
					print(report_df)
			else:
				plt.savefig(f'{output}/confusion-matrix.png')
				if report_df is not None:
					report_df.to_csv(f'{output}/clf-report.csv')
				pred_df.to_csv(f'{output}/pred.csv')

			print(f'Classification report and confusion matrix saved to {output}')

		if multi_target:
			return {
				'subset_accuracy': accuracy_score(y_true, y_pred),
				'label_ranking_precision': label_ranking_average_precision_score(y_true = y_true, y_score = y_pred)
			}

		return {
			'accuracy': accuracy_score(y_true, y_pred)
		}
