import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

from sfw_brood.model import ModelValidator, SnowfinchBroodClassifier


class CNNValidator(ModelValidator):
	def __init__(self, test_data: pd.DataFrame, label: str):
		self.test_data = test_data
		self.label = label

	def validate(self, model: SnowfinchBroodClassifier, output = '') -> float:
		rec_files = list(self.test_data.index)
		classes = list(self.test_data.columns)

		print('Running test prediction')
		pred_df = model.predict(rec_files)
		print('Test predictions made, checking accuracy')

		pred_classes = set(classes).intersection(set(pred_df.columns))

		print(f'Classes present in prediction output: {pred_classes}')

		y_pred = pred_df[pred_classes].idxmax(axis = 1)
		y_true = self.test_data.loc[pred_df.file][classes].idxmax(axis = 1)

		if output:
			print('Generating classification report and confusion matrix')

			y_pred.to_csv(f'{output}/y-pred.csv')
			y_true.to_csv(f'{output}/y-true.csv')

			report = classification_report(y_true, y_pred, output_dict = True)
			report_df = pd.DataFrame(report).transpose()

			ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
			plt.xlabel(f'Predicted {self.label}')
			plt.ylabel(f'True {self.label}')

			if output == 'show':
				plt.show()
				print(report_df)
			else:
				plt.savefig(f'{output}/confusion-matrix.png')
				report_df.to_csv(f'{output}/clf-report.csv')

			print(f'Classification report and confusion matrix saved to {output}')

		return accuracy_score(y_true, y_pred)
