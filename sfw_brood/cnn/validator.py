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

		pred_df = model.predict(rec_files)
		pred_classes = set(classes).intersection(set(pred_df.columns))

		y_pred = pred_df[pred_classes].idxmax(axis = 1)
		y_true = self.test_data.loc[pred_df.file][classes].idxmax(axis = 1)

		if output:
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

		return accuracy_score(y_true, y_pred)
