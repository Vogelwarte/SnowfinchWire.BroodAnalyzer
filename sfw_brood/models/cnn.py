import time

import pandas as pd

from .model import SnowfinchBroodClassifier


class SnowfinchBroodCNN(SnowfinchBroodClassifier):
	def test(self, test_data_path: str) -> float:
		print('Testing not implemented')
		return 0.0

	def predict(self) -> int:
		print('Prediction not implemented')
		return 0

	def serialize(self, path: str):
		print('Serialization not implemented')

	def _do_training_(self, train_data: tuple[pd.DataFrame, pd.DataFrame]):
		print('Training model ... ', end = '')
		time.sleep(1.0)
		print('OK')
