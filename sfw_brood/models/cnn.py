import pandas as pd
from opensoundscape.torch.models.cnn import CNN

from .model import SnowfinchBroodClassifier


class SnowfinchBroodCNN(SnowfinchBroodClassifier):
	def __init__(self, trained_cnn: CNN):
		self.cnn = trained_cnn

	def predict(self, recording_paths: list[str]) -> pd.DataFrame:
		return self.cnn.predict(recording_paths, activation_layer = 'softmax', num_workers = 12)[0]

	def serialize(self, path: str):
		self.cnn.save(path)
