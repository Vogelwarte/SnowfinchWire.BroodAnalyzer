from opensoundscape.torch.models.cnn import CNN

from .model import SnowfinchBroodClassifier


class SnowfinchBroodCNN(SnowfinchBroodClassifier):
	def __init__(self, trained_cnn: CNN):
		self.cnn = trained_cnn

	def predict(self, recording_paths: list[str]) -> list[int]:
		print('Prediction not implemented')
		return [0] * len(recording_paths)

	def serialize(self, path: str):
		self.cnn.save(path)
