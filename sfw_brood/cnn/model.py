import pandas as pd
from opensoundscape.torch.models.cnn import CNN, load_model

from sfw_brood.model import SnowfinchBroodClassifier, ModelType, ModelLoader


class SnowfinchBroodCNN(SnowfinchBroodClassifier):
	def __init__(self, trained_cnn: CNN, model_info: dict):
		super().__init__(ModelType.CNN, model_info)
		self.cnn = trained_cnn

	def predict(self, recording_paths: list[str], n_workers: int = 12) -> pd.DataFrame:
		pred_result = self.cnn.predict(recording_paths, activation_layer = 'softmax', num_workers = n_workers)
		result_df = pred_result[0] if type(pred_result) == tuple else pred_result
		print(f'Predictions made, result df shape = {result_df.shape}')
		return result_df.reset_index()

	def _serialize_(self, path: str):
		self.cnn.save(path)


class CNNLoader(ModelLoader):
	def __init__(self):
		super().__init__(ModelType.CNN)

	def _deserialize_model_(self, path: str, meta_data: dict) -> SnowfinchBroodClassifier:
		cnn = load_model(path)
		return SnowfinchBroodCNN(cnn, meta_data)
