import pandas as pd
from opensoundscape.torch.models.cnn import CNN, load_model
from opensoundscape.metrics import predict_multi_target_labels, predict_single_target_labels

from sfw_brood.model import SnowfinchBroodClassifier, ModelType, ModelLoader


class SnowfinchBroodCNN(SnowfinchBroodClassifier):
	def __init__(self, trained_cnn: CNN, model_info: dict):
		super().__init__(ModelType.CNN, model_info)
		self.cnn = trained_cnn
		self.multi_target = model_info['multi_target']
		self.mt_threshold = model_info['mt_threshold'] if 'mt_threshold' in model_info else 0.9

	def predict(self, recording_paths: list[str], n_workers: int = 12) -> pd.DataFrame:
		activation = 'sigmoid' if self.multi_target else 'softmax'
		pred_result = self.cnn.predict(recording_paths, activation_layer = activation, num_workers = n_workers)
		result_df = pred_result[0] if type(pred_result) == tuple else pred_result

		if self.multi_target:
			result_df = predict_multi_target_labels(result_df, threshold = self.mt_threshold)
		else:
			result_df = predict_single_target_labels(result_df)

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
