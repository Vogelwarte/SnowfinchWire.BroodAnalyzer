from typing import Optional

import pandas as pd
from opensoundscape.torch.models.cnn import CNN, load_model
from sklearn.model_selection import train_test_split

from sfw_brood.models.cnn import SnowfinchBroodCNN
from sfw_brood.models.model import SnowfinchBroodClassifier
from sfw_brood.train.trainer import ModelTrainer


class CNNTrainer(ModelTrainer):
	def _do_training_(self, train_data: pd.DataFrame) -> Optional[SnowfinchBroodClassifier]:
		if train_data.shape[0] == 0:
			print('No training data available')
			return None

		print('Training CNN...')

		cnn = CNN(
			architecture = 'resnet18',
			sample_duration = self.sample_duration_sec,
			classes = train_data.columns,
			single_target = True
		)

		train_df, validation_df = train_test_split(train_data, test_size = 0.2)
		cnn.train(train_df, validation_df, epochs = 5, batch_size = 100, save_path = f'{self.work_dir}/models')

		return SnowfinchBroodCNN(trained_cnn = load_model(f'{self.work_dir}/models/best.model'))
