import pandas as pd
from opensoundscape.torch.models.cnn import CNN
from sklearn.model_selection import train_test_split

from sfw_brood.models.cnn import SnowfinchBroodCNN
from sfw_brood.models.model import SnowfinchBroodClassifier
from sfw_brood.train.trainer import ModelTrainer


class CNNTrainer(ModelTrainer):
	def _do_training_(self, train_data: pd.DataFrame) -> SnowfinchBroodClassifier:
		print('Training CNN...')

		cnn = CNN(
			architecture = 'resnet18',
			sample_duration = self.sample_duration_sec,
			classes = train_data.columns
		)

		train_df, validation_df = train_test_split(train_data, test_size = 0.2)
		# cnn.train(train_df, validation_df)

		return SnowfinchBroodCNN(cnn)
