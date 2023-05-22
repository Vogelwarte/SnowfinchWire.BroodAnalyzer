from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix

from sfw_brood.model import ModelValidator, SnowfinchBroodClassifier
from sfw_brood.nemo.model import SnowfinchBroodMatchboxNet
from sfw_brood.validation import display_confusion_matrix, save_results


class MatchboxNetValidator(ModelValidator):
	def __init__(self, data_loader: DataLoader, label: str):
		self.data_loader = data_loader
		self.label = label

	def validate(self, model: SnowfinchBroodClassifier, output = '', multi_target = False) -> dict:
		if type(model) != SnowfinchBroodMatchboxNet:
			raise TypeError('Matchbox net validator can only validate matchbox net, invalid model type')

		n_classes = self.data_loader.dataset.num_classes
		accuracy = Accuracy(task = 'multiclass', num_classes = n_classes)
		confusion_matrix = ConfusionMatrix(task = 'multiclass', num_classes = n_classes)

		for batch in self.data_loader:
			audio_signal, audio_signal_len, labels, labels_len = batch
			pred = model.predict_for_audio_samples(audio_signal, audio_signal_len)
			accuracy.update(pred, labels)
			confusion_matrix.update(pred, labels)
			print('.', end = '')

		print()
		print('Finished custom test step!')

		confusion_matrix = confusion_matrix.compute().detach().cpu().numpy()
		accuracy = accuracy.compute().detach().cpu()

		classes = list(self.data_loader.dataset.labels)
		summary = { 'accuracy': float(accuracy) }

		display_confusion_matrix(confusion_matrix, title = self.label, classes = classes)
		save_results(
			target = self.label, classes = classes, scores = summary,
			cm = confusion_matrix, out_dir = output
		)

		return summary
