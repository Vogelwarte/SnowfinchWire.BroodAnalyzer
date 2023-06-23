import json

import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix

from sfw_brood.model import ModelValidator, SnowfinchBroodClassifier
from sfw_brood.nemo.model import SnowfinchBroodMatchboxNet
from sfw_brood.validation import display_confusion_matrix, save_results


@torch.no_grad()
def extract_logits(model, dataloader):
	logits_buffer = []
	label_buffer = []

	# Follow the above definition of the test_step
	for batch in dataloader:
		audio_signal, audio_signal_len, labels, labels_len = batch
		logits = model(input_signal = audio_signal, input_signal_length = audio_signal_len)

		logits_buffer.append(logits)
		label_buffer.append(labels)
		print(".", end = '')
	print()

	print("Finished extracting logits !")
	logits = torch.cat(logits_buffer, 0)
	labels = torch.cat(label_buffer, 0)
	return logits, labels


def parse_manifest(manifest):
	data = []
	for line in manifest:
		line = json.loads(line)
		data.append(line)

	return data


class ReverseMapLabel:
	def __init__(self, data_loader):
		self.label2id = dict(data_loader.dataset.label2id)
		self.id2label = dict(data_loader.dataset.id2label)

	def __call__(self, pred_idx, label_idx):
		return self.id2label[pred_idx], self.id2label[label_idx]


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

		# cpu_model = model.network.cpu()
		# cpu_model.eval()
		# logits, labels = extract_logits(cpu_model, self.data_loader)
		# print("Logits:", logits.shape, "Labels :", labels.shape)
		#
		# acc = cpu_model._accuracy(logits = logits, labels = labels)
		# print("Accuracy : ", float(acc[0] * 100))
		#
		# rev_map = ReverseMapLabel(self.data_loader)
		# results = []
		#
		# probs = torch.softmax(logits, dim = -1)
		# probas, preds = torch.max(probs, dim = -1)
		# total_count = cpu_model._accuracy.total_counts_k[0]
		#
		# print(total_count.items())
