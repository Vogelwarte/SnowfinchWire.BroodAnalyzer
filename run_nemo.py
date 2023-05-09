import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import torch
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics import Accuracy, ConfusionMatrix


def plot_confusion_matrix(cm: np.ndarray, labels: list, out_path: Path):
	fig, ax = plt.subplots(figsize = (6, 6))
	cm_disp = ConfusionMatrixDisplay(cm.round().astype('int'), display_labels = labels)
	cm_disp.plot(xticks_rotation = 'vertical', ax = ax, colorbar = False, values_format = 'd')
	fig.tight_layout()
	plt.savefig(out_path)


@torch.no_grad()
def test(model, config) -> Tuple[float, np.ndarray]:
	model.setup_test_data(config.model.test_ds)
	data_loader = model._test_dl

	n_classes = data_loader.dataset.num_classes
	accuracy = Accuracy(task = 'multiclass', num_classes = n_classes)
	confusion_matrix = ConfusionMatrix(task = 'multiclass', num_classes = n_classes)

	for batch in data_loader:
		audio_signal, audio_signal_len, labels, labels_len = batch
		logits = model(input_signal = audio_signal, input_signal_length = audio_signal_len)

		_, pred = logits.topk(1, dim=1, largest=True, sorted=True)
		pred = pred.squeeze()

		accuracy.update(pred, labels)
		confusion_matrix.update(pred, labels)

		print('.', end = '')

	print()
	print('Finished custom test step!')

	confusion_matrix = confusion_matrix.compute().detach().cpu().numpy()
	accuracy = accuracy.compute().detach().cpu()

	return accuracy, confusion_matrix


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-d', '--data-dir', type = str)
	arg_parser.add_argument('-c', '--config-path', type = str)
	arg_parser.add_argument('-n', '--n-epochs', type = int, default = 20)
	arg_parser.add_argument('-a', '--accelerator', type = str, default = 'gpu')
	arg_parser.add_argument('--from-checkpoint', type = str, default = None)
	args = arg_parser.parse_args()

	dataset_basedir = args.data_dir
	train_dataset = os.path.join(dataset_basedir, 'train_manifest.json')
	val_dataset = os.path.join(dataset_basedir, 'validation_manifest.json')
	test_dataset = os.path.join(dataset_basedir, 'test_manifest.json')

	config = OmegaConf.load(args.config_path)
	config = OmegaConf.to_container(config, resolve = True)
	config = OmegaConf.create(config)
	print(OmegaConf.to_yaml(config))

	config.model.train_ds.manifest_filepath = train_dataset
	config.model.validation_ds.manifest_filepath = val_dataset
	config.model.test_ds.manifest_filepath = test_dataset

	ckpt_path = args.from_checkpoint
	if ckpt_path:
		asr_model = nemo_asr.models.EncDecClassificationModel.load_from_checkpoint(ckpt_path)
		ckpt_path = Path(ckpt_path)
		exp_dir = ckpt_path.parent.joinpath(ckpt_path.stem)
		exp_dir.mkdir(exist_ok = True, parents = True)
	else:
		# Lets modify some trainer configs for this demo
		# Checks if we have GPU available and uses it
		accelerator = args.accelerator
		config.trainer.devices = 1
		config.trainer.accelerator = accelerator
		config.trainer.max_epochs = args.n_epochs

		# Remove distributed training flags
		config.trainer.strategy = None

		trainer = pl.Trainer(**config.trainer)

		exp_dir = exp_manager(trainer, config.get("exp_manager", None))
		asr_model = nemo_asr.models.EncDecClassificationModel(cfg = config.model, trainer = trainer)
		trainer.fit(asr_model)

	accuracy, cm = test(asr_model, config)
	print(f'Model accuracy = {accuracy}')

	plot_confusion_matrix(cm, labels = config.model.labels, out_path = exp_dir.joinpath('confusion-matrix.png'))


if __name__ == '__main__':
	main()
