import os
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import torch
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager

model_config = 'matchboxnet_3x1x64_v1.yaml'
data_dir = '/home/bartosz/nemo-data'


def main():
	dataset_path = 'google_speech_recognition_v1'
	dataset_basedir = os.path.join(data_dir, dataset_path)

	train_dataset = os.path.join(dataset_basedir, 'train_manifest.json')
	val_dataset = os.path.join(dataset_basedir, 'validation_manifest.json')
	test_dataset = os.path.join(dataset_basedir, 'validation_manifest.json')

	config_path = f'config/{model_config}'
	config = OmegaConf.load(config_path)
	config = OmegaConf.to_container(config, resolve = True)
	config = OmegaConf.create(config)
	print(OmegaConf.to_yaml(config))

	# Preserve some useful parameters
	labels = config.model.labels
	sample_rate = config.model.sample_rate

	config.model.train_ds.manifest_filepath = train_dataset
	config.model.validation_ds.manifest_filepath = val_dataset
	config.model.test_ds.manifest_filepath = test_dataset

	# Lets modify some trainer configs for this demo
	# Checks if we have GPU available and uses it
	accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
	config.trainer.devices = 1
	config.trainer.accelerator = accelerator

	# Reduces maximum number of epochs to 5 for quick demonstration
	config.trainer.max_epochs = 5

	# Remove distributed training flags
	config.trainer.strategy = None

	trainer = pl.Trainer(**config.trainer)

	exp_dir = exp_manager(trainer, config.get("exp_manager", None))
	asr_model = nemo_asr.models.EncDecClassificationModel(cfg = config.model, trainer = trainer)
	trainer.fit(asr_model)
	trainer.test(asr_model, ckpt_path = None)


if __name__ == '__main__':
	main()
