import argparse
import os
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import torch
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-d', '--data-dir', type = str)
	arg_parser.add_argument('-n', '--n-epochs', type = int)
	arg_parser.add_argument('-c', '--config-path', type = str)
	arg_parser.add_argument('-a', '--accelerator', type = str, default = 'gpu')
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
	trainer.test(asr_model, ckpt_path = None)


if __name__ == '__main__':
	main()
