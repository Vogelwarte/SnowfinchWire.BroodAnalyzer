from pathlib import Path
from typing import Optional, Tuple

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
import pytorch_lightning as pl

from sfw_brood.model import ModelTrainer, classes_from_data_config
from sfw_brood.nemo.model import SnowfinchBroodMatchboxNet
from sfw_brood.nemo.util import make_dataset_path
from sfw_brood.nemo.validator import MatchboxNetValidator


class MatchboxNetTrainer(ModelTrainer):
	def __init__(
			self, dataset_path: str, audio_path: str, data_config: dict, sample_duration: float,
			n_epochs: int, n_workers = 12, batch_size = 128, learn_rate = 0.05, samples_per_class = 10_000,
			age_range: Optional[Tuple[float, float]] = None
	):
		self.data_config = data_config
		self.age_range = age_range
		self.audio_path = audio_path
		self.dataset_root = Path(dataset_path)
		self.sample_duration = sample_duration
		self.samples_per_class = samples_per_class
		if self.samples_per_class not in ['min', 'mean', 'max']:
			self.samples_per_class = int(self.samples_per_class)

		self.config = OmegaConf.load('config/matchboxnet-sfw-brood.yaml')
		self.config.trainer.devices = 1
		self.config.trainer.accelerator = 'gpu'
		self.config.trainer.strategy = None
		self.config.trainer.max_epochs = n_epochs
		self.config.model.lr = learn_rate
		self.config.model.batch_size = batch_size
		self.config.model.num_workers = n_workers

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		pass

	def train_model_for_size(self, out_dir: str):
		return self.__train_and_validate__(target = 'size', out_dir = out_dir)

	def train_model_for_age(self, out_dir: str):
		return self.__train_and_validate__(target = 'age', out_dir = out_dir)

	def __train_and_validate__(self, target: str, out_dir: str):
		model = self.__train_model__(target, out_dir)
		validator = MatchboxNetValidator(data_loader = model.network._test_dl, label = target)
		test_result = validator.validate(model, output = out_dir, multi_target = False)
		print(f'CNN test result: {test_result}')
		return model

	def __train_model__(self, target: str, out_dir: str):
		config = self.config.copy()
		config.model.labels = classes_from_data_config(self.data_config[target])
		config = OmegaConf.to_container(config, resolve = True)
		config = OmegaConf.create(config)

		dataset_path = make_dataset_path(
			self.dataset_root, self.data_config['id'], target, self.samples_per_class, self.age_range
		)
		config.model.train_ds.manifest_filepath = self.__manifest_path__(dataset_path, 'train')
		config.model.validation_ds.manifest_filepath = self.__manifest_path__(dataset_path, 'validation')
		config.model.test_ds.manifest_filepath = self.__manifest_path__(dataset_path, 'test')

		trainer = pl.Trainer(**config.trainer)
		exp_dir = exp_manager(trainer, config.get('exp_manager', None))
		print(f'Saving experiments progress in {exp_dir} directory')

		model = EncDecClassificationModel(cfg = config.model, trainer = trainer)
		trainer.fit(model)

		trained_model = SnowfinchBroodMatchboxNet(
			trained_net = model,
			model_info = {
				'target': target,
				'learning_rate': config.model.lr,
				'batch_size': config.model.batch_size,
				'sample_duration': self.sample_duration,
				'samples_per_class': self.samples_per_class,
				'train_epochs': trainer.current_epoch,
				'dataset': dataset_path.as_posix(),
				'data_config': self.data_config[target],
				'age_range': self.age_range,
				'multi_target': False,
				'mt_threshold': 0.0
			}
		)
		trained_model.serialize(Path(out_dir).joinpath('model').as_posix())

		return trained_model

	def __manifest_path__(self, dataset_root: Path, dataset_id: str) -> str:
		return dataset_root.joinpath(f'{dataset_id}_manifest.json').as_posix()
