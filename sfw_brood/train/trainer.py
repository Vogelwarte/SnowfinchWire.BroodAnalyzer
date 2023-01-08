from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from sfw_brood.models.model import SnowfinchBroodClassifier


@dataclass
class TrainResult:
	model: Optional[SnowfinchBroodClassifier]
	validation_score: float


class ModelTrainer(ABC):
	@abstractmethod
	def train_model_for_size(self, validate = False) -> TrainResult:
		pass

	@abstractmethod
	def train_model_for_age(self, validate = False) -> TrainResult:
		pass
