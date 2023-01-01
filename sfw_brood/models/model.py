from abc import ABC, abstractmethod


class SnowfinchBroodClassifier(ABC):
	@abstractmethod
	def predict(self, recording_paths: list[str]) -> list[int]:
		pass

	@abstractmethod
	def serialize(self, path: str):
		pass
