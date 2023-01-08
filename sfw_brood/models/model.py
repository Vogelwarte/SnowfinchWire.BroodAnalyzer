from abc import ABC, abstractmethod

import pandas as pd


class SnowfinchBroodClassifier(ABC):
	@abstractmethod
	def predict(self, recording_paths: list[str]) -> pd.DataFrame:
		pass

	@abstractmethod
	def serialize(self, path: str):
		pass
