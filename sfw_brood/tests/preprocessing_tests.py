import math
import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from sfw_brood.common.preprocessing.io import SnowfinchNestRecording
from sfw_brood.preprocessing import prepare_training_data


class PrepareTrainingDataTests(TestCase):
	RECORDING_DURATION_SEC = 60
	RECORDING_SAMPLE_RATE = 48000
	CNN_SAMPLE_DURATION = 2
	BROOD_SIZE = 3
	BROOD_AGE = 10
	DATA_PATH = '../../_data'

	def setUp(self) -> None:
		self.recording = SnowfinchNestRecording(
			title = 'mock_recording',
			audio_data = np.random.random(self.RECORDING_SAMPLE_RATE * self.RECORDING_DURATION_SEC) * 2.0 - 1.0,
			audio_sample_rate = self.RECORDING_SAMPLE_RATE,
			labels = pd.DataFrame(),
			brood_size = self.BROOD_SIZE,
			brood_age = self.BROOD_AGE
		)

	def doCleanups(self) -> None:
		work_dir = f'{self.DATA_PATH}/{self.recording.title}'
		if Path(work_dir).exists():
			for file in os.listdir(work_dir):
				os.remove(f'{work_dir}/{file}')
			os.rmdir(work_dir)

	def test__returned_data_frame_shape(self):
		expected_n_rows = math.ceil(self.RECORDING_DURATION_SEC / self.CNN_SAMPLE_DURATION)
		expected_n_cols = self.BROOD_SIZE

		train_df = prepare_training_data(self.recording, self.DATA_PATH, self.CNN_SAMPLE_DURATION)

		self.assertIsInstance(train_df, pd.DataFrame)
		self.assertEqual((expected_n_rows, expected_n_cols), train_df.shape)

	def test__returned_data_frame_column_names(self):
		expected_columns = [str(i + 1) for i in range(self.BROOD_SIZE)]

		train_df = prepare_training_data(self.recording, self.DATA_PATH, self.CNN_SAMPLE_DURATION)

		self.assertEqual(len(expected_columns), len(train_df.columns))
		self.assertEqual(expected_columns, list(train_df.columns))
		self.assertEqual('file', train_df.index.name)

	def test__returned_data_frame_column_types(self):
		train_df = prepare_training_data(self.recording, self.DATA_PATH, self.CNN_SAMPLE_DURATION)

		self.assertEqual('object', train_df.index.dtype.name)

		for bs in range(1, self.BROOD_SIZE + 1):
			col_vals = set(train_df[str(bs)].unique())
			self.assertTrue(col_vals.issubset({ 0, 1 }))
