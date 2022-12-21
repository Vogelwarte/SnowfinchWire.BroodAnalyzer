import math
import os
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from sfw_brood.common.preprocessing.io import SnowfinchNestRecording
from sfw_brood.preprocessing import prepare_training_data, slice_audio


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


class SliceAudioTests(TestCase):
	AUDIO_LEN_SEC = 60
	AUDIO_SAMPLE_RATE = 48000

	def setUp(self) -> None:
		self.audio = np.random.random(self.AUDIO_SAMPLE_RATE * self.AUDIO_LEN_SEC) * 2.0 - 1.0

	def test__returned_data_type(self):
		slice_len_sec = 4.0
		slices = slice_audio(self.audio, self.AUDIO_SAMPLE_RATE, slice_len_sec)
		self.assertIsInstance(slices, list)
		for s in slices:
			self.assertIsInstance(s, np.ndarray)

	def test__slices_count_no_overlap(self):
		slice_len_sec = 4.0
		slices = slice_audio(self.audio, self.AUDIO_SAMPLE_RATE, slice_len_sec)
		self.assertEqual(math.ceil(self.AUDIO_LEN_SEC / slice_len_sec), len(slices))

	def test__slices_count_with_overlap(self):
		slice_len_sec = 5.0
		overlap_sec = 2.0
		slices = slice_audio(self.audio, self.AUDIO_SAMPLE_RATE, slice_len_sec, overlap_sec)

		self.assertEqual(math.ceil(self.AUDIO_LEN_SEC / (slice_len_sec - overlap_sec)), len(slices))

	def test__slices_durations_no_overlap(self):
		slice_len_sec = 4.0
		expected_samples_per_slice = round(slice_len_sec * self.AUDIO_SAMPLE_RATE)

		slices = slice_audio(self.audio, self.AUDIO_SAMPLE_RATE, slice_len_sec)
		for i in range(len(slices) - 1):
			self.assertEqual(expected_samples_per_slice, len(slices[i]))
		self.assertLessEqual(len(slices[-1]), expected_samples_per_slice)

	def test__slices_durations_with_overlap(self):
		slice_len_sec = 4.0
		expected_samples_per_slice = round(slice_len_sec * self.AUDIO_SAMPLE_RATE)

		slices = slice_audio(self.audio, self.AUDIO_SAMPLE_RATE, slice_len_sec, overlap_sec = 1.0)
		for i in range(len(slices) - 1):
			self.assertEqual(expected_samples_per_slice, len(slices[i]))
		self.assertLessEqual(len(slices[-1]), expected_samples_per_slice)

	def test__slices_overlap(self):
		slice_len_sec = 4.0
		overlap_sec = 1.0
		slices = slice_audio(self.audio, self.AUDIO_SAMPLE_RATE, slice_len_sec, overlap_sec)

		overlap_samples = round(overlap_sec * self.AUDIO_SAMPLE_RATE)

		for i in range(len(slices) - 1):
			s1 = slices[i]
			s2 = slices[i + 1]
			self.assertTrue(all(s1[-overlap_samples:] == s2[:overlap_samples]))
