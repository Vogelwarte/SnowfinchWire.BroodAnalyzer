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
	BROOD_SIZES = [2, 3, 4]
	BROOD_AGES = list(np.linspace(3, 20, num = 18, dtype = 'int'))
	DATA_PATH = '../../_data/test'

	def setUp(self) -> None:
		self.recording = SnowfinchNestRecording(
			title = 'mock_recording',
			audio_data = np.random.random(self.RECORDING_SAMPLE_RATE * self.RECORDING_DURATION_SEC) * 2.0 - 1.0,
			audio_sample_rate = self.RECORDING_SAMPLE_RATE,
			labels = pd.DataFrame(data = {
				'start': [0.0, self.RECORDING_DURATION_SEC / 2],
				'end': [self.RECORDING_DURATION_SEC / 2 - 1.0, self.RECORDING_DURATION_SEC - 1.0],
				'label': ['contact', 'feeding']
			}),
			brood_size = np.random.choice(self.BROOD_SIZES),
			brood_age = np.random.choice(self.BROOD_AGES)
		)

	def doCleanups(self) -> None:
		if Path(self.DATA_PATH).exists():
			for file in os.listdir(self.DATA_PATH):
				os.remove(f'{self.DATA_PATH}/{file}')
			os.rmdir(self.DATA_PATH)

	def test__brood_size_data_frame_shape(self):
		expected_n_rows = math.ceil(self.RECORDING_DURATION_SEC / self.CNN_SAMPLE_DURATION)
		expected_n_cols = len(self.BROOD_SIZES)

		bs_df, _ = prepare_training_data(
			self.recording, self.BROOD_SIZES, self.BROOD_AGES, self.DATA_PATH, self.CNN_SAMPLE_DURATION
		)

		self.assertIsInstance(bs_df, pd.DataFrame)
		self.assertEqual((expected_n_rows, expected_n_cols), bs_df.shape)

	def test__brood_age_data_frame_shape(self):
		expected_n_rows = math.ceil(self.RECORDING_DURATION_SEC / self.CNN_SAMPLE_DURATION)
		expected_n_cols = len(self.BROOD_AGES)

		_, ba_df = prepare_training_data(
			self.recording, self.BROOD_SIZES, self.BROOD_AGES, self.DATA_PATH, self.CNN_SAMPLE_DURATION
		)

		self.assertIsInstance(ba_df, pd.DataFrame)
		self.assertEqual((expected_n_rows, expected_n_cols), ba_df.shape)

	def test__brood_size_data_frame_column_names(self):
		expected_col_names = [str(bs) for bs in self.BROOD_SIZES]
		bs_df, _ = prepare_training_data(
			self.recording, self.BROOD_SIZES, self.BROOD_AGES, self.DATA_PATH, self.CNN_SAMPLE_DURATION
		)

		self.assertEqual(len(self.BROOD_SIZES), len(bs_df.columns))
		self.assertEqual(expected_col_names, list(bs_df.columns))
		self.assertEqual('file', bs_df.index.name)

	def test__brood_age_data_frame_column_names(self):
		expected_col_names = [str(ba) for ba in self.BROOD_AGES]
		_, ba_df = prepare_training_data(
			self.recording, self.BROOD_SIZES, self.BROOD_AGES, self.DATA_PATH, self.CNN_SAMPLE_DURATION
		)

		self.assertEqual(len(self.BROOD_AGES), len(ba_df.columns))
		self.assertEqual(expected_col_names, list(ba_df.columns))
		self.assertEqual('file', ba_df.index.name)

	def test__brood_size_data_frame_column_types(self):
		bs_df, _ = prepare_training_data(
			self.recording, self.BROOD_SIZES, self.BROOD_AGES, self.DATA_PATH, self.CNN_SAMPLE_DURATION
		)

		self.assertEqual('object', bs_df.index.dtype.name)

		for bs in self.BROOD_SIZES:
			col_vals = set(bs_df[str(bs)].unique())
			self.assertTrue(col_vals.issubset({ 0, 1 }))

	def test__brood_age_data_frame_column_types(self):
		_, ba_df = prepare_training_data(
			self.recording, self.BROOD_SIZES, self.BROOD_AGES, self.DATA_PATH, self.CNN_SAMPLE_DURATION
		)

		self.assertEqual('object', ba_df.index.dtype.name)

		for ba in self.BROOD_AGES:
			col_vals = set(ba_df[str(ba)].unique())
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
