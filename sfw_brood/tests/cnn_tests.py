from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
import soundfile
from opensoundscape.torch.models.cnn import CNN

from sfw_brood.cnn.model import SnowfinchBroodCNN
from sfw_brood.cnn.util import cleanup


class SnowfinchBroodCNNTests(TestCase):
	def setUp(self) -> None:
		self.classes = [str(i) for i in range(2, 5)]
		self.sample_duration = 2.0
		self.cnn = SnowfinchBroodCNN(trained_cnn = CNN(
			architecture = 'resnet18', classes = self.classes,
			single_target = True, sample_duration = self.sample_duration
		))

		self.work_dir = '_data/cnn_tests'
		Path(self.work_dir).mkdir(parents = True, exist_ok = True)

		self.sample_rate = 48000
		self.rec_duration_sec = 4.0
		self.rec_paths = []

		for i in range(2):
			file_path = f'{self.work_dir}/{i}.wav'
			audio = np.random.random(round(self.rec_duration_sec * self.sample_rate)) * 2.0 - 1.0
			soundfile.write(file_path, audio, self.sample_rate)
			self.rec_paths.append(file_path)

	def doCleanups(self) -> None:
		pred_result = self.cnn.predict(self.rec_paths)
		cleanup(Path(self.work_dir))

	def test__predict_return_type(self):
		pred_result = self.cnn.predict(self.rec_paths)
		self.assertIsInstance(pred_result, pd.DataFrame)

	def test__predict_return_shape(self):
		expected_height = len(self.rec_paths) * (self.rec_duration_sec / self.sample_duration)
		pred_result = self.cnn.predict(self.rec_paths)
		self.assertEqual(expected_height, pred_result.shape[0])
		self.assertEqual(len(self.classes) + 3, pred_result.shape[1])

	def test__predict_out_frame_column_names(self):
		expected_col_names = ['file', 'start_time', 'end_time'] + self.classes
		pred_result = self.cnn.predict(self.rec_paths)
		self.assertEqual(len(expected_col_names), len(pred_result.columns))
		self.assertEqual(expected_col_names, list(pred_result.columns))

	def test__predict_out_frame_column_types(self):
		pred_result = self.cnn.predict(self.rec_paths)
		self.assertEqual('object', pred_result.file.dtype.name)
		for col in self.classes + ['start_time', 'end_time']:
			self.assertTrue('float' in pred_result[col].dtype.name)

	def test__predict_softmax_output(self):
		pred_result = self.cnn.predict(self.rec_paths)
		class_sums = pred_result[self.classes].sum(axis = 1)
		self.assertTrue(all(np.abs(class_sums - 1.0) < 1e-6))

	def test__predict_out_frame_files(self):
		pred_result = self.cnn.predict(self.rec_paths)
		for file in list(pred_result.file):
			self.assertTrue(file in self.rec_paths)

	def test__predict_out_frame_audio_duration(self):
		pred_result = self.cnn.predict(self.rec_paths)
		for i in range(len(pred_result)):
			record = pred_result.iloc[i]
			duration = record['end_time'] - record['start_time']
			self.assertEqual(self.sample_duration, duration)


# TODO: mock CNN!
class CNNValidatorTests(TestCase):
	def test__accuracy_score(self):
		# TODO: implement
		pass

	def test__handle_missing_predictions(self):
		# TODO: implement
		pass
