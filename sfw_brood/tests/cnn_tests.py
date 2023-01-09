from pathlib import Path
from unittest import TestCase, mock

import numpy as np
import pandas as pd
import soundfile
from opensoundscape.torch.models.cnn import CNN

from sfw_brood.cnn.model import SnowfinchBroodCNN
from sfw_brood.cnn.util import cleanup
from sfw_brood.cnn.validator import CNNValidator


class SnowfinchBroodCNNTests(TestCase):
	def setUp(self) -> None:
		self.classes = [str(i) for i in range(2, 5)]
		self.sample_duration = 2.0
		self.cnn = SnowfinchBroodCNN(
			trained_cnn = CNN(
				architecture = 'resnet18', classes = self.classes,
				single_target = True, sample_duration = self.sample_duration
			),
			arch = 'resnet18',
			n_epochs = 0
		)

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


class CNNValidatorTests(TestCase):
	def setUp(self) -> None:
		rec_count = 10
		self.classes = ['2', '3', '4']
		class_dict = { }

		for cls in self.classes:
			class_dict[cls] = np.zeros(rec_count)

		for i in range(rec_count):
			cls = np.random.choice(self.classes)
			class_dict[cls][i] = 1

		self.test_data = pd.DataFrame(
			index = [f'{i}.wav' for i in range(rec_count)],
			data = class_dict
		)
		self.test_data.index.name = 'file'

	def test__accuracy_score(self):
		true_classes = self.test_data[self.classes].idxmax(axis = 1)
		missed_rows = round(0.3 * len(self.test_data))

		pred_data = { 'file': list(self.test_data.index) }
		for cls in self.classes:
			pred_data[cls] = list(self.test_data[cls])

		for i in range(missed_rows):
			pred_data[true_classes[i]][i] = 0
			other_classes = self.classes.copy()
			other_classes.remove(true_classes[i])
			pred_data[np.random.choice(other_classes)][i] = 1

		with mock.patch('sfw_brood.cnn.model.SnowfinchBroodCNN') as cnn_mock:
			model_mock = cnn_mock.return_value
			model_mock.predict.return_value = pd.DataFrame(data = pred_data)

		cnn_validator = CNNValidator(self.test_data)
		accuracy = cnn_validator.validate(model_mock)

		self.assertEqual(1 - missed_rows / len(self.test_data), accuracy)

	def test__handle_missing_predictions(self):
		data_indices = [i for i in range(len(self.test_data))]
		missing_indices = np.random.choice(data_indices, size = 5)
		pred_df = self.test_data.reset_index().drop(missing_indices)

		with mock.patch('sfw_brood.cnn.model.SnowfinchBroodCNN') as cnn_mock:
			model_mock = cnn_mock.return_value
			model_mock.predict.return_value = pred_df

		cnn_validator = CNNValidator(self.test_data)
		accuracy = cnn_validator.validate(model_mock)
		self.assertEqual(1.0, accuracy)
