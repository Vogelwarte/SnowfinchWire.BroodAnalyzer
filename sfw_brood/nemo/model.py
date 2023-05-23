import multiprocessing
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from nemo.collections.asr.models import EncDecClassificationModel
from torch import Tensor
from tqdm import tqdm

from sfw_brood.model import SnowfinchBroodClassifier, ModelType, ModelLoader, classes_from_data_config


def __load_samples__(sample_data: Tuple[str, pd.DataFrame, float]) -> Tuple[np.ndarray, pd.DataFrame]:
	audio_path, sample_df, sample_duration = sample_data
	audio, sample_rate = sf.read(audio_path)
	sample_len = int(sample_rate * sample_duration)
	audio_samples = []
	out_df = pd.DataFrame()

	for _, row in sample_df.iterrows():
		start_time = row['start_time']
		end_time = row['end_time']
		start_idx = int(start_time * sample_rate)
		end_idx = int(end_time * sample_rate)
		n_samples = min(round((end_idx - start_idx) / sample_len), (len(audio) - start_idx) // sample_len)
		end_idx = start_idx + n_samples * sample_len
		audio_samples.append(audio[start_idx:end_idx].reshape(-1, sample_len))
		slice_df = pd.DataFrame(data = {
			'start_time': np.linspace(start_time, start_time + (n_samples - 1) * sample_duration, n_samples),
			'end_time': np.linspace(start_time + sample_duration, start_time + n_samples * sample_duration, n_samples),
			'file': audio_path
		})
		out_df = pd.concat([out_df, slice_df])

	return np.concatenate(audio_samples), out_df


class SnowfinchBroodMatchboxNet(SnowfinchBroodClassifier):
	def __init__(self, trained_net: EncDecClassificationModel, model_info: dict):
		super().__init__(ModelType.MATCHBOX, model_info)
		self.network = trained_net
		self.batch_size = model_info['batch_size'] if 'batch_size' in model_info else 128
		self.sample_duration = model_info['sample_duration'] if 'sample_duration' in model_info else None
		self.classes = classes_from_data_config(self.model_info['data_config'])

	def predict(self, recordings: Union[List[str], pd.DataFrame], n_workers: int = 12) -> pd.DataFrame:
		if type(recordings) == pd.DataFrame:
			print('MatchboxNet: Running predictions')
			return self.__predict_for_data_frame__(recordings, n_workers)
		else:
			raise TypeError('Invalid prediction input data format')

	def __predict_for_data_frame__(self, recordings: pd.DataFrame, n_workers: int) -> pd.DataFrame:
		rec_df = recordings.reset_index()
		autio_paths = rec_df['file'].unique()
		out_df = pd.DataFrame()
		predictions = []

		batches = [[]]
		for audio_path in autio_paths:
			track_df = rec_df[rec_df['file'] == audio_path]
			if len(batches[-1]) + len(track_df) > self.batch_size:
				batches.append([])
			batches[-1].append((audio_path, track_df, self.sample_duration))

		for i, batch in enumerate(batches):
			print(f'Batch {i + 1} / {len(batches)}')
			audio_samples = []
			with multiprocessing.Pool(n_workers) as proc_pool:
				for samples, sample_df in tqdm(proc_pool.imap_unordered(__load_samples__, batch), total = len(batch)):
					audio_samples.append(samples)
					out_df = pd.concat([out_df, sample_df])

			audio_samples = np.concatenate(audio_samples)
			pred_labels = self.predict_for_audio_samples(
				samples = Tensor(audio_samples).to('cuda'),
				sample_lengths = Tensor([audio_samples.shape[1]] * audio_samples.shape[0]).to('cuda'),
				n_workers = n_workers
			)
			predictions.extend([self.classes[int(cls_idx)] for cls_idx in pred_labels])

		out_df['predicted_class'] = predictions
		out_df['predicted_class'] = out_df['predicted_class'].astype('category').cat.set_categories(self.classes)
		cls_1hot = pd.get_dummies(out_df['predicted_class'])
		out_df = pd.concat([out_df.drop(columns = 'predicted_class'), cls_1hot.astype('int')], axis = 1)

		return out_df

	def predict_for_audio_samples(self, samples: Tensor, sample_lengths: Tensor, n_workers: int = 12) -> Tensor:
		predictions = []
		for i in range(0, len(samples), self.batch_size):
			to_idx = min(i + self.batch_size, len(samples))
			input_signal = samples[i:to_idx]
			signal_length = sample_lengths[i:to_idx]
			logits = self.network(input_signal = input_signal, input_signal_length = signal_length)
			_, pred = logits.topk(1, dim = 1, largest = True, sorted = True)
			pred = pred.squeeze()
			if len(pred) > 0:
				predictions.extend(pred)
		return Tensor(predictions)

	def _serialize_(self, path: str):
		self.network.save_to(path)


class MatchboxNetLoader(ModelLoader):
	def __init__(self):
		super().__init__(ModelType.MATCHBOX)

	def _deserialize_model_(self, path: str, meta_data: dict) -> SnowfinchBroodClassifier:
		network = EncDecClassificationModel.restore_from(path)
		return SnowfinchBroodMatchboxNet(network, meta_data)
