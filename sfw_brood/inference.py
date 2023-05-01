from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union, List

import pandas as pd
import soundfile as sf

from sfw_brood.cnn.util import cleanup
from sfw_brood.common.preprocessing.io import load_recording_data
from sfw_brood.model import SnowfinchBroodClassifier
from sfw_brood.preprocessing import filter_recording


@dataclass
class SnowfinchBroodPrediction:
	raw: pd.DataFrame
	agg: pd.DataFrame

	def save(self, out: Union[Path, str]):
		out = Path(out)
		out.mkdir(parents = True, exist_ok = True)
		self.raw.to_csv(out.joinpath('raw.csv'), index = False)
		self.agg.to_csv(out.joinpath('agg.csv'), index = False)


class Inference:
	def __init__(self, model: SnowfinchBroodClassifier, work_dir: Path):
		self.model = model
		self.work_dir = work_dir

	def __enter__(self):
		self.work_dir.mkdir(parents = True, exist_ok = True)
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		cleanup(self.work_dir)

	def predict(self, paths: List[Path], n_workers: int) -> SnowfinchBroodPrediction:
		sample_paths = self.__prepare_data__(paths)
		print('Running predictions for samples:', sample_paths)
		pred_df = self.model.predict(sample_paths, n_workers = n_workers)
		pred_df.to_csv('_inference-pred.csv')
		pred_df, agg_df = self.__format_predictions__(pred_df)
		return SnowfinchBroodPrediction(pred_df, agg_df)

	def __prepare_data__(self, audio_paths: List[Path]) -> list[str]:
		sample_paths = []

		for audio_path in audio_paths:
			recordings = []
			if audio_path.is_dir():
				for fmt in ['wav', 'flac', 'WAV']:
					for file in audio_path.rglob(f'*.{fmt}'):
						recordings.append(file)
			else:
				recordings.append(audio_path)

			for rec_path in recordings:
				recording = load_recording_data(rec_path, include_brood_info = False)
				audio_samples = filter_recording(recording, target_labels = ['feeding'])

				rec_path_rel = rec_path.relative_to(rec_path.root) if rec_path.is_absolute() else rec_path
				work_dir = self.work_dir.joinpath(rec_path_rel.parent).joinpath(rec_path.stem)
				work_dir.mkdir(exist_ok = True, parents = True)

				for i, (sample, _) in enumerate(audio_samples):
					sample_path = work_dir.joinpath(f'{i}.wav')
					sf.write(sample_path, sample, samplerate = recording.audio_sample_rate)
					sample_paths.append(sample_path.as_posix())

		return sample_paths

	def __extract_rec_path__(self, sample_path: str) -> str:
		sample_path = Path(sample_path)
		return sample_path.parent.relative_to(self.work_dir).as_posix()

	def __extract_brood_id__(self, rec_path: str) -> str:
		return Path(rec_path).parent.parent.stem

	def __extract_datetime__(self, rec_path: str) -> datetime:
		rec_name = Path(rec_path).stem
		return datetime(
			int(rec_name[:4]), int(rec_name[4:6]), int(rec_name[6:8]),
			int(rec_name[9:11]), int(rec_name[11:13]), int(rec_name[13:15])
		)

	def __format_predictions__(self, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
		classes = [col for col in pred_df.columns if col not in ['start_time', 'end_time', 'file']]
		pred_df['predicted_class'] = pred_df[classes].idxmax(axis = 1)

		pred_df['duration'] = pred_df['end_time'] - pred_df['start_time']
		pred_df['rec_path'] = pred_df['file'].apply(self.__extract_rec_path__)

		agg_map = {
			'file': 'count',
			'duration': 'sum'
		}

		for cls in classes:
			agg_map[cls] = 'sum'

		agg_cols = ['rec_path'] + list(agg_map.keys())
		agg_df = pred_df[agg_cols].groupby('rec_path').agg(agg_map)
		agg_df = agg_df.reset_index().rename(columns = { 'file': 'n_samples' })
		agg_df['brood_id'] = agg_df['rec_path'].apply(self.__extract_brood_id__)
		agg_df['datetime'] = agg_df['rec_path'].apply(self.__extract_datetime__)
		for cls in classes:
			agg_df[f'{cls}_score'] = agg_df[cls] / agg_df['n_samples']
			agg_df.rename(inplace = True, columns = {cls: f'{cls}_n_samples'})

		pred_df = pred_df[['file', 'start_time', 'end_time', 'predicted_class']]

		return pred_df, agg_df


class InferenceValidator:
	def validate(self, inference: Inference, test_data: pd.DataFrame, output = '', multi_target = False) -> dict:
		audio_paths = [Path(path) for path in test_data['rec_path']]
		pred = inference.predict(audio_paths, n_workers = 10)

