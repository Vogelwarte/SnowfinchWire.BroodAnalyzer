from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .common.preprocessing.io import SnowfinchNestRecording
from .common.preprocessing.io import number_from_recording_name


@dataclass
class TrainingDataset:
	files: list[Path]
	brood_sizes: list[int]
	brood_ages: list[float]


def prepare_training_data(
		recording: SnowfinchNestRecording, brood_sizes: list[int], brood_ages: list[float],
		work_dir: str, slice_duration_sec: float, overlap_sec: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
	slices_dir = Path(f'{work_dir}')
	slices_dir.mkdir(exist_ok = True, parents = True)

	slices = slice_audio(recording.audio_data, recording.audio_sample_rate, slice_duration_sec, overlap_sec)
	files = []

	for i, audio in enumerate(slices):
		file_path = f'{slices_dir}/{recording.title}__{i}.wav'
		files.append(file_path)
		sf.write(file_path, audio, samplerate = recording.audio_sample_rate)

	bs_data = __make_training_frame__(files, brood_sizes, recording.brood_size)
	ba_data = __make_training_frame__(files, brood_ages, recording.brood_age)

	return bs_data, ba_data


def slice_audio(audio: np.ndarray, sample_rate: int, slice_len_sec: float, overlap_sec = 0.0) -> list[np.ndarray]:
	samples_per_slice = round(slice_len_sec * sample_rate)
	overlap_samples = round(overlap_sec * sample_rate)

	start = 0
	end = samples_per_slice
	slices = []

	while start < len(audio):
		slices.append(audio[start:end])
		start += (samples_per_slice - overlap_samples)
		end = min(len(audio), start + samples_per_slice)

	return slices


def __make_training_frame__(files: list[str], classes: list, match) -> pd.DataFrame:
	data = { 'file': files }
	for c in classes:
		if c == match:
			data[str(c)] = list(np.ones(len(files), dtype = 'int'))
		else:
			data[str(c)] = list(np.zeros(len(files), dtype = 'int'))
	return pd.DataFrame(data = data).set_index('file')


def discover_training_data(data_dir: str) -> TrainingDataset:
	file_paths = []
	brood_sizes = set()
	brood_ages = set()

	for path in Path(data_dir).rglob('*.flac'):
		file_paths.append(path)
		rec_title = path.stem
		brood_age = number_from_recording_name(rec_title, label = 'BA', terminator = '_')
		brood_ages.add(brood_age)
		brood_size = number_from_recording_name(rec_title, label = 'BS', terminator = '-')
		brood_sizes.add(brood_size)

	return TrainingDataset(file_paths, list(brood_sizes), list(brood_ages))
