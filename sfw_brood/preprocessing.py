from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import soundfile as sf

from .common.preprocessing.io import SnowfinchNestRecording, load_recording_data, validate_recording_data
from .common.preprocessing.io import number_from_recording_name


@dataclass
class SnowfinchDataset:
	files: list[Path]
	brood_sizes: list[int]
	brood_ages: list[int]


def __detect_silence__(audio: np.ndarray, min_len, threshold: int) -> list:
	silence_segments = []
	start = 0
	while start < len(audio):
		while start < len(audio) and audio[start] >= threshold:
			start += 1

		i = 0
		while start + i < len(audio) and audio[start + i] < threshold:
			i += 1
		if i >= min_len:
			silence_segments.append((start, start + i))

		start += i

	return silence_segments


def __is_silent__(audio: np.ndarray, min_silence_len: int, threshold: int) -> bool:
	start = 0

	while start < len(audio):
		while start < len(audio) and audio[start] >= threshold:
			start += 1

		i = 0
		while start + i < len(audio) and i < min_silence_len and audio[start + i] < threshold:
			i += 1

		if i >= min_silence_len:
			return True

		start += i

	return False


def __to_dbfs__(audio: np.ndarray) -> np.ndarray:
	audio_abs = np.abs(audio)
	return 20 * np.log10(np.where(audio_abs > 1e-8, audio_abs, 1e-8))


def prepare_training_data(
		recording: SnowfinchNestRecording, brood_sizes: list[int], brood_ages: list[float],
		work_dir: Union[str, Path], slice_duration_sec: float, overlap_sec: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
	slices_dir = Path(work_dir)
	slices_dir.mkdir(exist_ok = True, parents = True)

	intereseting_audio = filter_recording(recording, target_labels = ['feeding', 'contact'])
	audio_slices = []
	slice_labels = []

	for audio, label in intereseting_audio:
		slices = slice_audio(audio, recording.audio_sample_rate, slice_duration_sec, overlap_sec)
		audio_slices.extend(slices)
		slice_labels.extend([label] * len(slices))

	silence_threshold = -45
	files = []
	is_silence = []

	for i, audio in enumerate(audio_slices):
		audio_dbfs = __to_dbfs__(audio)
		is_silent = __is_silent__(
			audio_dbfs, min_silence_len = round(0.7 * recording.audio_sample_rate), threshold = silence_threshold
		)
		is_silence.append(is_silent)

		file_name = f'{recording.title}__{i}.wav'
		files.append(file_name)
		file_path = Path(slices_dir).joinpath(file_name)
		sf.write(file_path, audio, samplerate = recording.audio_sample_rate)

	bs_data = __make_training_frame__(
		files, slice_labels, is_silence, classes = brood_sizes, match = recording.brood_size
	)
	ba_data = __make_training_frame__(
		files, slice_labels, is_silence, classes = brood_ages, match = recording.brood_age
	)

	return bs_data, ba_data


def prepare_training(
		dataset: SnowfinchDataset, work_dir: Union[str, Path], slice_duration_sec: float, overlap_sec: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
	bs_train_df = pd.DataFrame()
	ba_train_df = pd.DataFrame()

	for file in dataset.files:
		print(f'Loading recording {file.stem}')
		recording = load_recording_data(file)
		validate_recording_data(recording)

		bs_df, ba_df = prepare_training_data(
			recording, dataset.brood_sizes, dataset.brood_ages, work_dir,
			slice_duration_sec, overlap_sec
		)

		print(f'Extracted {len(bs_df)} samples from recording', end = '\n\n')

		bs_train_df = pd.concat([bs_train_df, bs_df])
		ba_train_df = pd.concat([ba_train_df, ba_df])

	return bs_train_df, ba_train_df


def balance_data(data: pd.DataFrame, classes: list[str], tolerance = 0.2) -> pd.DataFrame:
	data_per_class = []

	for cls in classes:
		data_per_class.append(data[data[cls] == 1])

	cls_counts = [len(df) for df in data_per_class]
	sample_size = round(min(cls_counts) * (1.0 + tolerance))

	balanced_df = pd.DataFrame()

	for i in range(len(data_per_class)):
		if len(data_per_class[i]) > sample_size:
			data_per_class[i] = data_per_class[i].sample(n = sample_size)
		balanced_df = pd.concat([balanced_df, data_per_class[i]])

	return balanced_df


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


def __make_training_frame__(
		files: list[str], labels: list[str], is_silence: list[bool], classes: list, match: Union[int, float, str],
) -> pd.DataFrame:
	data = {
		'file': files,
		'event': labels,
		'is_silence': is_silence,
		'class': [match] * len(files)
	}

	for c in classes:
		if c == match:
			data[str(c)] = list(np.ones(len(files), dtype = 'int'))
		else:
			data[str(c)] = list(np.zeros(len(files), dtype = 'int'))

	return pd.DataFrame(data = data).set_index('file')


def discover_training_data(data_dir: str) -> SnowfinchDataset:
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

	return SnowfinchDataset(file_paths, list(brood_sizes), list(brood_ages))


def filter_recording(recording: SnowfinchNestRecording, target_labels: list[str]) -> list[tuple[np.ndarray, str]]:
	labelled_audio = []
	matching_events = recording.labels[recording.labels.label.isin(target_labels)]

	for i in range(len(matching_events)):
		audio_event = matching_events.iloc[i]
		event_start = round(audio_event.start * recording.audio_sample_rate)
		event_end = round(audio_event.end * recording.audio_sample_rate)
		event_audio = recording.audio_data[event_start:event_end]
		labelled_audio.append((event_audio, audio_event.label))

	return labelled_audio
