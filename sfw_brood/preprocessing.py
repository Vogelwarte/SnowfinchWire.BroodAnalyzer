import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

from sklearn.preprocessing import OneHotEncoder
from opensoundscape.data_selection import resample
from tqdm import tqdm

from .common.preprocessing.io import SnowfinchNestRecording, load_recording_data, validate_recording_data
from .common.preprocessing.io import number_from_recording_name


@dataclass
class SnowfinchDataset:
	data_root: Path
	files: list[Path]
	brood_sizes: list[int]


@dataclass
class PreprocessorConfig:
	work_dir: Union[Path, str]
	slice_duration_sec: float
	overlap_sec: float


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
		recording: SnowfinchNestRecording, brood_sizes: list[int], config: PreprocessorConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
	slices_dir = Path(config.work_dir)
	slices_dir.mkdir(exist_ok = True, parents = True)

	intereseting_audio = filter_recording(recording, target_labels = ['feeding', 'contact'])
	audio_slices = []
	slice_labels = []

	for audio, label in intereseting_audio:
		slices = slice_audio(audio, recording.audio_sample_rate, config.slice_duration_sec, config.overlap_sec)
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
		files, slice_labels, is_silence, match = recording.brood_age
	)

	return bs_data, ba_data


def __process_recording__(
		rec_data: tuple[Path, Optional[pd.Series], list[int], PreprocessorConfig], verbose = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
	rec_path, rec_info, brood_sizes, config = rec_data
	rec_title = rec_path.stem

	if verbose:
		print(f'Loading recording {rec_title}')

	try:
		recording = load_recording_data(rec_path, rec_info = rec_info)
		validate_recording_data(recording)
		bs_df, ba_df = prepare_training_data(recording, brood_sizes, config)

		bs_df['recording'] = rec_title
		bs_df['brood_id'] = rec_info['brood_id']

		ba_df['recording'] = rec_title
		ba_df['brood_id'] = rec_info['brood_id']

		if verbose:
			print(f'Extracted {len(bs_df)} samples from recording', end = '\n\n')

	except Exception as error:
		print(f'Failed to process recording {rec_path}: {error}')
		bs_df, ba_df = pd.DataFrame(), pd.DataFrame()

	return bs_df, ba_df


def prepare_training(
		dataset: SnowfinchDataset, work_dir: Union[str, Path], slice_duration_sec: float,
		overlap_sec: float = 0.0, rec_df: Optional[pd.DataFrame] = None, n_workers = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
	bs_train_df = pd.DataFrame()
	ba_train_df = pd.DataFrame()

	def find_rec_data(rec_path: Path):
		if rec_df is None:
			return None
		return rec_df[rec_df['rec_path'] == rec_path.relative_to(dataset.data_root).as_posix()].iloc[0]

	preproc_config = PreprocessorConfig(work_dir, slice_duration_sec, overlap_sec)
	rec_data = [(file, find_rec_data(file), dataset.brood_sizes, preproc_config) for file in dataset.files]

	with multiprocessing.Pool(n_workers) as proc_pool:
		for size_df, age_df in tqdm(proc_pool.imap_unordered(__process_recording__, rec_data), total = len(rec_data)):
			bs_train_df = pd.concat([bs_train_df, size_df])
			ba_train_df = pd.concat([ba_train_df, age_df])

	bs_train_df = bs_train_df[~bs_train_df.index.duplicated(keep = 'first')]
	ba_train_df = ba_train_df[~ba_train_df.index.duplicated(keep = 'first')]

	return bs_train_df, ba_train_df


def balance_data(data: pd.DataFrame, classes: list[str], samples_per_class: str) -> pd.DataFrame:
	class_samples = [np.count_nonzero(data[cls]) for cls in classes]

	if samples_per_class == 'min':
		return resample(data, n_samples_per_class = np.min(class_samples))
	elif samples_per_class == 'max':
		return resample(data, n_samples_per_class = np.max(class_samples))
	elif samples_per_class == 'mean':
		return resample(data, n_samples_per_class = round(np.mean(class_samples)))
	else:
		return resample(data, n_samples_per_class = int(samples_per_class))


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


def __map_class_to_group__(cls: float, groups: list[tuple[float, float]], group_labels: list[str]) -> str:
	for (low, high), label in zip(groups, group_labels):
		if low <= cls <= high:
			return label
	return 'none'


# this function modifies input data frame
def __groups_to_1hot__(groups_df: pd.DataFrame) -> pd.DataFrame:
	groups_df = groups_df \
		.sort_values(by = 'class') \
		.reset_index() \
		.drop(columns = 'index')

	groups_encoder = OneHotEncoder()
	groups_1hot = groups_encoder.fit_transform(groups_df['class'].values.reshape(-1, 1))
	groups_1hot_df = pd.DataFrame(
		data = groups_1hot.toarray(),
		columns = groups_encoder.categories_
	)

	group_1hot_columns = [col[0] for col in groups_1hot_df.columns]
	groups_df[group_1hot_columns] = groups_1hot_df[group_1hot_columns]

	return groups_df


def group_sizes(size_df: pd.DataFrame, groups: list[tuple[float, float]]) -> tuple[pd.DataFrame, list[str]]:
	group_labels = ['{:02}-{:02}'.format(low, high) for low, high in groups]

	def map_size(size: float) -> str:
		return __map_class_to_group__(size, groups, group_labels)

	size_group_df = size_df.rename(columns = { 'class': 'size' })
	size_group_df['class'] = size_group_df['size'].apply(map_size)

	return __groups_to_1hot__(size_group_df), group_labels


def label_age_groups(groups: list[tuple[float, float]]) -> list[str]:
	return ['{:04.1f}-{:04.1f}'.format(low, high) for low, high in groups]


def group_ages(
		age_df: pd.DataFrame, groups: List[Tuple[float, float]], multi_target = False
) -> Tuple[pd.DataFrame, List[str]]:
	group_labels = label_age_groups(groups)
	age_group_df = age_df.rename(columns = {
		'class_min': 'age_min',
		'class_max': 'age_max'
	})

	if multi_target:
		for (low, high), label in zip(groups, group_labels):
			is_cls_min = (age_group_df['age_min'] >= low) & (age_group_df['age_min'] <= high)
			is_cls_max = (age_group_df['age_max'] >= low) & (age_group_df['age_max'] <= high)
			is_cls_between = (age_group_df['age_min'] < low) & (age_group_df['age_max'] > high)
			is_cls = is_cls_min | is_cls_max | is_cls_between
			age_group_df[label] = is_cls.astype('float')

		age_group_df = age_group_df \
			.reset_index() \
			.drop(columns = 'index')
	else:
		def map_age(age: float) -> str:
			return __map_class_to_group__(age, groups, group_labels)

		age_group_df['class_min'] = age_group_df['age_min'].apply(map_age)
		age_group_df['class_max'] = age_group_df['age_max'].apply(map_age)
		age_group_df = __groups_to_1hot__(
			groups_df = age_group_df[age_group_df['class_min'] == age_group_df['class_max']] \
				.drop(columns = ['class_min']) \
				.rename(columns = { 'class_max': 'class' })
		)

	return age_group_df, group_labels


def __make_training_frame__(
		files: list[str], labels: list[str], is_silence: list[bool],
		match: Union[int, float, str, tuple[float, float]], classes: Optional[list] = None
) -> pd.DataFrame:
	data = {
		'file': files,
		'event': labels,
		'is_silence': is_silence
	}

	if type(match) == tuple:
		cls_min, cls_max = match
		data['class_min'] = [cls_min] * len(files)
		data['class_max'] = [cls_max] * len(files)
	else:
		data['class'] = [match] * len(files)

	if classes is not None:
		for c in classes:
			if c == match:
				data[str(c)] = list(np.ones(len(files), dtype = 'int'))
			else:
				data[str(c)] = list(np.zeros(len(files), dtype = 'int'))

	return pd.DataFrame(data = data).set_index('file')


def discover_training_data(data_dir: str, rec_df: Optional[pd.DataFrame] = None) -> SnowfinchDataset:
	file_paths = set()
	brood_sizes = set()
	data_path = Path(data_dir)

	rec_patterns = ['*.flac', '*.wav', '*.WAV']
	for pattern in rec_patterns:
		for path in data_path.rglob(pattern):
			if rec_df is None:
				file_paths.add(path)
				rec_title = path.stem
				brood_size = number_from_recording_name(rec_title, label = 'BS', terminator = '-')
				brood_sizes.add(brood_size)
			elif path.relative_to(data_path).as_posix() in rec_df['rec_path'].values:
				file_paths.add(path)

	if rec_df is not None:
		brood_sizes.update(rec_df['brood_size'].unique())

	return SnowfinchDataset(data_path, list(file_paths), list(brood_sizes))


def filter_recording(recording: SnowfinchNestRecording, target_labels: list[str]) -> list[tuple[np.ndarray, str]]:
	if not len(recording.labels):
		return []

	labelled_audio = []
	matching_events = recording.labels[recording.labels.label.isin(target_labels)]

	for i in range(len(matching_events)):
		audio_event = matching_events.iloc[i]
		event_start = round(audio_event.start * recording.audio_sample_rate)
		event_end = round(audio_event.end * recording.audio_sample_rate)
		event_audio = recording.audio_data[event_start:event_end]
		labelled_audio.append((event_audio, audio_event.label))

	return labelled_audio
