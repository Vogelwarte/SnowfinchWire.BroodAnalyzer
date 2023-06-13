from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Set

import numpy as np
import pandas as pd
import soundfile as sf
import csv
from typing import Callable
from pathlib import Path


@dataclass
class InputRecord:
	start: float
	end: float
	label: str


@dataclass
class SnowfinchNestRecording:
	title: str
	audio_data: np.ndarray
	audio_sample_rate: int
	labels: pd.DataFrame
	brood_age: Tuple[float, float]
	brood_size: int

	@property
	def audio_len_sec(self):
		return len(self.audio_data) * self.audio_sample_rate


def read_audacity_labels(data_path: Union[str, Path]) -> List[InputRecord]:
	result = []
	try:
		with open(data_path) as f:
			cf = csv.DictReader(f, fieldnames = ['start', 'end', 'label'], delimiter = '\t')
			for row in cf:
				result.append(InputRecord(float(row['start']), float(row['end']), row['label']))
	except Exception as error:
		print(f'Error loading labels from file {Path(data_path).as_posix()}')

	return result


def load_recording_data(
		path: Path, label_reader: Callable[[Union[str, Path]], List[InputRecord]] = read_audacity_labels,
		rec_info: Optional[pd.Series] = None, include_brood_info = False
) -> SnowfinchNestRecording:
	rec_title = path.stem
	full_rec_title = rec_title

	if include_brood_info:
		if rec_info is None:
			age_min = age_max = number_from_recording_name(rec_title, label = 'BA', terminator = '_')
			brood_size = number_from_recording_name(rec_title, label = 'BS', terminator = '-')
		else:
			brood_id = rec_info['brood_id']
			brood_size = rec_info['brood_size']
			age_min = rec_info['age_min']
			age_max = rec_info['age_max']
			full_rec_title = f'{brood_id}_{rec_title}'
	else:
		brood_size = age_min = age_max = -1

	try:
		audio_data, sample_rate = sf.read(path)

		labels_file = next(Path(path.parent).glob(f'*{rec_title}*.txt'))
		labels_list = label_reader(labels_file)
		labels = pd.DataFrame(labels_list).convert_dtypes()

		return SnowfinchNestRecording(
			full_rec_title, audio_data, sample_rate, labels,
			brood_size = brood_size, brood_age = (age_min, age_max)
		)

	except sf.LibsndfileError:
		raise FileNotFoundError('Audio file not found')
	except StopIteration:
		raise FileNotFoundError('Labels file not found')


def number_from_recording_name(recording_title: str, label: str, terminator: chr) -> int:
	try:
		start_idx = recording_title.index(label) + len(label)
		end_idx = recording_title.index(terminator, start_idx)
		return int(recording_title[start_idx:end_idx])
	except ValueError:
		raise ValueError(f'Invalid recording title format: failed to read {label} parameter')


def validate_recording_data(data: SnowfinchNestRecording, expected_labels: Optional[Set[str]] = None):
	if not len(data.labels):
		return

	if not pd.Index(data.labels.start).is_monotonic_increasing:
		raise ValueError('Label start timestamps are not in ascending order')

	if not pd.Index(data.labels.end).is_monotonic_increasing:
		raise ValueError('Label end timestamps are not in ascending order')

	if any(data.labels.start >= data.labels.end):
		raise ValueError('Start timestamp of some labels is after their end timestamp')

	audio_length_sec = data.audio_data.shape[0] / data.audio_sample_rate
	if data.labels.start.iloc[0] < 0.0 or data.labels.start.iloc[-1] > audio_length_sec:
		raise ValueError('Labels timestamps do not fit in the audio')

	if expected_labels:
		for i in range(data.labels.shape[0]):
			if data.labels.label.iloc[i] not in expected_labels:
				raise ValueError(f'Unexpected label at position {i + 1}')
