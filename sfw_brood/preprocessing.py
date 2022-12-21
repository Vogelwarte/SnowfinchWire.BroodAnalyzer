from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .common.preprocessing.io import SnowfinchNestRecording


def prepare_training_data(
		recording: SnowfinchNestRecording, data_dir: str, slice_duration_sec: float, overlap_sec: float = 0.0
) -> pd.DataFrame:
	samples_per_slice = round(slice_duration_sec * recording.audio_sample_rate)
	overlap_samples = round(overlap_sec * recording.audio_sample_rate)

	slices_dir = Path(f'{data_dir}/{recording.title}')
	slices_dir.mkdir(exist_ok = True, parents = True)

	start = 0
	end = samples_per_slice
	files = []

	while start < len(recording.audio_data):
		file_no = len(files)
		print(f'{file_no}. {start / recording.audio_sample_rate} - {end / recording.audio_sample_rate}')

		file_name = f'{slices_dir}/{file_no}.wav'
		sf.write(file_name, recording.audio_data[start:end], samplerate = recording.audio_sample_rate)
		files.append(file_name)

		start += (samples_per_slice - overlap_samples)
		end = min(len(recording.audio_data), start + samples_per_slice)

	out_data = { 'file': files }
	for bs in range(1, recording.brood_size + 1):
		if bs == recording.brood_size:
			out_data[str(bs)] = list(np.ones(len(files), dtype = 'int'))
		else:
			out_data[str(bs)] = list(np.zeros(len(files), dtype = 'int'))

	return pd.DataFrame(data = out_data).set_index('file')
