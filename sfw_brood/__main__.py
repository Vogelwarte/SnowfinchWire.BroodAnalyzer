import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf

from sfw_brood.cnn.model import CNNLoader
from sfw_brood.preprocessing import filter_recording, load_recording_data
from sfw_brood.cnn.trainer import cleanup


def __prepare_data__(audio_path: Path, work_dir: Path) -> list[str]:
	recordings = []

	if audio_path.is_dir():
		for fmt in ['wav', 'flac']:
			for file in audio_path.rglob(f'*.{fmt}'):
				recordings.append(file)
	else:
		recordings.append(audio_path)

	work_dir.mkdir(parents = True, exist_ok = True)
	sample_paths = []

	for rec_path in recordings:
		recording = load_recording_data(rec_path)
		audio_samples = filter_recording(recording, target_labels = ['feeding'])

		for i, (sample, _) in enumerate(audio_samples):
			sample_path = work_dir.joinpath(f'{rec_path.stem}.{i}.wav')
			sf.write(sample_path, sample, samplerate = recording.audio_sample_rate)
			sample_paths.append(sample_path.as_posix())

	return sample_paths


def __format_predictions__(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	classes = [col for col in pred_df.columns if col not in ['start_time', 'end_time', 'file']]
	pred_df['predicted_class'] = pred_df[classes].idxmax(axis = 1)

	pred_df['duration'] = pred_df['end_time'] - pred_df['start_time']
	pred_df['rec'] = pred_df['file'].apply(lambda path: Path(path).stem.split('.')[0])

	agg_map = {
		'file': 'count',
		'duration': 'sum'
	}

	for cls in classes:
		agg_map[cls] = 'sum'

	agg_cols = ['rec'] + list(agg_map.keys())
	agg_df = pred_df[agg_cols].groupby('rec').agg(agg_map).reset_index().rename(columns = {'file': 'n_samples'})
	for cls in classes:
		agg_df[cls] /= agg_df['n_samples']

	pred_df = pred_df[['file', 'start_time', 'end_time', 'predicted_class']]

	return pred_df, agg_df


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('recording_path', type = str, help = 'Path to audio file or directory')
	arg_parser.add_argument('-m', '--model', type = str, help = 'Path to serialized model')
	arg_parser.add_argument('-o', '--output-dir', type = str, default = '_out')
	arg_parser.add_argument('-w', '--n-workers', type = int, default = 10)
	args = arg_parser.parse_args()

	work_dir = Path('_work')
	sample_paths = __prepare_data__(audio_path = Path(args.recording_path), work_dir = work_dir)

	cnn_loader = CNNLoader()
	model = cnn_loader.load_model(args.model)

	pred_df = model.predict(sample_paths, n_workers = args.n_workers)
	pred_df, agg_df = __format_predictions__(pred_df)

	print('\nPrediction result:')
	print(agg_df)

	time_str = datetime.now().isoformat()[:19].replace(':', '-')
	out_path = Path(args.output_dir).joinpath(f'result__{time_str}')
	out_path.mkdir(parents = True, exist_ok = True)

	pred_df.to_csv(out_path.joinpath('pred.csv'), index = False)
	agg_df.to_csv(out_path.joinpath('agg.csv'), index = False)

	cleanup(work_dir)
	print(f'Prediction results saved to directory {out_path}')


if __name__ == '__main__':
	main()
