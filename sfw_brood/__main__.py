import argparse
from datetime import datetime
from pathlib import Path

import soundfile as sf

from sfw_brood.cnn.model import CNNLoader
from sfw_brood.preprocessing import filter_recording, load_recording_data
from sfw_brood.cnn.trainer import cleanup

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('recording_path', type = str, help = 'Path to audio file or directory')
	arg_parser.add_argument('-m', '--model', type = str, help = 'Path to serialized model')
	arg_parser.add_argument('-o', '--output-dir', type = str, default = '_out')
	args = arg_parser.parse_args()

	audio_path = Path(args.recording_path)
	recordings = []

	if audio_path.is_dir():
		for file in audio_path.rglob('*.flac'):
			recordings.append(file)
	else:
		recordings.append(audio_path)

	work_dir = Path('_work')
	work_dir.mkdir(parents = True, exist_ok = True)

	sample_paths = []

	for rec_path in recordings:
		recording = load_recording_data(rec_path)
		audio_samples = filter_recording(recording, target_labels = ['feeding'])

		for i, (sample, _) in enumerate(audio_samples):
			sample_path = work_dir.joinpath(f'{rec_path.stem}_{i}.wav')
			sf.write(sample_path, sample, samplerate = recording.audio_sample_rate)
			sample_paths.append(str(sample_path))

	cnn_loader = CNNLoader()
	model = cnn_loader.load_model(args.model)

	pred_df = model.predict(sample_paths)
	classes = [col for col in pred_df.columns if col not in ['start_time', 'end_time', 'file']]
	pred_df['predicted_class'] = pred_df[classes].idxmax(axis = 1)

	pred_agg_df = pred_df[['file', 'predicted_class']] \
		.groupby('predicted_class') \
		.count() \
		.rename(columns = { 'file': 'score' }) \
		.sort_values(by = 'score', ascending = False) \
		.reset_index()

	print('\nPrediction result:')

	for i in range(len(pred_agg_df)):
		row = pred_agg_df.iloc[i]
		cls = row['predicted_class']
		percent = round(100 * row['score'] / len(pred_df), ndigits = 2)
		print(f'\t{cls}: {percent}%')

	print('\n')

	out_path = Path(args.output_dir)
	out_path.mkdir(parents = True, exist_ok = True)

	time_str = datetime.now().isoformat()[:19].replace(':', '-')
	result_file = f'result__{time_str}.csv'
	result_path = out_path.joinpath(result_file)
	pred_df.to_csv(result_path, index = False, columns = ['file', 'start_time', 'end_time', 'predicted_class'])

	cleanup(work_dir)
	print(f'Prediction results saved to file {result_path}')
