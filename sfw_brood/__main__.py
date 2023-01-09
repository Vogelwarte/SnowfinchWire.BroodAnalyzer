import argparse
from datetime import datetime
from pathlib import Path

from sfw_brood.cnn.model import CNNLoader

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
			recordings.append(str(file))
	else:
		recordings.append(str(audio_path))

	cnn_loader = CNNLoader()
	model = cnn_loader.load_model(args.model)

	pred_df = model.predict(recordings)
	classes = [col for col in pred_df.columns if col not in ['start_time', 'end_time', 'file']]
	pred_df['predicted_class'] = pred_df[classes].idxmax(axis = 1)

	out_path = Path(args.output_dir)
	out_path.mkdir(parents = True, exist_ok = True)

	time_str = datetime.now().isoformat()[:19].replace(':', '-')
	result_file = f'result__{time_str}.csv'
	result_path = out_path.joinpath(result_file)
	pred_df.to_csv(result_path, index = False, columns = ['file', 'start_time', 'end_time', 'predicted_class'])

	print(f'Prediction results saved to file {result_path}')
