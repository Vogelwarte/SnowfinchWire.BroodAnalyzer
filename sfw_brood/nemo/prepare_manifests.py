import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm


def create_dataset_manifest(dataset_id: str, data: pd.DataFrame, data_dir: Path, out_dir: Path):
	with open(out_dir.joinpath(f'{dataset_id}_manifest.json'), mode = 'wt') as manifest:
		for _, row in tqdm(data.iterrows(), desc = dataset_id):
			row_json = {
				'audio_filepath': data_dir.joinpath(row['file']).as_posix(),
				'duration': 0.0,
				'command': row['class']
			}
			manifest.write(f'{json.dumps(row_json)}\n')


def sample_data(data: pd.DataFrame, classes: List[str], samples_per_class: Optional[int] = None) -> pd.DataFrame:
	if samples_per_class is None:
		data_agg = data[['file', 'class']].groupby('class').count()
		samples_per_class = data_agg['file'].min()

	out_df = pd.DataFrame()

	for cls in classes:
		cls_df = data[data['class'] == cls]
		cls_df = cls_df.sample(min(samples_per_class, cls_df.shape[0]))
		out_df = pd.concat([out_df, cls_df])

	return out_df


def prepare_manifests(data: pd.DataFrame, data_dir: Path, split_conf: dict, samples_per_class: int, out_dir: Path):
	if 'classes' in split_conf.keys() and 'class' in data.columns:
		classes = [str(cls) for cls in split_conf['classes']]
		data = data[data['class'].astype('str').isin(classes)]
	else:
		classes = set()
		for cls_col in [col for col in data.columns if 'class' in col]:
			classes.update(data[cls_col].unique())

	classes = sorted(classes)
	selector = split_conf['selector']

	test_idx = data[selector].isin(split_conf['test'])
	test_df = sample_data(data[test_idx], classes)

	val_idx = data[selector].isin(split_conf['validation'])
	val_df = sample_data(data[val_idx], classes)

	train_df = sample_data(data[~(test_idx | val_idx)], classes, samples_per_class)
	test_size = round(0.45 * len(train_df))
	val_size = round(0.2 * len(train_df))

	if test_size < len(test_df):
		test_df = test_df.sample(n = test_size)

	if val_size < len(val_df):
		val_df = val_df.sample(n = val_size)

	out_dir.mkdir(parents = True, exist_ok = True)
	create_dataset_manifest('train', train_df, data_dir, out_dir)
	create_dataset_manifest('test', test_df, data_dir, out_dir)
	create_dataset_manifest('validation', val_df, data_dir, out_dir)


def main():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('-i', '--in-path', type = str)
	arg_parser.add_argument('-o', '--out-path', type = str)
	arg_parser.add_argument('-d', '--data-dir', type = str)
	arg_parser.add_argument('-c', '--split-conf-path', type = str)
	arg_parser.add_argument('-s', '--samples-per-class', type = int)
	arg_parser.add_argument('-t', '--target', type = str, choices = ['BA', 'BS'])
	args = arg_parser.parse_args()

	with open(args.split_conf_path) as split_conf_file:
		split_conf = json.load(split_conf_file)

	prepare_manifests(
		data = pd.read_csv(args.in_path, dtype = {'class': str}),
		data_dir = Path(args.data_dir),
		split_conf = split_conf[args.target],
		samples_per_class = args.samples_per_class,
		out_dir = Path(args.out_path)
	)

if __name__ == '__main__':
	main()
