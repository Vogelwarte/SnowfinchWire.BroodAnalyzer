import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from sfw_brood.preprocessing import group_ages


def create_dataset_manifest(dataset_id: str, data: pd.DataFrame, data_dir: Path, out_dir: Path):
	with open(out_dir.joinpath(f'{dataset_id}_manifest.json'), mode = 'wt') as manifest:
		for _, row in tqdm(data.iterrows(), total = data.shape[0], desc = dataset_id):
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


def prepare_manifests(data: pd.DataFrame, data_dir: Path, data_config: dict, samples_per_class: int, out_dir: Path):
	if 'classes' in data_config.keys() and 'class' in data.columns:
		classes = [str(cls) for cls in data_config['classes']]
		data = data[data['class'].astype('str').isin(classes)]
	else:
		classes = set()
		for cls_col in [col for col in data.columns if 'class' in col]:
			classes.update(data[cls_col].unique())

	classes = sorted(classes)
	selector = data_config['selector']

	test_idx = data[selector].isin(data_config['test'])
	test_df = sample_data(data[test_idx], classes)

	val_idx = data[selector].isin(data_config['validation'])
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
	arg_parser.add_argument('-c', '--data-config-path', type = str)
	arg_parser.add_argument('-s', '--samples-per-class', type = int)
	arg_parser.add_argument('-t', '--target', type = str, choices = ['age', 'size'])
	args = arg_parser.parse_args()

	with open(args.data_config_path) as data_conf_file:
		data_config = json.load(data_conf_file)[args.target]

	data = pd.read_csv(args.in_path, dtype = {'class': str})
	if args.target == 'BA' and 'groups' in data_config.keys():
		data, _ = group_ages(data, groups = data_config['groups'])

	prepare_manifests(
		data = data,
		data_dir = Path(args.data_dir),
		data_config = data_config,
		samples_per_class = args.samples_per_class,
		out_dir = Path(args.out_path)
	)

if __name__ == '__main__':
	main()
