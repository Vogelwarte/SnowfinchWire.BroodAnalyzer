import argparse
import json
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from sfw_brood.nemo.util import make_dataset_path
from sfw_brood.preprocessing import group_ages, group_sizes


def create_dataset_manifest(dataset_id: str, data: pd.DataFrame, data_dir: Path, out_dir: Path):
	with open(out_dir.joinpath(f'{dataset_id}_manifest.json'), mode = 'wt') as manifest:
		for _, row in tqdm(data.iterrows(), total = data.shape[0], desc = dataset_id):
			audio_path = data_dir.joinpath(row['file'])
			row_json = {
				'audio_filepath': audio_path.as_posix(),
				'duration': librosa.core.get_duration(path = audio_path),
				'command': row['class']
			}
			manifest.write(f'{json.dumps(row_json)}\n')


# def sample_data(data: pd.DataFrame, classes: List[str], samples_per_class: Optional[int] = None) -> pd.DataFrame:
# 	if samples_per_class is None:
# 		data_agg = data[['file', 'class']].groupby('class').count()
# 		samples_per_class = data_agg['file'].min()
#
# 	out_df = pd.DataFrame()
#
# 	for cls in classes:
# 		cls_df = data[data['class'] == cls]
# 		cls_df = cls_df.sample(min(samples_per_class, cls_df.shape[0]))
# 		out_df = pd.concat([out_df, cls_df])
#
# 	return out_df


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
	test_df = sample_dataset(data[test_idx], classes, int(samples_per_class * 0.45))

	val_idx = data[selector].isin(data_config['validation'])
	val_df = sample_dataset(data[val_idx], classes, int(samples_per_class * 0.2))

	train_df = sample_dataset(data[~(test_idx | val_idx)], classes, samples_per_class)
	# test_size = round(0.45 * len(train_df))
	# val_size = round(0.2 * len(train_df))

	# if test_size < len(test_df):
	# 	test_df = test_df.sample(n = test_size)

	# if val_size < len(val_df):
	# 	val_df = val_df.sample(n = val_size)

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
		data_config = json.load(data_conf_file)
		data_config_id = data_config['id']
		data_config = data_config[args.target]

	data = pd.read_csv(args.in_path, dtype = { 'class': str })
	if 'groups' in data_config.keys():
		if args.target == 'age':
			data, _ = group_ages(data, groups = data_config['groups'])
		else:
			data, _ = group_sizes(data, groups = data_config['groups'])

	prepare_manifests(
		data = data,
		data_dir = Path(args.data_dir),
		data_config = data_config,
		samples_per_class = args.samples_per_class,
		out_dir = make_dataset_path(args.out_path, data_config_id, args.target, args.samples_per_class)
	)


def __make_rec_name__(file_path: str) -> str:
	file_name = Path(file_path).stem
	rec_name_end_idx = file_name.rindex('__')
	return file_name[:rec_name_end_idx]


# df has 'rec_path' column and unique 'file' column, 'n' column is created
def __find_n_max__(df: pd.DataFrame, sample_count: int, sample_steps: List[int]) -> Tuple[List[int], pd.DataFrame]:
	df['n'] = np.NaN
	rec_df = df[['rec_name', 'file']] \
		.groupby('rec_name') \
		.count() \
		.reset_index() \
		.rename(columns = { 'file': 'n_samples' })

	i = 0
	total_count = 0

	while total_count + sample_steps[i] * np.count_nonzero(rec_df['n_samples'] >= sample_steps[i]) < sample_count:
		if i == len(sample_steps) - 1:
			print(f'Total count = {total_count} and only {np.count_nonzero(rec_df["n_samples"] >= sample_steps[i])} {sample_steps[i]}-sample recordings')
			raise Exception('You ask for too much, babe')

		n = sample_steps[i]
		selection = (rec_df['n_samples'] >= n) & (rec_df['n_samples'] < sample_steps[i + 1])
		df.loc[df['rec_name'].isin(rec_df.loc[selection, 'rec_name']), 'n'] = n
		total_count += np.count_nonzero(selection) * n
		i += 1

	df['n'].fillna(sample_steps[i], inplace = True)
	df['n'] = df['n'].astype(int)

	return sample_steps[:(i + 1)], df


# requires 'file' and 'class' columns in data
def sample_dataset(data: pd.DataFrame, classes: list, n_samples_per_class: int) -> pd.DataFrame:
	sample_steps = [3, 6, 10, 16, 24, 32, 48, 64, 96, 128, 180, 240, 300, 360, 420, 512]
	data['rec_name'] = data['file'].apply(__make_rec_name__)

	out_df = pd.DataFrame()

	for cls in classes:
		print(f'Sampling for class {cls}')

		cls_sample_steps, cls_df = __find_n_max__(data[data['class'] == cls], n_samples_per_class, sample_steps)
		out_cls_df = pd.DataFrame()

		for n in cls_sample_steps:
			sample_df = cls_df[cls_df['n'] == n].groupby('rec_name').sample(n)
			out_cls_df = pd.concat([out_cls_df, sample_df])

		out_df = pd.concat([out_df, out_cls_df])

	# cleanup
	data.drop(columns = 'rec_name', inplace = True)

	return out_df


if __name__ == '__main__':
	main()
