from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt


def plot_sample_count(
		df: pd.DataFrame, x_label_col: str, x_rotation: str, x_label: str, title: str, out: Optional[Path] = None
):
	all_colors = list(plt.get_cmap('Set3').colors)
	color_map = { }
	class_colors = []

	for i, cls in enumerate(df['class'].unique()):
		color_map[cls] = all_colors[i]

	for cls in df['class']:
		class_colors.append(color_map[cls])

	fig = plt.figure()

	plt.bar(df.index.values, df['n_samples'], color = class_colors)
	plt.xticks(range(0, len(df)), df[x_label_col], rotation = x_rotation)
	plt.xlabel(x_label)
	plt.title(title)

	handles = [plt.Rectangle((0, 0), 1, 1, color = color_map[label]) for label in df['class'].unique()]
	plt.legend(handles, df['class'].unique())

	fig.tight_layout()

	if out:
		plt.savefig(out)
	else:
		plt.show()


def split_size_data(size_df, age_df, age_ranges):
	split_dfs = []
	for low, high in age_ranges:
		df = size_data_for_age_range(low, high, age_df, size_df)
		agg_df = agg_size_df(df)
		split_dfs.append(agg_df)
	return split_dfs


def join_val_broods(brood_id: str) -> str:
	return brood_id[:-4] if '_val' in brood_id else brood_id


def agg_size_df(df):
	return df[['brood_id', 'file', 'class']] \
		.groupby(['brood_id', 'class']).count() \
		.rename(columns = { 'file': 'n_samples' }) \
		.sort_values(by = 'class') \
		.reset_index()


def size_data_for_age_range(low, high, age_df, size_df):
	files = age_df.loc[(age_df['class_min'] >= low) & (age_df['class_max'] < high), 'file']
	return size_df.loc[files].reset_index()


def analyse_data_in_age_ranges(path, age_ranges):
	path = Path(path)
	age_data = pd.read_csv(path.joinpath('brood-age.csv'))
	size_data = pd.read_csv(path.joinpath('brood-size.csv'), index_col = 'file')
	size_data_split = split_size_data(size_data, age_data, age_ranges)

	for size_df, age_range in zip(size_data_split, age_ranges):
		print(f'Size data in age range {age_range}')
		print(size_df)
		print()
