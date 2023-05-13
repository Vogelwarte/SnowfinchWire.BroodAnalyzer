import pandas as pd


def split_size_data(size_df, age_df, age_ranges):
	split_dfs = []
	for low, high in age_ranges:
		df = size_data_for_age_range(low, high, age_df, size_df)
		agg_df = agg_size_df(df)
		split_dfs.append(agg_df)
	return split_dfs


def agg_size_df(df):
	return df[['brood_id', 'file', 'class']] \
		.groupby(['brood_id', 'class']).count() \
		.rename(columns = { 'file': 'n_samples' }) \
		.sort_values(by = 'class') \
		.reset_index()


def size_data_for_age_range(low, high, age_df, size_df):
	files = age_df.loc[(age_df['class_min'] >= low) & (age_df['class_max'] < high), 'file']
	return size_df.loc[files].reset_index()


if __name__ == '__main__':
	age_data = pd.read_csv('/home/gardzielb/sfw-brood-work/out/s5.0-o1.0/brood-age.csv')
	size_data = pd.read_csv('/home/gardzielb/sfw-brood-work/out/s5.0-o1.0/brood-size.csv', index_col = 'file')
	age_ranges = [(0, 5), (5, 10), (10, 15), (15, 30)]
	size_data_split = split_size_data(size_data, age_data, age_ranges)

	for size_df, age_range in zip(size_data_split, age_ranges):
		print(f'Size data in age range {age_range}')
		print(size_df)
		print()
