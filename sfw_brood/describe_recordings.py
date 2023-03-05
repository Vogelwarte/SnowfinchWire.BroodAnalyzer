import argparse
import re

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


def collect_rec_data(data_path: Path, brood_df: pd.DataFrame):
	rec_data = {
		'rec_path': [],
		'brood_id': [],
		'datetime': [],
		'age_min': [],
		'age_max': [],
		'brood_size': []
	}

	for rec_path in data_path.rglob('*.WAV'):
		print(str(rec_path))
		brood_id = rec_path.parent.parent.stem

		rec_title = rec_path.stem
		rec_title_match = re.match('20[0-9]{6}_[0-9]{6}', rec_title)
		if not rec_title_match or rec_title_match.start() != 0 or rec_title_match.end() != len(rec_title):
			continue

		rec_datetime = datetime(
			int(rec_title[:4]), int(rec_title[4:6]), int(rec_title[6:8]),
			int(rec_title[9:11]), int(rec_title[11:13]), int(rec_title[13:])
		)

		visit_datetimes = brood_df.loc[brood_df['brood_id'] == brood_id, 'datetime']
		if len(visit_datetimes) == 0:
			continue

		time_delta = abs(pd.to_datetime(visit_datetimes, format = '%Y-%m-%d %H:%M:%S') - rec_datetime)
		rec_idx = time_delta.idxmin(axis = 0)
		if time_delta[rec_idx] > timedelta(hours = 12):
			continue

		brood_info = brood_df.iloc[rec_idx]

		rec_data['rec_path'].append(str(rec_path.relative_to(data_path)))
		rec_data['brood_id'].append(brood_id)
		rec_data['datetime'].append(rec_datetime)
		rec_data['age_min'].append(brood_info['age_min'])
		rec_data['age_max'].append(brood_info['age_max'])
		rec_data['brood_size'].append(round(brood_info['brood_size']))

	return pd.DataFrame(data = rec_data)


if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('--brood-data', type = str)
	arg_parser.add_argument('-o', '--out', type = str)
	arg_parser.add_argument('data_path', type = str)
	args = arg_parser.parse_args()

	rec_df = collect_rec_data(Path(args.data_path), brood_df = pd.read_csv(args.brood_data))
	rec_df.to_csv(args.out, index = False)
