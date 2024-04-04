from datetime import datetime, timedelta, time
from typing import Optional, Tuple

import pandas as pd


def __timedelta_from_hours__(hours: int) -> timedelta:
	return timedelta(days = hours // 24, hours = hours % 24)


def __timedelta_to_hours__(td: timedelta) -> int:
	return td.days * 24 + td.seconds // 3600


# rec_df columns: brood_id, datetime
def assign_recording_periods(
		rec_df: pd.DataFrame, period_hours: int, period_map: Optional[dict] = None, overlap_hours = 0
) -> Tuple[pd.DataFrame, dict]:
	def calculate_period_start(rec_time, min_date):
		period_offset = __timedelta_to_hours__(rec_time - min_date) // period_hours
		return min_date + __timedelta_from_hours__(period_hours * period_offset)

	period_df = pd.DataFrame()
	period_map_out = { }

	for brood in rec_df['brood_id'].unique():
		brood_df = rec_df[rec_df['brood_id'] == brood]

		if period_map and brood in period_map.keys():
			period_starts = period_map[brood]
		else:
			min_datetime = brood_df['datetime'].min()
			base_hour = (min_datetime.hour // period_hours) * period_hours if period_hours <= 12 else 0
			base_period_start = datetime.combine(min_datetime.date(), time(base_hour))
			period_starts = [base_period_start]
			if overlap_hours > 0:
				period_starts.append(base_period_start + __timedelta_from_hours__(period_hours - overlap_hours))

		period_map_out[brood] = period_starts

		for period_start in period_starts:
			brood_period_df = brood_df[brood_df['datetime'] >= period_start]
			brood_period_df['period_start'] = brood_period_df['datetime'].apply(
				lambda dt: calculate_period_start(dt, period_start)
			)
			period_df = pd.concat([period_df, brood_period_df])

	return period_df, period_map_out


def reject_underrepresented_samples(pred_df: pd.DataFrame, threshold = 50) -> pd.DataFrame:
	return pred_df[pred_df['n_samples'] >= threshold].reset_index().drop(columns = 'index')
