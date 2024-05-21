"""
Columns:
* target
* architecture
* lr
* bs
* data_config
* sample_duration
* samples_per_class
* max_epochs
* seed
* duration
* n_epochs
* accuracy
* inference_accuracy
* classes
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def collect_results(root_path: Path) -> pd.DataFrame:
	records = []

	for seed_dir in root_path.rglob('seed-*'):
		seed = int(seed_dir.name.split('-')[1])

		session_title = seed_dir.parent.parent.name
		session_start_time = datetime.strptime(session_title[4:], '%Y-%m-%dT%H-%M-%S')

		result_dir = list(seed_dir.glob('*'))[0]
		train_finish_time = datetime.strptime(result_dir.name[4:], '%Y-%m-%dT%H-%M-%S')

		record = {
			'experiment_id': seed_dir.parent.name,
			'seed': seed,
			'session_start_time': session_start_time,
			'train_finish_time': train_finish_time
		}

		setup_path = seed_dir.parent.joinpath('experiment.json')
		if setup_path.is_file():
			with open(seed_dir.parent.joinpath('experiment.json')) as setup_file:
				setup = json.load(setup_file)
				record.update({ key: setup[key] for key in setup if key not in ['durations'] })

		with open(result_dir.joinpath('meta.json')) as meta_file:
			meta_data = json.load(meta_file)
			record['n_epochs'] = meta_data['train_epochs']

		with open(result_dir.joinpath('test-result.json')) as result_file:
			result = json.load(result_file)
			record['accuracy'] = result['result']['accuracy']
			record['classes'] = ','.join(result['classes'])

		for inference_dir in result_dir.glob('inference*'):
			with open(inference_dir.joinpath('test-result.json')) as inference_result_file:
				inference_result = json.load(inference_result_file)
				record[f'{inference_dir.name}_accuracy'] = inference_result['result']['accuracy']

		records.append(record)

	return pd.DataFrame.from_records(records)
