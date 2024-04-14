import subprocess
import sys
from datetime import datetime
from pathlib import Path


class UAT:
	def __init__(self, uat_id: str, data_path: Path, out_path: Path, config_path: Path):
		self.id = uat_id
		self.data_path = data_path
		self.out_path = out_path
		self.config_path = config_path

	def run_test(self, model_path: Path):
		print(f'========================= {self.id} =========================')
		print()

		start_time = datetime.now()
		self._test_()
		result_dir = self.__find_result_directory__(self.out_path, start_time)
		self.__verify_result_dir_structure__(model_path, result_dir)

		print()
		print(f'{self.id} PASSED!')
		print()
		print()

	def _test_(self):
		proc_args = [sys.executable, 'classify.py', '-c', self.config_path.as_posix()]
		process = subprocess.run(proc_args, text = True, stderr = subprocess.DEVNULL)
		if process.returncode != 0:
			self.__fail__('Classifier error')

	def __verify_result_dir_structure__(self, model_path: Path, result_path: Path):
		models = [path.stem for path in model_path.glob('*.zip')]

		for model in models:
			model_result_dir = result_path.joinpath(model)
			self.__assert_true__(
				model_result_dir.is_dir(),
				f'Result directory for model {model} is present'
			)
			self.__assert_true__(
				model_result_dir.joinpath('sample-preds.csv').is_file(),
				f'Sample predictions file for model {model} is present'
			)
			self.__assert_true__(
				model_result_dir.joinpath('rec-preds.csv').is_file(),
				f'Recording predictions file for model {model} is present'
			)
			self.__assert_true__(
				model_result_dir.joinpath('brood-period-preds.csv').is_file(),
				f'Brood-period predictions file for model {model} is present'
			)
			self.__assert_true__(
				model_result_dir.joinpath('1_6_1_22.png').is_file(),
				f'Brood 1_6_1_22 age graph for model {model} is present'
			)
			self.__assert_true__(
				model_result_dir.joinpath('8_1_1_23.png').is_file(),
				f'Brood 8_1_1_23 age graph for model {model} is present'
			)

	def __find_result_directory__(self, out_path: Path, start_time: datetime) -> Path:
		for subdir in out_path.glob('result__*'):
			if not subdir.is_dir():
				continue

			try:
				result_time = datetime.strptime(subdir.name[8:], '%Y-%m-%dT%H-%M-%S')
				if result_time >= start_time:
					return subdir
			except ValueError:
				pass

		self.__fail__('Result directory not found')

	def __assert_true__(self, predicate: bool, msg: str):
		if predicate:
			print(f'{msg}: OK')
		else:
			self.__fail__(msg)

	def __fail__(self, msg: str):
		raise AssertionError(f'{self.id} FAILED: {msg}')


def main():
	data_path = Path('uat').joinpath('data')
	model_path = Path('uat').joinpath('models')
	out_path = Path('uat').joinpath('out')
	cfg_path = Path('uat').joinpath('config')

	uat1 = UAT('[UAT1] Given labels', data_path, out_path, cfg_path.joinpath('given-labels.yml'))
	uat1.run_test(model_path)

	uat2 = UAT('[UAT2] Embedded BCA', data_path, out_path, cfg_path.joinpath('with-bca.yml'))
	uat2.run_test(model_path)


if __name__ == '__main__':
	main()
