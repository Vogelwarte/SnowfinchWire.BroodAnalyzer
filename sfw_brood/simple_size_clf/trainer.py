from pathlib import Path

from sfw_brood.model import ModelTrainer
from sfw_brood.simple_size_clf.ensemble_clf import EnsemleClassifier
from sfw_brood.simple_size_clf.model import SimpleBroodSizeClassifier
from sfw_brood.simple_size_clf.validator import SimpleClfValidator
from sfw_brood.simple_size_clf.preprocessing import prepare_feeding_data, generate_size_labels, split_dataset


class SimpleClfTrainer(ModelTrainer):
	def __init__(self, data_path: str, data_config: dict, n_models: int, voting: str):
		self.data_path = Path(data_path)
		self.data_config = data_config['size']
		self.n_models = n_models
		self.voting = voting
		self.x_features = [
			'duration', 'feeding_count', 'age_min', 'age_max',
			'min_sin', 'min_cos', 'agg_duration', 'agg_feeding_count'
		]

		feeding_data = prepare_feeding_data(
			feeding_stats_path = self.data_path.joinpath('feeding-stats.csv'),
			brood_data_path = self.data_path.joinpath('snowfinch-broods.csv')
		)

		self.label_columns = generate_size_labels(
			feeding_data, class_configs = [
				[2, 3, 4, 5],
				[(2, 3), (4, 5)],
				[(2, 3), 4, 5],
				[2, (3, 4), 5],
				[2, 3, (4, 5)]
			]
		)

		self.train_data, self.test_data = split_dataset(feeding_data, test_broods = self.data_config['test'])

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		pass

	def train_model_for_size(self, out_dir: str):
		classifiers = []

		for label_col in self.label_columns:
			clf = EnsemleClassifier(n_models = self.n_models, svm = True, bayes = True, name = label_col)
			print(f'Training simple size CLF for label group {label_col}')
			clf.fit(self.train_data, self.x_features, y_col = label_col)
			classifiers.append(clf)

		trained_model = SimpleBroodSizeClassifier(
			classifiers,
			model_info = {
				'target': 'size',
				'n_models': self.n_models,
				'voting': self.voting,
				'classifiers': ['SVM', 'Bayessian'],
				'dataset': self.data_path.absolute().as_posix(),
				'data_config': self.data_config,
				'labels': self.label_columns,
				'features': self.x_features
			}
		)

		trained_model.serialize(Path(out_dir).joinpath('model').as_posix())

		model_validator = SimpleClfValidator(self.test_data, self.label_columns)
		model_validator.validate(trained_model, output = out_dir)

		return trained_model

	def train_model_for_age(self, out_dir: str):
		raise AssertionError('Simple size classifier does not support age classification')
