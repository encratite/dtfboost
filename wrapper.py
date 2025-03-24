from abc import ABC, abstractmethod
from typing import Any
import platform
from pprint import pprint

from catboost import CatBoostRegressor, Pool

if platform.system() == "Linux":
	import autosklearn.regression
	AUTOML_SUPPORTED = True
else:
	AUTOML_SUPPORTED = False

class RegressionWrapper(ABC):
	def set_validation(self, x_validation: Any, y_validation: Any):
		pass

	@abstractmethod
	def fit(self, x: Any, y: Any) -> None:
		pass

	@abstractmethod
	def predict(self, x) -> Any:
		pass

class CatBoostWrapper(RegressionWrapper):
	_model: CatBoostRegressor
	_x_validation: Any | None
	_y_validation: Any | None

	def __init__(self, **kwargs) -> None:
		self._model = CatBoostRegressor(**kwargs)
		self._x_validation = None
		self._y_validation = None

	def set_validation(self, x_validation: Any, y_validation: Any):
		self._x_validation = x_validation
		self._y_validation = y_validation

	def fit(self, x: Any, y: Any) -> None:
		training_pool = Pool(data=x, label=y)
		validation_pool = Pool(data=self._x_validation, label=self._y_validation)
		self._model.fit(training_pool, eval_set=validation_pool, verbose=0)

	def predict(self, x: Any) -> Any:
		return self._model.predict(x)

class AutoMLWrapper(RegressionWrapper):
	_model: Any

	def __init__(self) -> None:
		self._model = autosklearn.regression.AutoSklearnRegressor(
			time_left_for_this_task=120,
			per_run_time_limit=30,
			tmp_folder="autosklearn",
		)

	def fit(self, x: Any, y: Any) -> None:
		self._model.fit(x, y, dataset_name="dtfboost")
		print(self._model.leaderboard())
		pprint(self._model.show_models(), indent=4)

	def predict(self, x: Any) -> Any:
		return self._model.predict(x)