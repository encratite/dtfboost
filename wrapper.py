from abc import ABC, abstractmethod
from typing import Any, Final

from catboost import CatBoostRegressor, Pool
from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd

class RegressionWrapper(ABC):
	def set_validation(self, x_validation: Any, y_validation: Any):
		pass

	def permit_transform(self):
		return True

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

class AutoGluonWrapper(RegressionWrapper):
	LABEL_COLUMN: Final[str] = "label"

	_model: Any

	def __init__(self) -> None:
		self._model = TabularPredictor(label=self.LABEL_COLUMN, problem_type="regression")

	def permit_transform(self):
		return False

	def fit(self, x: Any, y: Any) -> None:
		features_df = pd.DataFrame(x)
		labels_df = pd.Series(y, name=self.LABEL_COLUMN)
		training_data_df = pd.concat([features_df, labels_df], axis=1)
		training_data = TabularDataset(training_data_df)
		self._model.fit(training_data)

	def predict(self, x: Any) -> Any:
		features_df = pd.DataFrame(x)
		return self._model.predict(features_df)