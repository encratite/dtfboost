from abc import ABC, abstractmethod
from typing import Any, Final

from catboost import CatBoostRegressor, Pool
from autogluon.tabular import TabularPredictor, TabularDataset
import pandas as pd

class RegressionWrapper(ABC):
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

	def __init__(self, **kwargs) -> None:
		self._model = CatBoostRegressor(**kwargs)

	def fit(self, x: Any, y: Any) -> None:
		training_pool = Pool(data=x, label=y)
		self._model.fit(training_pool, verbose=0)

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