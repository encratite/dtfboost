from typing import Any, Callable, Final
from collections import defaultdict

import pandas as pd
import numpy as np

from enums import RebalanceFrequency

class InterleavedModel:
	WEEK_DAYS: Final[int] = 5

	_model: Any | None
	_models: dict[int, Any] | None
	_rebalance_frequency: RebalanceFrequency
	_is_interleaved: bool

	def __init__(
		self,
		model_factory: Callable[[], Any],
		rebalance_frequency: RebalanceFrequency
	):
		self._model = None
		self._models = None
		match rebalance_frequency:
			case RebalanceFrequency.DAILY | RebalanceFrequency.WEEKLY | RebalanceFrequency.MONTHLY:
				self._model = model_factory()
				self._is_interleaved = False
			case RebalanceFrequency.DAILY_INTERLEAVED:
				# One model per day of the week
				self._models = {}
				for x in range(self.WEEK_DAYS):
					self._models[x] = model_factory()
				self._is_interleaved = True
			case _:
				raise Exception(f"Unknown rebalance frequency: {rebalance_frequency}")
		self._rebalance_frequency = rebalance_frequency

	def fit(self, x_training: list[list[float]], y_training: list[float], training_times: list[pd.Timestamp]) -> None:
		assert len(x_training) == len(y_training) == len(training_times)
		if self._is_interleaved:
			training_data: defaultdict[int, list[tuple[list[float], float]]] = defaultdict(list)
			for i in range(len(x_training)):
				x = x_training[i]
				y = y_training[i]
				time = training_times[i]
				interleaved_id = self._get_interleaved_id(time)
				training_data[interleaved_id].append((x, y))
			for interleaved_id, tuples in training_data.items():
				model = self._models[interleaved_id]
				interleaved_x_training = [x for x, _y in tuples]
				interleaved_y_training = [y for _x, y in tuples]
				model.fit(interleaved_x_training, interleaved_y_training)
		else:
			self._model.fit(x_training, y_training)

	def predict(self, x: list[list[float]], times: list[pd.Timestamp]) -> Any:
		assert len(x) == len(times)
		if self._is_interleaved:
			prediction: list[float] = []
			for i in range(len(x)):
				features = x[i]
				time = times[i]
				interleaved_id = self._get_interleaved_id(time)
				model = self._models[interleaved_id]
				y = model.predict([features])
				prediction.append(y[0])
			return np.array(prediction)
		else:
			return self._model.predict(x)

	def _get_interleaved_id(self, time: pd.Timestamp) -> int:
		match self._rebalance_frequency:
			case RebalanceFrequency.DAILY_INTERLEAVED:
				assert time.dayofweek < self.WEEK_DAYS
				return int(time.dayofweek)
			case _:
				raise Exception("Unsupported rebalance ID")