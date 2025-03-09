from __future__ import annotations
from typing import cast, TypeVar, Generic
from sortedcontainers import SortedDict
import pandas as pd
from ohlc import OHLC

T = TypeVar("T")

class TimeSeries(Generic[T]):
	_data: SortedDict[pd.Timestamp, T]

	def __init__(self, data: SortedDict[pd.Timestamp, T]):
		self._data = data

	@staticmethod
	def read_csv(path: str) -> TimeSeries[float]:
		df = pd.read_csv(path, parse_dates=[0])
		data = SortedDict()
		for row in df.itertuples(index=False):
			time = cast(pd.Timestamp, row[0])
			value = cast(float, row[1])
			data[time] = value
		series = TimeSeries(data)
		return series

	@staticmethod
	def read_ohlc_csv(path: str) -> TimeSeries[OHLC]:
		df = pd.read_csv(path, parse_dates=[0])
		data = SortedDict()
		for row in df.itertuples(index=False):
			ohlc = OHLC(row)
			data[ohlc.time] = ohlc
		series = TimeSeries(data)
		return series

	def get(self, time: pd.Timestamp, count: int = 1) -> list[Generic[T]] | Generic[T]:
		value = self._data.get(time)
		if count == 1 and value is not None:
			return value
		index = self._data.bisect_right(time)
		if index == 0:
			raise Exception("No record for that date")
		values: list[Generic[T]] = []
		keys = self._data.keys()
		for i in range(count):
			key_index = index - i - 1
			if key_index < 0:
				raise Exception("Not enough data available")
			key = keys[key_index] # type: ignore
			value = self._data[key]
			values.append(value)
		if count == 1:
			return values[0]
		else:
			return values