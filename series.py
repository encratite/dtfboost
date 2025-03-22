from __future__ import annotations

from typing import cast, TypeVar, Generic, Iterator
from math import isnan

import pandas as pd
from sortedcontainers import SortedDict

from ohlc import OHLC

T = TypeVar("T")

class TimeSeries(Generic[T]):
	start: pd.Timestamp
	_data: SortedDict[pd.Timestamp, T]

	def __init__(self, data: SortedDict[pd.Timestamp, T]):
		self._data = data
		self.start = data.keys()[0] # type: ignore

	@staticmethod
	def read_csv(path: str, is_daily: bool) -> TimeSeries[float]:
		df = pd.read_csv(path, parse_dates=[0, 2], date_format="%Y-%m-%d")
		data = SortedDict()
		time_index = 0 if is_daily else 2
		for row in df.itertuples(index=False):
			time = cast(pd.Timestamp, row[time_index])
			value = cast(float, row[1])
			if isinstance(value, (int, float)) and not isnan(value):
				data[time] = value
		series = TimeSeries(data)
		return series

	@staticmethod
	def read_ohlc_csv(path: str) -> TimeSeries[OHLC]:
		df = pd.read_csv(path)
		data = SortedDict()
		for row in df.itertuples(index=False):
			ohlc = OHLC(row)
			data[ohlc.time] = ohlc
		series = TimeSeries(data)
		return series

	def get(self, time: pd.Timestamp, count: int | None = None, offsets: list[int] | None = None, right: bool = False) -> list[Generic[T]] | Generic[T]:
		assert count is None or offsets is None
		single_mode = count is None and offsets is None
		if single_mode:
			value = self._data.get(time)
			if value is not None:
				return value
		index = self._data.bisect_right(time) if right else self._data.bisect_left(time)
		if index == 0:
			raise Exception("No record for that date")
		if index == len(self._data):
			index -= 1
		values: list[Generic[T]] = []
		keys = self._data.keys()
		if offsets is None:
			if single_mode:
				offsets = [0]
			else:
				offsets = range(count)
		for offset in offsets:
			key_index = index - offset
			if key_index < 0:
				raise Exception("Not enough data available")
			key = keys[key_index] # type: ignore
			value = self._data[key]
			values.append(value)
		if single_mode:
			return values[0]
		else:
			return values

	def values(self) -> list[Generic[T]]:
		return list(self._data.values())

	def __iter__(self) -> Iterator[pd.Timestamp]:
		return self._data.keys().__iter__()