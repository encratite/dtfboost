from __future__ import annotations

from typing import cast, TypeVar, Generic, Iterator

import pandas as pd
from sortedcontainers import SortedDict

from ohlc import OHLC

T = TypeVar("T")

class TimeSeries(Generic[T]):
	_data: SortedDict[pd.Timestamp, T]

	def __init__(self, data: SortedDict[pd.Timestamp, T]):
		self._data = data

	@staticmethod
	def read_csv(path: str) -> TimeSeries[float]:
		df = pd.read_csv(path, parse_dates=[0], date_format="%Y-%m-%d")
		# Some FRED data files like T10Y2Y lack numeric entries, skip them
		# This causes the rate of change and difference calculations to use the preceding value instead
		df.dropna(how="any")
		data = SortedDict()
		for row in df.itertuples(index=False):
			time = cast(pd.Timestamp, row[0])
			value = cast(float, row[1])
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

	def get(self, time: pd.Timestamp, count: int | None = None, offsets: list[int] | None = None) -> list[Generic[T]] | Generic[T]:
		assert count is None or offsets is None
		single_mode = count is None and offsets is None
		if single_mode:
			value = self._data.get(time)
			if value is not None:
				return value
		index = self._data.bisect_right(time)
		if index == 0:
			raise Exception("No record for that date")
		values: list[Generic[T]] = []
		keys = self._data.keys()
		if offsets is None:
			if single_mode:
				offsets = [1]
			else:
				offsets = range(count)
		for offset in offsets:
			key_index = index - offset - 1
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