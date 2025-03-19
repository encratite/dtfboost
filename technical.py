from collections import defaultdict
import os
from statistics import stdev
from math import tanh
from typing import Final

import pandas as pd

from config import Configuration
from data import TrainingData
from enums import FeatureCategory
from feature import Feature
from ohlc import OHLC

MOMENTUM_DAYS: Final[list[int]] = [
	2,
	3,
	5,
	10,
	25,
	50,
	100,
	150,
	200,
	250
]

LAGGED_MOMENTUM_DAYS: Final[list[tuple[int, int]]] = [
	(25, 50),
	(100, 250)
]

MOVING_AVERAGE_DAYS: Final[list[int]] = [
	4,
	5,
	10,
	25,
	50,
	100,
	150,
	200,
	250
]

VOLATILITY_DAYS: Final[list[int]] = [
	4,
	5,
	10,
	20,
	40,
	60,
	120
]

DAYS_SINCE_X = [
		5,
		10,
		20,
		40,
		60,
		120
	]

def add_technical_features(today: OHLC, records: list[OHLC], features: defaultdict[str, list[float]]):
	# add_rate_of_change("Close/Open", today.close, today.open)
	high_low = today.high - today.low
	if high_low == 0:
		high_low = 0.01
	close_high_low = tanh(today.close / high_low)
	# features["(Close-Open)/(High-Low)"].append(close_high_low)
	close_values = [x.close for x in records]

	def add_rate_of_change(name: str, new_value: float, old_value: float) -> None:
		value = get_rate_of_change(new_value, old_value)
		features[name].append(value)

	for days in MOMENTUM_DAYS:
		then = records[days - 1]
		add_rate_of_change(f"Momentum ({days} Days)", today.close, then.close)
		add_rate_of_change(f"Volume Momentum ({days} Days)", today.volume, then.volume)
		add_rate_of_change(f"Open Interest Momentum ({days} Days)", today.open_interest, then.open_interest)

	for days1, days2 in LAGGED_MOMENTUM_DAYS:
		add_rate_of_change(f"Lagged Momentum ({days1}, {days2} Days)", records[days1 - 1].close, records[days2 - 1].close)

	for days in MOVING_AVERAGE_DAYS:
		moving_average_values = close_values[:days]
		moving_average = sum(moving_average_values) / days
		feature_name = f"Close to Moving Average Ratio ({days} Days)"
		feature_value = get_rate_of_change(today.close, moving_average)
		features[feature_name].append(feature_value)

	for days in MOVING_AVERAGE_DAYS:
		moving_average_values1 = close_values[:days]
		moving_average1 = sum(moving_average_values1) / days
		moving_average_values2 = close_values[1:days + 1]
		moving_average2 = sum(moving_average_values2) / days
		feature_name = f"Moving Average Rate of Change ({days} Days)"
		feature_value = get_rate_of_change(moving_average1, moving_average2)
		features[feature_name].append(feature_value)

	for days in VOLATILITY_DAYS:
		feature_name = f"Volatility ({days} Days)"
		feature_value = get_daily_volatility(close_values, days)
		features[feature_name].append(feature_value)

	days_since_x_features = get_days_since_x_features(None, records, None)
	for feature in days_since_x_features:
		features[feature.name].append(feature.value)

def get_days_since_x_features(time: pd.Timestamp | None, records: list[OHLC], days_since_high_map: dict[pd.Timestamp, int] | None) -> list[Feature]:
	technical_features = []
	high_values = [ohlc.high for ohlc in records]
	low_values = [ohlc.low for ohlc in records]

	if time is not None and days_since_high_map is not None:
		# Days since last all-time high
		days_since_all_time_high = days_since_high_map[time]
		feature = Feature("Days Since Last All-Time High", FeatureCategory.TECHNICAL_DAYS_SINCE_X, days_since_all_time_high)
		technical_features.append(feature)

	# Days since last high within the last n days
	for days in DAYS_SINCE_X:
		values = high_values[:days]
		feature_name = f"Days Since High ({days} Days)"
		feature_value = values.index(max(values))
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_DAYS_SINCE_X, feature_value)
		technical_features.append(feature)

	# Days since last low within the last n days
	for days in DAYS_SINCE_X:
		values = low_values[:days]
		feature_name = f"Days Since Low ({days} Days)"
		feature_value = values.index(min(values))
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_DAYS_SINCE_X, feature_value)
		technical_features.append(feature)
	return technical_features

def get_rate_of_change(new_value: float | int, old_value: float | int):
	if old_value == 0:
		old_value = 0.01
	rate = tanh(float(new_value) / float(old_value) - 1.0)
	return rate

def get_days_since_high_map(data: TrainingData) -> dict[pd.Timestamp, int]:
	high_time = None
	high = None
	days_since_high_map = {}
	for ohlc in data.ohlc_series.values():
		if high is not None:
			days_since_high_map[ohlc.time] = (ohlc.time - high_time).days
		if high is None or ohlc.high > high:
			high_time = ohlc.time
			high = ohlc.high
	return days_since_high_map

def get_daily_volatility(close_values: list[float], days: int) -> float:
	values = close_values[:days + 1]
	returns = [get_rate_of_change(a, b) for a, b in zip(values, values[1:])]
	volatility = stdev(returns)
	return volatility

def get_barchart_path(symbol: str) -> str:
	return os.path.join(Configuration.BARCHART_DIRECTORY, f"{symbol}.D1.csv")