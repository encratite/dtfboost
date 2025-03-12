import os
from statistics import stdev

import pandas as pd

from ohlc import OHLC
from data import TrainingData
from config import Configuration

def get_technical_features(time: pd.Timestamp, days_since_high_map: dict[pd.Timestamp, int], data: TrainingData) -> tuple[list[str], list[float], int]:
	# Offset 0 for the current day, offset 1 for yesterday, to calculate the binary label p(t) / p(t - 1) > 1
	# Offset 1 also serves as a reference for momentum features
	offsets = [0, 1]
	# Number of days to look into the past to calculate relative returns, i.e. p(t - 1) / p(t - n) - 1
	momentum_offsets = [
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
	moving_averages = [
		5,
		10,
		25,
		50,
		100,
		150,
		200,
		250
	]
	volatility_days = [
		5,
		10,
		20,
		40,
		60,
		120
	]
	technical_feature_names = [f"Momentum ({x} Days)" for x in momentum_offsets]
	technical_feature_names += [f"Price Minus Moving Average ({x} Days)" for x in moving_averages]
	offsets += momentum_offsets
	# Technically days_since_x should be part of this calculation
	ohlc_count = max(max(offsets), max(moving_averages)) + 1
	records: list[OHLC] = data.ohlc_series.get(time, count=ohlc_count)
	today = records[0].close
	# Truncate records to prevent any further access to data from the future
	records = records[1:]
	close_values = [ohlc.close for ohlc in records]
	yesterday = close_values[0]
	momentum_close_values = [close_values[i - 1] for i in momentum_offsets]
	technical_features = [get_rate_of_change(yesterday, close) for close in momentum_close_values]
	for moving_average_days in moving_averages:
		moving_average_values = close_values[:moving_average_days]
		assert len(moving_average_values) == moving_average_days
		# Calculate price minus simple moving average
		moving_average = sum(moving_average_values) / moving_average_days
		moving_average_feature = yesterday - moving_average
		technical_features.append(moving_average_feature)
	days_since_x_feature_names, days_since_x_features = get_days_since_x_features(time, records, days_since_high_map)
	technical_feature_names += days_since_x_feature_names
	technical_features += days_since_x_features
	# Daily volatility
	technical_feature_names += [f"Volatility ({x} Days)" for x in volatility_days]
	technical_features += [get_daily_volatility(close_values, x) for x in volatility_days]
	# Create a simple binary label for the most recent returns (i.e. comparing today and yesterday)
	label_rate = get_rate_of_change(today, yesterday)
	label = 1 if label_rate > 0 else 0
	return technical_feature_names, technical_features, label

def get_days_since_x_features(time: pd.Timestamp, records: list[OHLC], days_since_high_map: dict[pd.Timestamp, int]) -> tuple[list[str], list[float]]:
	days_since_x = [
		20,
		40,
		60,
		120
	]
	technical_feature_names: list[str] = []
	technical_features: list[float] = []
	high_values = [ohlc.high for ohlc in records]
	low_values = [ohlc.low for ohlc in records]
	# Days since last all-time high
	technical_feature_names.append("Days Since Last All-Time High")
	days_since_all_time_high = days_since_high_map[time]
	technical_features.append(days_since_all_time_high)
	# Days since last high within the last n days
	technical_feature_names += [f"Days Since High ({x} Days)" for x in days_since_x]
	for days in days_since_x:
		values = high_values[:days]
		index = values.index(max(values))
		technical_features.append(index)
	# Days since last low within the last n days
	technical_feature_names += [f"Days Since Low ({x} Days)" for x in days_since_x]
	for days in days_since_x:
		values = low_values[:days]
		index = values.index(min(values))
		technical_features.append(index)
	return technical_feature_names, technical_features

def get_rate_of_change(new_value: float | int, old_value: float | int):
	rate = float(new_value) / float(old_value) - 1.0
	# Limit the value to reduce the impact of outliers
	rate = min(rate, 1.0)
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
	values = close_values[:days]
	returns = [get_rate_of_change(a, b) for a, b in zip(values, values[1:])]
	volatility = stdev(returns)
	return volatility

def get_barchart_path(symbol: str) -> str:
	return os.path.join(Configuration.BARCHART_DIRECTORY, f"{symbol}.D1.csv")