import os
from statistics import stdev
from math import tanh

import pandas as pd

from config import Configuration
from data import TrainingData
from enums import FeatureCategory
from feature import Feature
from ohlc import OHLC

def get_technical_features(time: pd.Timestamp, days_since_high_map: dict[pd.Timestamp, int], data: TrainingData) -> tuple[list[Feature], int]:
	# Offset 0 for the current day, offset 1 for yesterday, to calculate the binary label p(t) / p(t - 1) > 1
	# Offset 1 also serves as a reference for momentum features
	offsets = [0, 1]
	# Number of days to look into the past to calculate relative returns, i.e. p(t - 1) / p(t - n) - 1
	momentum_days = [
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
	lagged_momentum_days = [
		(25, 50),
		(100, 250)
	]
	moving_average_days = [
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
		60
	]
	offsets += momentum_days
	# Technically days_since_x should be part of this calculation
	ohlc_count = max(max(offsets), max(moving_average_days)) + 1
	records: list[OHLC] = data.ohlc_series.get(time, count=ohlc_count)
	today = records[0]
	# Truncate records to prevent any further access to data from the future
	records = records[1:]
	close_values = [ohlc.close for ohlc in records]
	yesterday = records[0]
	technical_features: list[Feature] = []

	# Price momentum features
	for days in momentum_days:
		close = close_values[days - 1]
		feature_value = get_rate_of_change(yesterday.close, close)
		feature_name = f"Momentum ({days} Days)"
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_MOMENTUM, feature_value)
		technical_features.append(feature)

	# Price minus moving average features
	for days in moving_average_days:
		moving_average_values = close_values[:days]
		# Calculate price minus simple moving average
		moving_average = sum(moving_average_values) / days
		feature_value = yesterday.close - moving_average
		feature_name = f"Price Minus Moving Average ({days} Days)"
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_MOVING_AVERAGE, feature_value)
		technical_features.append(feature)

	technical_features += get_days_since_x_features(time, records, days_since_high_map)

	# Daily volatility
	for days in volatility_days:
		feature_name = f"Volatility ({days} Days)"
		feature_value = get_daily_volatility(close_values, days)
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_VOLATILITY, feature_value)
		technical_features.append(feature)

	# Volume/open interest momentum features
	# This is problematic because lots of OHLC sources don't consolidate these values until after 4 PM ET
	for days in momentum_days:
		record = records[days - 1]
		feature_value = get_rate_of_change(yesterday.volume, record.volume)
		feature_name = f"Volume Momentum ({days} Days)"
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_VOLUME, feature_value)
		technical_features.append(feature)
		feature_value = get_rate_of_change(yesterday.open_interest, record.open_interest)
		feature_name = f"Open Interest ({days} Days)"
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_OPEN_INTEREST, feature_value)
		technical_features.append(feature)

	# Experimental indicators
	feature_name = f"Close-Open"
	feature_value = get_rate_of_change(yesterday.close, yesterday.open)
	feature = Feature(feature_name, FeatureCategory.TECHNICAL_EXPERIMENTAL, feature_value)
	technical_features.append(feature)

	feature_name = f"Close-High-Low"
	feature_value = yesterday.close / (yesterday.high - yesterday.low)
	feature = Feature(feature_name, FeatureCategory.TECHNICAL_EXPERIMENTAL, feature_value)
	technical_features.append(feature)

	for lag1, lag2 in lagged_momentum_days:
		close1 = close_values[lag1 - 1]
		close2 = close_values[lag2 - 1]
		feature_value = get_rate_of_change(close1, close2)
		feature_name = f"Lagged Momentum ({lag1}, {lag2} Days)"
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_EXPERIMENTAL, feature_value)
		technical_features.append(feature)

	# Create a simple binary label for the most recent returns (i.e. comparing today and yesterday)
	label_rate = get_rate_of_change(today.close, yesterday.close)
	label = 1 if label_rate > 0 else 0
	return technical_features, label

def get_days_since_x_features(time: pd.Timestamp | None, records: list[OHLC], days_since_high_map: dict[pd.Timestamp, int] | None) -> list[Feature]:
	days_since_x = [
		5,
		10,
		20,
		40,
		60,
		120
	]
	technical_features = []
	high_values = [ohlc.high for ohlc in records]
	low_values = [ohlc.low for ohlc in records]

	if time is not None and days_since_high_map is not None:
		# Days since last all-time high
		days_since_all_time_high = days_since_high_map[time]
		feature = Feature("Days Since Last All-Time High", FeatureCategory.TECHNICAL_DAYS_SINCE_X, days_since_all_time_high)
		technical_features.append(feature)

	# Days since last high within the last n days
	for days in days_since_x:
		values = high_values[:days]
		feature_name = f"Days Since High ({days} Days)"
		feature_value = values.index(max(values))
		feature = Feature(feature_name, FeatureCategory.TECHNICAL_DAYS_SINCE_X, feature_value)
		technical_features.append(feature)

	# Days since last low within the last n days
	for days in days_since_x:
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