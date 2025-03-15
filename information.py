import os
import sys
from collections import defaultdict

import pandas as pd
from scipy.stats import spearmanr

from data import TrainingData
from economic import get_barchart_features
from fred import get_fred_features
from technical import get_rate_of_change, get_daily_volatility, get_days_since_x_features
from config import Configuration

def analyze(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
	data = TrainingData(symbol)
	time_range = [t for t in data.ohlc_series if start <= t < end]
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
	volatility_days = [
		4,
		5,
		10,
		20,
		40,
		60,
		120
	]
	series_count = max(momentum_days)

	returns = []
	features: defaultdict[str, list[float]] = defaultdict(list)

	def add_rate_of_change(feature_name: str, new_value: float, old_value: float):
		feature_value = get_rate_of_change(new_value, old_value)
		features[feature_name].append(feature_value)

	for time in time_range:
		future_time = time + pd.Timedelta(days=1)
		future = data.ohlc_series.get(future_time)
		records = data.ohlc_series.get(time, count=series_count)
		close_values = [x.close for x in records]
		today = records[0]
		future_returns = get_rate_of_change(future.close, today.close)
		returns.append(future_returns)

		add_rate_of_change("Close/Open", today.close, today.open)
		high_low = today.high - today.low
		if high_low == 0:
			high_low = 0.01
		close_high_low = max(today.close / high_low, 10.0)
		features["(Close-Open)/(High-Low)"].append(close_high_low)

		for days in momentum_days:
			then = records[days - 1]
			add_rate_of_change(f"Momentum ({days} Days)", today.close, then.close)
			add_rate_of_change(f"Volume Momentum ({days} Days)", today.volume, then.volume)
			add_rate_of_change(f"Open Interest Momentum ({days} Days)", today.open_interest, then.open_interest)

		for days1, days2 in lagged_momentum_days:
			add_rate_of_change(f"Lagged Momentum ({days1}, {days2} Days)", records[days1 - 1].close, records[days2 - 1].close)

		for days in moving_average_days:
			moving_average_values = close_values[:days]
			moving_average = sum(moving_average_values) / days
			feature_name = f"Close to Moving Average Ratio ({days} Days)"
			feature_value = get_rate_of_change(today.close, moving_average)
			features[feature_name].append(feature_value)

		for days in moving_average_days:
			moving_average_values1 = close_values[:days]
			moving_average1 = sum(moving_average_values1) / days
			moving_average_values2 = close_values[1:days + 1]
			moving_average2 = sum(moving_average_values2) / days
			feature_name = f"Moving Average Rate of Change ({days} Days)"
			feature_value = get_rate_of_change(moving_average1, moving_average2)
			features[feature_name].append(feature_value)

		for days in volatility_days:
			feature_name = f"Volatility ({days} Days)"
			feature_value = get_daily_volatility(close_values, days)
			features[feature_name].append(feature_value)

		days_since_x_features = get_days_since_x_features(None, records, None)
		for feature in days_since_x_features:
			features[feature.name].append(feature.value)

		fred_features = get_fred_features(time, data)
		barchart_features = get_barchart_features(time, data)
		economic_features = fred_features + barchart_features
		for economic_feature in economic_features:
			features[economic_feature.name].append(economic_feature.value)

	results: list[tuple[str, float, float]] = []
	for feature_name, feature_values in features.items():
		significance = spearmanr(returns, feature_values) # type: ignore
		if significance.pvalue < 0.05:
			results.append((feature_name, significance.statistic, significance.pvalue))
	results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
	results_df = pd.DataFrame(results, columns=["Feature", "Spearman's rho", "p-value"])
	path = os.path.join(Configuration.PLOT_DIRECTORY, "IC", f"{symbol}.csv")
	results_df.to_csv(path, index=False, float_format="%.5f")
	print(f"Wrote {path}")

def main() -> None:
	if len(sys.argv) != 4:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbols> <start date> <end date>")
		return
	symbols = [x.strip() for x in sys.argv[1].split(",")]
	start = pd.Timestamp(sys.argv[2])
	end = pd.Timestamp(sys.argv[3])
	for symbol in symbols:
		analyze(symbol, start, end)

main()