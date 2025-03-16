import os
import sys
from collections import defaultdict
from math import tanh
from multiprocessing import Pool
import calendar

import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from config import Configuration
from data import TrainingData
from economic import get_barchart_features
from fred import get_fred_features
from technical import get_rate_of_change, get_daily_volatility, get_days_since_x_features

def analyze(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp, p_value: float) -> None:
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

	def add_rate_of_change(name: str, new_value: float, old_value: float):
		value = get_rate_of_change(new_value, old_value)
		features[name].append(value)

	for time in time_range:
		future_time = time + pd.Timedelta(days=1)
		future = data.ohlc_series.get(future_time, right=True)
		records = data.ohlc_series.get(time, count=series_count)
		close_values = [x.close for x in records]
		today = records[0]
		assert time == today.time
		assert future.time > today.time
		future_returns = get_rate_of_change(future.close, today.close)
		returns.append(future_returns)

		for i in range(len(calendar.day_name)):
			feature_name = f"Seasonality: {calendar.day_name[i]}"
			feature_value = 1 if i == time.dayofweek else 0
			features[feature_name].append(feature_value)

		for i in range(len(calendar.month_name) - 1):
			month_index = i + 1
			feature_name = f"Seasonality: {calendar.month_name[month_index]}"
			feature_value = 1 if month_index == time.month else 0
			features[feature_name].append(feature_value)

		add_rate_of_change("Close/Open", today.close, today.open)
		high_low = today.high - today.low
		if high_low == 0:
			high_low = 0.01
		close_high_low = tanh(today.close / high_low)
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
	significant_features = []
	for feature_name, feature_values in features.items():
		if all(x == feature_values[0] for x in feature_values):
			continue
		significance = spearmanr(returns, feature_values) # type: ignore
		if significance.pvalue < p_value:
			results.append((feature_name, significance.statistic, significance.pvalue))
			significant_features.append(feature_values)
	results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
	results_df = pd.DataFrame(results, columns=["Feature", "Spearman's rho", "p-value"])
	path = os.path.join(Configuration.PLOT_DIRECTORY, "IC", f"{symbol}.csv")
	results_df.to_csv(path, index=False, float_format="%.5f")
	print(f"Wrote {path}")

	training_samples = len([time for time in time_range if time < split])
	regression_features = [list(row) for row in zip(*significant_features)]
	x_training = regression_features[:training_samples]
	x_validation = regression_features[training_samples:]
	y_training = returns[:training_samples]
	y_validation = returns[training_samples:]
	regression_test(symbol, x_training, y_training, x_validation, y_validation)

def regression_test(symbol: str, x_training: list[list[float]], y_training: list[float], x_validation: list[list[float]], y_validation: list[float]) -> None:
	model = LinearRegression()
	model.fit(x_training, y_training)
	predictions = model.predict(x_validation)
	buy_and_hold_performance = 1
	model_performance_long = 1
	model_performance_short = 1
	model_performance_long_short = 1
	for i in range(len(y_validation)):
		y_real = y_validation[i]
		y_predicted = predictions[i]
		returns = y_real + 1
		buy_and_hold_performance *= returns
		if y_predicted > 0:
			model_performance_long *= returns
			model_performance_long_short *= returns
		if y_predicted < 0:
			model_performance_short /= returns
			model_performance_long_short /= returns

	print(f"[{symbol}] Buy and hold performance: {buy_and_hold_performance - 1:+.2%}")
	print(f"[{symbol}] Linear model performance (long): {model_performance_long - 1:+.2%}")
	print(f"[{symbol}] Linear model performance (short): {model_performance_short - 1:+.2%}")
	print(f"[{symbol}] Linear model performance (long/short): {model_performance_long_short - 1:+.2%}")

def main() -> None:
	if len(sys.argv) != 6:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbols> <start date> <split date> <end date> <max p-value>")
		return
	symbols = [x.strip() for x in sys.argv[1].split(",")]
	start = pd.Timestamp(sys.argv[2])
	split = pd.Timestamp(sys.argv[3])
	end = pd.Timestamp(sys.argv[4])
	p_value = float(sys.argv[5])
	arguments = [(symbol, start, split, end, p_value) for symbol in symbols]
	with Pool(8) as pool:
		pool.starmap(analyze, arguments)

if __name__ == "__main__":
	main()