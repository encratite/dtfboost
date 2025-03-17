import calendar
import os
import sys
import random
from collections import defaultdict
from math import tanh
from statistics import mean
from multiprocessing import Pool

import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, ARDRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from config import Configuration
from data import TrainingData
from economic import get_barchart_features
from fred import get_fred_features
from technical import get_rate_of_change, get_daily_volatility, get_days_since_x_features

def analyze(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp, p_value: float) -> dict[str, float]:
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
	deltas = []
	features: defaultdict[str, list[float]] = defaultdict(list)

	def add_rate_of_change(name: str, new_value: float, old_value: float) -> None:
		value = get_rate_of_change(new_value, old_value)
		features[name].append(value)

	def skip_date(t: pd.Timestamp) -> bool:
		return Configuration.SKIP_COVID and pd.Timestamp("2020-03-01") <= t < pd.Timestamp("2020-11-01")

	data = TrainingData(symbol)
	# time_range = [t for t in data.ohlc_series if start <= t < end and not skip_date(t)]
	# time_range = [t for t in data.ohlc_series if start <= t < end and not skip_date(t) and t.dayofweek == 0]
	# time_range = [t for t in data.ohlc_series if start <= t < end and not skip_date(t) and t.dayofweek == 0 and t.week % 2 == 0]
	time_range = [t for t in data.ohlc_series if start <= t < end and not skip_date(t) and t.dayofweek == 0 and t.week % 4 == 0]

	for time in time_range:
		# future_time = time + pd.Timedelta(days=1)
		# future_time = time + pd.Timedelta(days=7)
		# future_time = time + pd.Timedelta(days=14)
		future_time = time + pd.Timedelta(days=28)
		future = data.ohlc_series.get(future_time, right=True)
		records = data.ohlc_series.get(time, count=series_count)
		close_values = [x.close for x in records]
		today = records[0]
		assert time == today.time
		assert future.time > today.time
		future_returns = get_rate_of_change(future.close, today.close)
		returns.append(future_returns)
		delta = future.close - today.close
		deltas.append(delta)

		for i in range(len(calendar.day_name)):
			feature_name = f"Seasonality: {calendar.day_name[i]}"
			feature_value = 1 if i == time.dayofweek else 0
			features[feature_name].append(feature_value)

		for i in range(len(calendar.month_name) - 1):
			month_index = i + 1
			feature_name = f"Seasonality: {calendar.month_name[month_index]}"
			feature_value = 1 if month_index == time.month else 0
			features[feature_name].append(feature_value)

		# add_rate_of_change("Close/Open", today.close, today.open)
		high_low = today.high - today.low
		if high_low == 0:
			high_low = 0.01
		close_high_low = tanh(today.close / high_low)
		# features["(Close-Open)/(High-Low)"].append(close_high_low)

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
		# This static offset is problematic, it should actually be specific to the contract
		# Closes are calculated at a different time for each Globex code
		barchart_features = get_barchart_features(time - pd.Timedelta(days=1), data)
		economic_features = fred_features + barchart_features
		for economic_feature in economic_features:
			features[economic_feature.name].append(economic_feature.value)

	results: list[tuple[str, float, float]] = []
	all_features = list(features.values())
	filtered_features = []
	for feature_name, feature_values in features.items():
		if all(x == feature_values[0] for x in feature_values):
			continue
		significance = spearmanr(returns, feature_values) # type: ignore
		if significance.pvalue < p_value:
			results.append((feature_name, significance.statistic, significance.pvalue))
			filtered_features.append(feature_values)
	results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
	results_df = pd.DataFrame(results, columns=["Feature", "Spearman's rho", "p-value"])
	path = os.path.join(Configuration.PLOT_DIRECTORY, "IC", f"{symbol}.csv")
	results_df.to_csv(path, index=False, float_format="%.5f")
	print(f"Wrote {path}")

	all_regression_data = get_regression_features(all_features, returns, deltas, time_range, split)
	filtered_regression_data = get_regression_features(filtered_features, returns, deltas, time_range, split)
	output = regression_test(symbol, all_regression_data, filtered_regression_data)
	return output

def get_regression_features(features: list[list[float]], returns: list[float], deltas: list[float], time_range: list[pd.Timestamp], split: pd.Timestamp) -> tuple[list[list[float]], list[list[float]], list[float], list[float], list[float]]:
	training_samples = len([time for time in time_range if time < split])
	regression_features = [list(row) for row in zip(*features)]
	x_training = regression_features[:training_samples]
	x_validation = regression_features[training_samples:]
	y_training = returns[:training_samples]
	y_validation = returns[training_samples:]
	deltas_validation = deltas[training_samples:]
	return x_training, x_validation, y_training, y_validation, deltas_validation

def regression_test(
		symbol: str,
		all_regression_data: tuple[list[list[float]], list[list[float]], list[float], list[float], list[float]],
		filtered_regression_data: tuple[list[list[float]], list[list[float]], list[float], list[float], list[float]]
	) -> dict[str, float]:
	assets = pd.read_csv(Configuration.ASSETS_CONFIG)
	rows = assets[assets["symbol"] == symbol]
	if len(rows) == 0:
		raise Exception(f"No such symbol in assets configuration: {symbol}")
	asset = rows.iloc[0].to_dict()
	tick_size = asset["tick_size"]
	tick_value = asset["tick_value"]
	broker_fee = asset["broker_fee"]
	exchange_fee = asset["exchange_fee"]
	margin = asset["margin"]
	contracts = min(int(round(10000.0 / margin)), 1)
	slippage = 2 * contracts * (broker_fee + exchange_fee + tick_value)
	models = [
		# ("LinearRegression", LinearRegression(), True),
		# ("LassoCV", LassoCV(max_iter=10000), True),
		# ("ElasticNetCV", ElasticNetCV(max_iter=10000), True),
		("ARDRegression", ARDRegression(), True),
		# ("BayesianRidge", BayesianRidge(), True),
		# ("RandomForestRegressor(n_estimators=100)", RandomForestRegressor(n_estimators=100, random_state=random.seed(Configuration.SEED)), True),
		("RandomForestRegressor(n_estimators=120)", RandomForestRegressor(n_estimators=120, random_state=random.seed(Configuration.SEED)), True),
		# ("RandomForestRegressor(n_estimators=50, max_depth=2)", RandomForestRegressor(n_estimators=50, max_depth=2, random_state=random.seed(Configuration.SEED)), True),
		# ("RandomForestRegressor(n_estimators=50, max_depth=3)", RandomForestRegressor(n_estimators=50, max_depth=3, random_state=random.seed(Configuration.SEED)), True),
		# ("RandomForestRegressor(n_estimators=100, max_depth=3)", RandomForestRegressor(n_estimators=100, max_depth=3, random_state=random.seed(Configuration.SEED)), True),
		# ("RandomForestRegressor(n_estimators=100, max_depth=4)", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=random.seed(Configuration.SEED)), True),
		# ("RandomForestRegressor(n_estimators=100, max_depth=5)", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor1", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor2", MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", solver="adam", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor3", MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor4", MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation="tanh", solver="adam", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor5", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="lbfgs", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor6", MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", solver="lbfgs", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor7", MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", solver="lbfgs", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor8", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor9", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", solver="sgd", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor10", MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", solver="sgd", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor11", MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh", solver="sgd", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor12", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="sgd", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor13", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=1500, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor14", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=2000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor15.1", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=1000, random_state=random.seed(Configuration.SEED)), True),
		("MLPRegressor15.2", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=1500, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor15.3", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=2000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor15.4", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=2500, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor15.5", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=3000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor16", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=10000, random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor17", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="tanh", solver="lbfgs", max_iter=10000, learning_rate="adaptive", random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor18", MLPRegressor(hidden_layer_sizes=(20, 10, 5), activation="tanh", solver="lbfgs", max_iter=10000, learning_rate="adaptive", random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor19", MLPRegressor(hidden_layer_sizes=(14, 6, 4), activation="tanh", solver="lbfgs", max_iter=10000, learning_rate="adaptive", random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor20", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="logistic", solver="lbfgs", max_iter=1000, learning_rate="adaptive", random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor21", MLPRegressor(hidden_layer_sizes=(16, 8, 4), activation="logistic", solver="lbfgs", max_iter=5000, learning_rate="adaptive", random_state=random.seed(Configuration.SEED)), True),
		# ("MLPRegressor22", MLPRegressor(hidden_layer_sizes=(32, 32, 16), activation="tanh", solver="lbfgs", max_iter=1000,  learning_rate="adaptive", random_state=random.seed(Configuration.SEED)), True),
	]
	output = {}
	for model_name, model, enable_filtering in models:
		x_training, x_validation, y_training, y_validation, deltas = filtered_regression_data if enable_filtering else all_regression_data
		model.fit(x_training, y_training)
		predictions = model.predict(x_validation)
		buy_and_hold_cash = Configuration.INITIAL_CASH
		long_only_cash = Configuration.INITIAL_CASH
		short_only_cash = Configuration.INITIAL_CASH
		long_short_cash = Configuration.INITIAL_CASH
		for i in range(len(y_validation)):
			delta = deltas[i]
			returns = contracts * delta / tick_size * tick_value
			buy_and_hold_cash += returns
			y_predicted = predictions[i]
			if y_predicted >= 0:
				long_only_cash += returns
				long_only_cash -= slippage
				long_short_cash += returns
			elif y_predicted < 0:
				short_only_cash -= returns
				short_only_cash -= slippage
				long_short_cash -= returns
			long_short_cash -= slippage

		print(f"[{symbol} {model_name}] Number of features: {len(x_training[0])}")
		print(f"[{symbol} {model_name}] Number of samples: {len(x_training)} for training, {len(x_validation)} for validation")
		print(f"[{symbol} {model_name}] Buy and hold performance: {get_performance(buy_and_hold_cash)}")
		print(f"[{symbol} {model_name}] Model performance (long): {get_performance(long_only_cash)}")
		print(f"[{symbol} {model_name}] Model performance (short): {get_performance(short_only_cash)}")
		print(f"[{symbol} {model_name}] Model performance (long/short): {get_performance(long_short_cash)}")
		output[model_name] = long_short_cash

	return output

def format_currency(value):
	if value >= 0:
		return f"${value:,.2f}"
	else:
		return f"(${abs(value):,.2f})"

def get_performance(cash):
	return f"{cash / Configuration.INITIAL_CASH - 1:+.2%}"

def main() -> None:
	if len(sys.argv) != 6:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbols> <start date> <split date> <end date> <max p-value>")
		return
	symbols = [x.strip() for x in sys.argv[1].split(",")]
	start = pd.Timestamp(sys.argv[2])
	split = pd.Timestamp(sys.argv[3])
	end = pd.Timestamp(sys.argv[4])
	assert start < split < end
	p_value = float(sys.argv[5])
	if Configuration.ENABLE_MULTIPROCESSING:
		arguments = [(symbol, start, split, end, p_value) for symbol in symbols]
		with Pool(8) as pool:
			model_performance = pool.starmap(analyze, arguments)
			total_model_performance = defaultdict(list)
			for performance_dict in model_performance:
				for model_name, cash in performance_dict.items():
					total_model_performance[model_name].append(cash)
			print("")
			all_cash_values = []
			for model_name, cash_values in total_model_performance.items():
				cash = mean(cash_values)
				all_cash_values += cash_values
				print(f"[{model_name}] Mean performance (long/short): {get_performance(cash)}")
			print(f"Mean of all models with p-value {p_value}: {get_performance(mean(all_cash_values))}")
	else:
		for symbol in symbols:
			analyze(symbol, start, split, end, p_value)

if __name__ == "__main__":
	main()