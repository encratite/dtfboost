import calendar
import os
import sys
from collections import defaultdict
from math import tanh
from multiprocessing import Pool
from statistics import mean
from typing import cast, Any

import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, ARDRegression, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import Configuration
from data import TrainingData
from economic import get_barchart_features
from enums import RebalanceFrequency
from fred import get_fred_features
from technical import get_rate_of_change, get_daily_volatility, get_days_since_x_features
from results import EvaluationResults

def evaluate(
		symbol: str,
		start: pd.Timestamp,
		split: pd.Timestamp,
		end: pd.Timestamp,
		rebalance_frequency: RebalanceFrequency,
		feature_limit: int | None
) -> list[EvaluationResults]:
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
	time_range = [t for t in data.ohlc_series if start <= t < end and not skip_date(t)]
	validation_times = [time for time in time_range if time >= split]

	match rebalance_frequency:
		case RebalanceFrequency.DAILY:
			forecast_days = 1
		case RebalanceFrequency.WEEKLY:
			forecast_days = 7
		case RebalanceFrequency.MONTHLY:
			forecast_days = 30
		case _:
			raise Exception("Unknown rebalance frequency value")

	first = data.ohlc_series.get(validation_times[0])
	last = data.ohlc_series.get(validation_times[-1])
	buy_and_hold_performance = last.close / first.close

	for time in time_range:
		future_time = time + pd.Timedelta(days=forecast_days)
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

		for i in range(31):
			day = i + 1
			feature_name = f"Seasonality: Day {day}"
			feature_value = 1 if day == time.day else 0
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
	rho_features = []
	for feature_name, feature_values in features.items():
		if all(x == feature_values[0] for x in feature_values):
			continue
		significance = spearmanr(returns, feature_values) # type: ignore
		results.append((feature_name, significance.statistic, significance.pvalue))
		rho_features.append((feature_values, significance.statistic))
	results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
	results_df = pd.DataFrame(results, columns=["Feature", "Spearman's rho", "p-value"])
	path = os.path.join(Configuration.PLOT_DIRECTORY, "IC", f"{symbol}.csv")
	results_df.to_csv(path, index=False, float_format="%.5f")
	print(f"Wrote {path}")

	rho_features = sorted(rho_features, key=lambda x: abs(x[1]), reverse=True)
	if feature_limit is not None:
		rho_features = rho_features[:feature_limit]
	significant_features = [features for features, _rho in rho_features]
	training_samples = len([time for time in time_range if time < split])
	regression_features = [list(row) for row in zip(*significant_features)]
	x_training = regression_features[:training_samples]
	x_validation = regression_features[training_samples:]
	y_training = returns[:training_samples]
	y_validation = returns[training_samples:]
	deltas_validation = deltas[training_samples:]
	output = regression_test(symbol, x_training, y_training, x_validation, y_validation, validation_times, deltas_validation, rebalance_frequency, buy_and_hold_performance)
	return output

def regression_test(
		symbol: str,
		x_training: list[list[float]],
		y_training: list[float],
		x_validation: list[list[float]],
		y_validation: list[float],
		validation_times: list[pd.Timestamp],
		deltas: list[float],
		rebalance_frequency: RebalanceFrequency,
		buy_and_hold_performance: float
	) -> list[EvaluationResults]:
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
	contracts = max(int(round(10000.0 / margin)), 1)
	slippage = 2 * contracts * (broker_fee + exchange_fee + tick_value)
	models = [
		("LinearRegression", LinearRegression(), {}, False),
		("LassoCV", LassoCV(max_iter=10000, random_state=Configuration.SEED), {}, False),
		("ElasticNetCV", ElasticNetCV(max_iter=10000, random_state=Configuration.SEED), {}, False),
		("ARDRegression", ARDRegression(), {}, False),
		("BayesianRidge", BayesianRidge(), {}, False),
		("RandomForestRegressor", RandomForestRegressor(n_estimators=200, criterion="squared_error", max_depth=6, random_state=Configuration.SEED), {}, False),
		# ("MLPRegressor", MLPRegressor(hidden_layer_sizes=(64, 32), activation="logistic", solver="lbfgs", max_iter=50, random_state=Configuration.SEED), {}, True),
	]

	# models += get_random_forest_models()
	models += get_mlp_models()

	scaler = StandardScaler()
	scaler.fit(x_training)

	x_training_scaled = scaler.transform(x_training, copy=True)
	x_validation_scaled = scaler.transform(x_validation, copy=True)

	print(f"[{symbol}] Contracts: {contracts}")
	print(f"[{symbol}] Number of features: {len(x_training[0])}")
	print(f"[{symbol}] Number of samples: {len(x_training)} for training, {len(x_validation)} for validation")
	output = []
	for model_name, model, parameters, enable_scaling in models:
		evaluation_results = EvaluationResults(buy_and_hold_performance, slippage, validation_times[0], validation_times[-1], rebalance_frequency, model_name, parameters)
		if enable_scaling:
			x_training_selected = x_training_scaled
			x_validation_selected = x_validation_scaled
		else:
			x_training_selected = x_training
			x_validation_selected = x_validation
		model.fit(x_training_selected, y_training)
		predictions = model.predict(x_validation_selected)
		last_trade_time: pd.Timestamp | None = None
		for i in range(len(y_validation)):
			time = validation_times[i]
			if last_trade_time is not None:
				if rebalance_frequency == RebalanceFrequency.WEEKLY:
					if time.week == last_trade_time.week:
						continue
				if rebalance_frequency == RebalanceFrequency.MONTHLY:
					if time.month == last_trade_time.month:
						continue
			delta = deltas[i]
			returns = contracts * delta / tick_size * tick_value
			y_predicted = predictions[i]
			long = y_predicted >= 0
			evaluation_results.submit_trade(returns, long)
			last_trade_time = time
		evaluation_results.print_stats(symbol)
		output.append(evaluation_results)

	return output

def get_random_forest_models() -> list[Any]:
	n_estimators_values = [
		# 50,
		# 100,
		# 150,
		200
	]
	criterion_values = [
		"squared_error",
		# "absolute_error",
		# "friedman_mse"
	]
	max_depths_values = [
		# 3,
		# 4,
		# 5,
		6,
		7,
		8
	]
	models = []
	for n_estimators in n_estimators_values:
		for criterion in criterion_values:
			for max_depth in max_depths_values:
				model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=Configuration.SEED)
				parameters = {
					"n_estimators": n_estimators,
					"criterion": criterion,
					"max_depth": max_depth
				}
				models.append(("RandomForestRegressor", model, parameters, False))
	return models

def get_mlp_models():
	hidden_layer_sizes_values = [
		# (16, 8),
		# (16, 8, 4),
		# (20, 10),
		(24, 12),
		# (32, 16),
		# (32, 16, 8),
		# (32, 32),
		# (40, 20),
		# (48, 24),
		# (56, 28),
		# (64, 32),
		# (64, 64)
	]
	activation_values = [
		# "identity",
		"logistic",
		# "tanh",
		# "relu"
	]
	solver_values = [
		"lbfgs",
		# "sgd",
		# "adam"
	]
	max_iter_values = [
		25,
		30,
		40,
		50,
		# 55,
		60,
		# 70,
		# 100,
		200,
		# 500,
		1000,
		# 2000
	]
	models = []
	for hidden_layer_sizes in hidden_layer_sizes_values:
		for activation in activation_values:
			for solver in solver_values:
				for max_iter in max_iter_values:
					parameters = {
						"hidden_layer_sizes": hidden_layer_sizes,
						"activation": activation,
						"solver": solver,
						"max_iter": max_iter
					}
					model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, random_state=Configuration.SEED)
					models.append(("MLPRegressor", model, parameters, True))
	return models

def format_currency(value: float) -> str:
	if value >= 0:
		return f"${value:,.2f}"
	else:
		return f"(${abs(value):,.2f})"

def main() -> None:
	if len(sys.argv) != 7:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbols> <start date> <split date> <end date> <rebalance frequency> <feature limit>")
		print(f"Supported rebalance frequencies: daily, weekly, monthly")
		return
	symbols = [x.strip() for x in sys.argv[1].split(",")]
	start = pd.Timestamp(sys.argv[2])
	split = pd.Timestamp(sys.argv[3])
	end = pd.Timestamp(sys.argv[4])
	rebalance_frequency = cast(RebalanceFrequency, RebalanceFrequency[sys.argv[5].upper()])
	feature_limit = int(sys.argv[6])
	assert start < split < end
	results: list[EvaluationResults]
	if Configuration.ENABLE_MULTIPROCESSING:
		arguments = [(symbol, start, split, end, rebalance_frequency, feature_limit) for symbol in symbols]
		with Pool(8) as pool:
			nested_results = pool.starmap(evaluate, arguments)
			results = [item for sublist in nested_results for item in sublist]
	else:
		results = []
		for symbol in symbols:
			results += evaluate(symbol, start, split, end, rebalance_frequency, feature_limit)
	total_model_performance = defaultdict(list)
	for evaluation_results in results:
		total_model_performance[evaluation_results.model_name].append(evaluation_results)
	print("")
	hyperparameters = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	for evaluation_results in results:
		for parameter_name, parameter_value in evaluation_results.parameters.items():
			performance = evaluation_results.get_annualized_performance()
			hyperparameters[evaluation_results.model_name][parameter_name][parameter_value].append(performance)
	for model_name, parameter_dict in hyperparameters.items():
		print(f"Hyperparameters for {model_name}:")
		for parameter_name, parameter_value_dict in parameter_dict.items():
			print(f"\t{parameter_name}:")
			for parameter_value, performance_values in parameter_value_dict.items():
				mean_performance = mean(performance_values)
				print(f"\t\t{parameter_value}: {EvaluationResults.get_performance_string(mean_performance)}")

	all_model_performance_values = []
	for model_name, evaluation_results in total_model_performance.items():
		long_performance_values = mean([x.get_annualized_long_performance() for x in evaluation_results])
		short_performance_values = mean([x.get_annualized_short_performance() for x in evaluation_results])
		all_performance_values = mean([x.get_annualized_performance() for x in evaluation_results])
		all_model_performance_values.append(all_performance_values)
		# print(f"[{model_name}] Mean performance (long): {EvaluationResults.get_performance_string(long_performance_values)}")
		# print(f"[{model_name}] Mean performance (short): {EvaluationResults.get_performance_string(short_performance_values)}")
		print(f"[{model_name}] Mean performance (all): {EvaluationResults.get_performance_string(all_performance_values)}")
	mean_model_performance = mean(all_model_performance_values)
	print(f"Mean of all models with a feature limit of {feature_limit}: {EvaluationResults.get_performance_string(mean_model_performance)}")
	print(f"Symbols evaluated: {symbols}")

if __name__ == "__main__":
	main()