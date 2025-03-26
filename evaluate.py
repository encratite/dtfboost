import os
from collections import defaultdict
from functools import partial
from typing import cast, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression

from config import Configuration
from data import TrainingData, RegressionDataset
from economic import get_barchart_features
from enums import RebalanceFrequency
from fred import get_fred_features
from regression import perform_regression
from results import EvaluationResults
from seasonality import add_seasonality_features
from technical import add_technical_features, get_rate_of_change, MOMENTUM_DAYS

def get_forecast_days(rebalance_frequency: RebalanceFrequency):
	match rebalance_frequency:
		case RebalanceFrequency.DAILY | RebalanceFrequency.DAILY_SPLIT:
			forecast_days = 1
		case RebalanceFrequency.WEEKLY:
			forecast_days = 7
		case RebalanceFrequency.MONTHLY:
			forecast_days = 30
		case _:
			raise Exception("Unknown rebalance frequency value")
	return forecast_days

def add_features(
		start: pd.Timestamp,
		time_range: list[pd.Timestamp],
		rebalance_frequency: RebalanceFrequency,
		data: TrainingData,
		returns: list[float],
		deltas: list[float],
		features: defaultdict[str, list[float]]
):
	series_count = max(MOMENTUM_DAYS)
	forecast_days = get_forecast_days(rebalance_frequency)
	for time in time_range:
		future_time = time + pd.Timedelta(days=forecast_days)
		future = data.ohlc_series.get(future_time, right=True)
		records = data.ohlc_series.get(time, count=series_count)
		today = records[0]
		assert time == today.time
		assert future.time > today.time
		future_returns = get_rate_of_change(future.close, today.close)
		returns.append(future_returns)
		delta = future.close - today.close
		deltas.append(delta)
		if Configuration.ENABLE_SEASONALITY_FEATURES:
			add_seasonality_features(time, features)
		if Configuration.ENABLE_TECHNICAL_FEATURES:
			add_technical_features(today, records, features)
		if Configuration.ENABLE_ECONOMIC_FEATURES:
			fred_features = get_fred_features(start, time, data)
			# This static offset is problematic, it should actually be specific to the contract
			# Closes are calculated at a different time for each Globex code
			barchart_features = get_barchart_features(time - pd.Timedelta(days=1), data)
			economic_features = fred_features + barchart_features
			for economic_feature in economic_features:
				features[economic_feature.name].append(economic_feature.value)

def evaluate(
		symbol: str,
		start: pd.Timestamp,
		split: pd.Timestamp,
		end: pd.Timestamp,
		rebalance_frequency: RebalanceFrequency,
		rebalance_frequency_string: str,
		feature_limit: int | None,
		process_id: int,
		process_count: int
) -> list[EvaluationResults]:
	assert not (Configuration.USE_PCA and Configuration.SELECT_K_BEST)
	returns = []
	deltas = []
	features: defaultdict[str, list[float]] = defaultdict(list)

	def filter_by_dayofweek(day: int, filter_time: pd.Timestamp) -> bool:
		return filter_time.dayofweek == day

	def skip_date(t: pd.Timestamp) -> bool:
		return Configuration.SKIP_COVID and pd.Timestamp("2020-03-01") <= t < pd.Timestamp("2020-11-01")

	data = TrainingData(symbol)
	time_range = [t for t in data.ohlc_series if start <= t < end and not skip_date(t)]
	training_times = [time for time in time_range if time < split]
	validation_times = [time for time in time_range if time >= split]
	first = data.ohlc_series.get(validation_times[0])
	last = data.ohlc_series.get(validation_times[-1])
	buy_and_hold_performance = last.close / first.close
	add_features(start, time_range, rebalance_frequency, data, returns, deltas, features)

	category_filters: list[tuple[int, str, Callable[[pd.Timestamp], bool]] | None] = [None]
	if rebalance_frequency == RebalanceFrequency.DAILY_SPLIT:
		days = [
			"Monday",
			"Tuesday",
			"Wednesday",
			"Thursday",
			"Friday",
		]
		category_filters = [(i, days_value, partial(filter_by_dayofweek, i)) for i, days_value in enumerate(days)]

	category_datasets: dict[int, RegressionDataset] = {}

	for category_configuration in category_filters:
		if category_configuration is None:
			category_id = None
			category_name = None
			time_filter = None
			filtered_indexes_training = None
			filtered_indexes_validation = None
		else:
			def get_filtered_indexes(time_values: list[pd.Timestamp]) -> list[int]:
				return [i for i in range(len(time_values)) if time_filter(time_values[i])]

			category_id, category_name, time_filter = category_configuration
			filtered_indexes_training = get_filtered_indexes(training_times)
			filtered_indexes_validation = get_filtered_indexes(validation_times)
		ranking_results: list[tuple[str, float, float]] = []
		ranked_features: list[tuple[list[float], list[float], float]] = []
		for feature_name, feature_values in features.items():
			if category_configuration is None:
				training_samples = len(training_times)
				training_features = feature_values[:training_samples]
				training_returns = returns[:training_samples]
				validation_features = feature_values[training_samples:]
			else:
				training_features = [feature_values[i] for i in filtered_indexes_training]
				training_returns = [returns[i] for i in filtered_indexes_training]
				validation_features = [feature_values[i] for i in filtered_indexes_validation]
			if all(x == training_features[0] for x in training_features):
				continue
			if Configuration.USE_PEARSON:
				significance = pearsonr(training_features, training_returns)
			else:
				significance = spearmanr(training_features, training_returns) # type: ignore
			ranking_results.append((feature_name, significance.statistic, significance.pvalue))
			ranked_features.append((training_features, validation_features, significance.statistic))
		ranking_results = sorted(ranking_results, key=lambda x: abs(x[1]), reverse=True)
		ranking_results_df = pd.DataFrame(ranking_results, columns=["Feature", "Pearson" if Configuration.USE_PEARSON else "Spearman", "p-value"])
		file_name = f"{symbol}-{rebalance_frequency_string}.csv" if category_configuration is None else f"{symbol}-{rebalance_frequency_string}-{category_name}.csv"
		path = os.path.join(Configuration.CORRELATION_DIRECTORY, file_name)
		ranking_results_df.to_csv(path, index=False, float_format="%.5f")
		ranked_features = sorted(ranked_features, key=lambda x: abs(x[2]), reverse=True)
		if feature_limit is not None and not Configuration.SELECT_K_BEST:
			limit = Configuration.PCA_RANK_FILTER if Configuration.USE_PCA else feature_limit
			ranked_features = ranked_features[:limit]
		significant_training_features = [trainig_features for trainig_features, _validation_features, _statistic in ranked_features]
		significant_validation_features = [validation_features for _trainig_features, validation_features, _statistic in ranked_features]
		if Configuration.WINSORIZE:
			for i, training_values in enumerate(significant_training_features):
				features_array = np.array(training_values)
				winsorized_array = winsorize(features_array, (Configuration.WINSORIZE_LIMIT, Configuration.WINSORIZE_LIMIT))
				minimum = np.min(winsorized_array)
				maximum = np.max(winsorized_array)
				significant_training_features[i] = winsorized_array.tolist()
				# Winsorize validation data separately using the min/max parameters from the training data
				validation_values = significant_validation_features[i]
				significant_validation_features[i] = [max(min(x, maximum), minimum) for x in validation_values]

		def transpose_features(f: list[list[float]]):
			return [list(row) for row in zip(*f)]

		x_training = transpose_features(significant_training_features)
		x_validation = transpose_features(significant_validation_features)
		if category_configuration is None:
			training_samples = len(training_times)
			y_training = returns[:training_samples]
			y_validation = returns[training_samples:]
			deltas_validation = deltas[training_samples:]
		else:
			y_training = [returns[i] for i in filtered_indexes_training]
			y_validation = [returns[i] for i in filtered_indexes_validation]
			deltas_validation = [deltas[i] for i in filtered_indexes_validation]

		if Configuration.USE_PCA:
			x_training, x_validation = apply_pca(x_training, x_validation, feature_limit)
		elif Configuration.SELECT_K_BEST:
			x_training, x_validation = select_k_best(
				symbol,
				category_name,
				x_training,
				y_training,
				x_validation,
				features,
				feature_limit,
				rebalance_frequency_string
			)

		dataset = RegressionDataset(
			x_training,
			y_training,
			x_validation,
			y_validation,
			training_times,
			validation_times,
			deltas_validation
		)
		category_datasets[category_id] = dataset

	output = perform_regression(
		symbol,
		category_datasets,
		category_filters,
		rebalance_frequency,
		buy_and_hold_performance,
		process_id,
		process_count
	)
	return output

def apply_pca(
		x_training: list[list[float]],
		x_validation: list[list[float]],
		feature_limit: int
) -> tuple[list[list[float]], list[list[float]]]:
	pca = PCA(n_components=feature_limit)
	pca.fit(x_training)
	x_training = pca.transform(x_training)
	x_validation = pca.transform(x_validation)
	return x_training, x_validation

def select_k_best(
		symbol: str,
		category_name: str | None,
		x_training: list[list[float]],
		y_training: list[list[float]],
		x_validation: list[list[float]],
		features: defaultdict[str, list[float]],
		feature_limit: int,
		rebalance_frequency_string: str
) -> tuple[list[list[float]], list[list[float]]]:
	match Configuration.SELECT_K_BEST_SCORE:
		case "mutual_info_regression":
			score_func = mutual_info_regression
		case "f_regression":
			score_func = f_regression
		case _:
			raise Exception(f"Unknown SelectKBest score function: \"{Configuration.SELECT_K_BEST_SCORE}\"")
	selector = SelectKBest(score_func=score_func, k=feature_limit)
	selector.fit(x_training, y_training)
	support = cast(list[bool], selector.get_support())
	selected_features = []
	feature_names = list(features.keys())
	for i in range(len(selector.scores_)):
		if support[i]:
			score = selector.scores_[i]
			feature_name = feature_names[i]
			selected_features.append((feature_name, score))
	selected_features = sorted(selected_features, key=lambda x: x[1], reverse=True)
	selected_features_dict = {
		"Feature": [feature_name for feature_name, _score in selected_features],
		"Score": [score for _feature_name, score in selected_features]
	}
	if category_name is None:
		file_name = f"{symbol}-{rebalance_frequency_string}-{Configuration.SELECT_K_BEST_SCORE}-{feature_limit}.csv"
	else:
		file_name = f"{symbol}-{rebalance_frequency_string}-{category_name}-{Configuration.SELECT_K_BEST_SCORE}-{feature_limit}.csv"
	selected_features_path = os.path.join(Configuration.SELECTION_DIRECTORY, file_name)
	selected_features_df = pd.DataFrame.from_dict(selected_features_dict)
	selected_features_df.to_csv(selected_features_path, index=False)
	print(f"Wrote {selected_features_path}")
	x_training = selector.transform(x_training)
	x_validation = selector.transform(x_validation)
	return x_training, x_validation