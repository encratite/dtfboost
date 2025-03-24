import os
from collections import defaultdict
from typing import cast

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from scipy.stats.mstats import winsorize

from config import Configuration
from data import TrainingData
from economic import get_barchart_features
from enums import RebalanceFrequency
from fred import get_fred_features
from regression import perform_regression
from results import EvaluationResults
from seasonality import add_seasonality_features
from technical import add_technical_features, get_rate_of_change, MOMENTUM_DAYS

def get_forecast_days(rebalance_frequency: RebalanceFrequency):
	match rebalance_frequency:
		case RebalanceFrequency.DAILY:
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
		feature_limit: int | None
) -> list[EvaluationResults]:
	assert not (Configuration.USE_PCA and Configuration.SELECT_K_BEST)
	returns = []
	deltas = []
	features: defaultdict[str, list[float]] = defaultdict(list)

	def skip_date(t: pd.Timestamp) -> bool:
		return Configuration.SKIP_COVID and pd.Timestamp("2020-03-01") <= t < pd.Timestamp("2020-11-01")

	data = TrainingData(symbol)
	time_range = [t for t in data.ohlc_series if start <= t < end and not skip_date(t)]
	validation_times = [time for time in time_range if time >= split]
	first = data.ohlc_series.get(validation_times[0])
	last = data.ohlc_series.get(validation_times[-1])
	buy_and_hold_performance = last.close / first.close
	add_features(start, time_range, rebalance_frequency, data, returns, deltas, features)
	results: list[tuple[str, float, float]] = []
	ranked_features = []
	for feature_name, feature_values in features.items():
		if all(x == feature_values[0] for x in feature_values):
			continue
		if Configuration.USE_PEARSON:
			significance = pearsonr(returns, feature_values)
		else:
			significance = spearmanr(returns, feature_values) # type: ignore
		results.append((feature_name, significance.statistic, significance.pvalue))
		ranked_features.append((feature_values, significance.statistic))
	results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
	results_df = pd.DataFrame(results, columns=["Feature", "Pearson" if Configuration.USE_PEARSON else "Spearman", "p-value"])
	path = os.path.join(Configuration.CORRELATION_DIRECTORY, f"{symbol}.csv")
	results_df.to_csv(path, index=False, float_format="%.5f")
	print(f"Wrote {path}")
	ranked_features = sorted(ranked_features, key=lambda x: abs(x[1]), reverse=True)
	if feature_limit is not None and not Configuration.SELECT_K_BEST:
		limit = Configuration.PCA_RANK_FILTER if Configuration.USE_PCA else feature_limit
		ranked_features = ranked_features[:limit]
	significant_features = [features for features, _statistic in ranked_features]
	if Configuration.WINSORIZE:
		for i in range(len(significant_features)):
			features_array = np.array(significant_features[i])
			winsorized_array = winsorize(features_array, (Configuration.WINSORIZE_LIMIT, Configuration.WINSORIZE_LIMIT))
			significant_features[i] = winsorized_array.tolist()
	training_samples = len([time for time in time_range if time < split])
	regression_features = [list(row) for row in zip(*significant_features)]
	x_training = regression_features[:training_samples]
	x_validation = regression_features[training_samples:]
	y_training = returns[:training_samples]
	y_validation = returns[training_samples:]
	deltas_validation = deltas[training_samples:]
	if Configuration.USE_PCA:
		pca = PCA(n_components=feature_limit)
		pca.fit(x_training)
		x_training = pca.transform(x_training)
		x_validation = pca.transform(x_validation)
	elif Configuration.SELECT_K_BEST:
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
		selected_features_path = os.path.join(Configuration.SELECTION_DIRECTORY, f"{symbol}-{feature_limit}.csv")
		selected_features_df = pd.DataFrame.from_dict(selected_features_dict)
		selected_features_df.to_csv(selected_features_path, index=False)
		print(f"Wrote {selected_features_path}")
		x_training = selector.transform(x_training)
		x_validation = selector.transform(x_validation)
	output = perform_regression(
		symbol,
		x_training,
		y_training,
		x_validation,
		y_validation,
		validation_times,
		deltas_validation,
		rebalance_frequency,
		buy_and_hold_performance
	)
	return output