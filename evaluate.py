import os
from collections import defaultdict

import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA

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
		add_seasonality_features(time, features)
		add_technical_features(today, records, features)
		fred_features = get_fred_features(time, data)
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
	add_features(time_range, rebalance_frequency, data, returns, deltas, features)
	results: list[tuple[str, float, float]] = []
	rho_features = []
	for feature_name, feature_values in features.items():
		if all(x == feature_values[0] for x in feature_values):
			continue
		if Configuration.USE_PEARSON:
			significance = pearsonr(returns, feature_values)
		else:
			significance = spearmanr(returns, feature_values) # type: ignore
		results.append((feature_name, significance.statistic, significance.pvalue))
		rho_features.append((feature_values, significance.statistic))
	results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
	results_df = pd.DataFrame(results, columns=["Feature", "Pearson" if Configuration.USE_PEARSON else "Spearman", "p-value"])
	path = os.path.join(Configuration.PLOT_DIRECTORY, "IC", f"{symbol}.csv")
	results_df.to_csv(path, index=False, float_format="%.5f")
	print(f"Wrote {path}")
	rho_features = sorted(rho_features, key=lambda x: abs(x[1]), reverse=True)
	if feature_limit is not None:
		limit = Configuration.PCA_RANK_FILTER if Configuration.USE_PCA else feature_limit
		rho_features = rho_features[:limit]
	significant_features = [features for features, _rho in rho_features]
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