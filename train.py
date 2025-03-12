import random
import sys
from collections import defaultdict
from itertools import islice
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from boosting import train_lightgbm, train_catboost, train_xgboost
from config import Configuration
from data import TrainingData
from economic import get_economic_features
from enums import Algorithm, FeatureCategory
from technical import get_technical_features, get_days_since_high_map
from stats import generate_stats
from feature import Feature

def get_features(start: pd.Timestamp, end: pd.Timestamp, data: TrainingData) -> tuple[pd.DataFrame, pd.DataFrame]:
	assert start < end

	feature_names = None
	all_features: list[list[float]] = []
	labels: list[float] = []

	# Skip initial OHLC time series keys to make sure that there is enough past data to calculate momentum
	ohlc_count = 250
	ohlc_keys_offset = islice(data.ohlc_series, ohlc_count, None)
	ohlc_keys_in_range = [t for t in ohlc_keys_offset if start <= t < end]
	days_since_high_map = get_days_since_high_map(data)
	for time in ohlc_keys_in_range:
		features = get_seasonality_features(time)
		technical_features, label = get_technical_features(time, days_since_high_map, data)
		features += technical_features
		labels.append(label)
		features += get_economic_features(time, data)
		if feature_names is None:
			feature_names = [x.name for x in features]
		feature_values = [x.value for x in features]
		all_features.append(feature_values)

	df_features = pd.DataFrame(all_features, columns=feature_names)
	df_labels = pd.DataFrame(labels, columns=["Label"])
	return df_features, df_labels

def get_seasonality_features(time: pd.Timestamp) -> list[Feature]:
	features = [
		Feature("Month", FeatureCategory.SEASONALITY, time.month),
		Feature("Day of the Month", FeatureCategory.SEASONALITY, time.day),
		Feature("Day of the Week", FeatureCategory.SEASONALITY, time.dayofweek),
		Feature("Week of the Year", FeatureCategory.SEASONALITY, time.week),
	]
	return features

def train(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp, algorithm: Algorithm) -> None:
	assert start < split < end
	data = TrainingData(symbol)
	x_training, y_training = get_features(start, split, data)
	x_validation, y_validation = get_features(split, end, data)

	# Scale features to improve model performance (or so they say)
	if Configuration.ENABLE_SCALING:
		scaler = StandardScaler()
		scaler.fit(x_training)
		x_train_scaled = scaler.transform(x_training)
		x_validation_scaled = scaler.transform(x_validation)
		x_training = pd.DataFrame(x_train_scaled, columns=x_training.columns)
		x_validation = pd.DataFrame(x_validation_scaled, columns=x_validation.columns)

	# Train models
	match algorithm:
		case Algorithm.LIGHTGBM:
			results = train_lightgbm(x_training, x_validation, y_training, y_validation)
		case Algorithm.CATBOOST:
			results = train_catboost(x_training, x_validation, y_training, y_validation)
		case Algorithm.XGBOOST:
			results = train_xgboost(x_training, x_validation, y_training, y_validation)
		case _:
			raise Exception("Unknown algorithm specified")

	generate_stats(symbol, x_training, x_validation, y_validation, results)

def main() -> None:
	if len(sys.argv) != 6:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbol> <start date> <split date> <end date> <algorithm>")
		algorithms = [x.lower() for x in Algorithm.__members__.keys()]
		algorithms_string = ", ".join(algorithms)
		print(f"Available algorithms: {algorithms_string}")
		return
	symbol = sys.argv[1]
	start = pd.Timestamp(sys.argv[2])
	split = pd.Timestamp(sys.argv[3])
	end = pd.Timestamp(sys.argv[4])
	algorithm = cast(Algorithm, Algorithm[sys.argv[5].upper()])
	train(symbol, start, split, end, algorithm)

main()