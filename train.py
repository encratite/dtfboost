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
from enums import Algorithm
from technical import get_technical_features, get_days_since_high_map
from stats import generate_stats

def get_features(start: pd.Timestamp, end: pd.Timestamp, data: TrainingData, balanced_samples: bool = False, shuffle: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
	assert start < end
	unbalanced_features: defaultdict[int, list[list[float]]] = defaultdict(list)
	feature_names = []
	ohlc_count = 250
	# Skip initial OHLC time series keys to make sure that there is enough past data to calculate momentum
	ohlc_keys_offset = islice(data.ohlc_series, ohlc_count, None)
	ohlc_keys_in_range = [t for t in ohlc_keys_offset if start <= t < end]
	days_since_high_map = get_days_since_high_map(data)
	for time in ohlc_keys_in_range:
		seasonality_feature_names, seasonality_features = get_seasonality_features(time)
		technical_feature_names, technical_features, label = get_technical_features(time, days_since_high_map, data)
		economic_feature_names, economic_features = get_economic_features(time, data)
		features = seasonality_features + technical_features + economic_features
		feature_names = seasonality_feature_names + technical_feature_names + economic_feature_names
		unbalanced_features[label].append(features)

	# Try to create a balanced data set by adding samples from the smaller class
	class_sizes = sorted(unbalanced_features.items(), key=lambda x: len(x[1]))
	assert len(class_sizes) == 2
	small_class_label, small_class_features = class_sizes[0]
	large_class_label, large_class_features = class_sizes[1]
	if balanced_samples:
		imbalance = len(large_class_features) - len(small_class_features)
	else:
		imbalance = 0
	if imbalance < len(small_class_features):
		# The imbalance is small enough to use sampling without replacement
		additional_samples = random.sample(small_class_features, k=imbalance)
	else:
		# The imbalance is too great, use sampling with replacement
		additional_samples = random.choices(small_class_features, k=imbalance)
	small_class_features += additional_samples
	balanced_data = [(x, small_class_label) for x in small_class_features] + [(x, large_class_label) for x in large_class_features]
	if shuffle:
		# Randomize the order of the samples (not sure if relevant for most algorithms)
		random.shuffle(balanced_data)
	features = [x[0] for x in balanced_data]
	np_features = np.array(features, dtype=np.float64)
	labels = [x[1] for x in balanced_data]
	np_labels = np.array(labels, dtype=np.int8)
	df_features = pd.DataFrame(np_features, columns=feature_names)
	df_labels = pd.DataFrame(np_labels, columns=["Label"])
	return df_features, df_labels

def get_seasonality_features(time: pd.Timestamp) -> tuple[list[str], list[float]]:
	seasonality_feature_names = [
		"Month",
		"Day of the Month",
		"Day of the Week",
		"Week of the Year"
	]
	seasonality_features = [
		time.month,
		time.day,
		time.dayofweek,
		time.week,
	]
	return seasonality_feature_names, seasonality_features

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