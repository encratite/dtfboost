import sys
from itertools import islice, combinations
from typing import cast

import pandas as pd
from tqdm import tqdm

from boosting import train_lightgbm, train_catboost, train_xgboost
from config import Configuration
from data import TrainingData
from economic import get_economic_features
from enums import Algorithm, FeatureCategory
from feature import Feature
from results import TrainingResults
from stats import generate_stats
from technical import get_technical_features, get_days_since_high_map

def get_features(start: pd.Timestamp, end: pd.Timestamp, data: TrainingData, feature_categories: frozenset[FeatureCategory] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
		if feature_categories is None:
			feature_names = [x.name for x in features]
			feature_values = [x.value for x in features]
		else:
			filtered_features = [x for x in features if x.category in feature_categories]
			if feature_names is None:
				feature_names = [x.name for x in filtered_features]
			feature_values = [x.value for x in filtered_features]
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
	results = TrainingResults()
	if Configuration.EVALUATE_FEATURE_CATEGORIES:
		feature_category_enums = list(FeatureCategory)
		enum_subsets = []
		for i in range(len(feature_category_enums)):
			for x in combinations(feature_category_enums, i + 1):
				enum_subsets.append(frozenset(x))
		for feature_categories in tqdm(enum_subsets, desc="Evaluating feature categories", colour="green"):
			x_training, y_training = get_features(start, split, data, feature_categories)
			x_validation, y_validation = get_features(split, end, data, feature_categories)
			execute_algorithm(x_training, x_validation, y_training, y_validation, algorithm, False, feature_categories, results)
		generate_stats(symbol, results, feature_categories=True)
	else:
		feature_categories = frozenset([
			FeatureCategory.SEASONALITY,
			FeatureCategory.TECHNICAL_MOMENTUM,
			FeatureCategory.TECHNICAL_MOVING_AVERAGE,
			FeatureCategory.TECHNICAL_VOLUME,
			FeatureCategory.TECHNICAL_OPEN_INTEREST,
			FeatureCategory.TECHNICAL_DAYS_SINCE_X,
			FeatureCategory.TECHNICAL_VOLATILITY,
			FeatureCategory.TECHNICAL_EXPERIMENTAL,
			FeatureCategory.ECONOMIC_INTEREST_RATES,
			FeatureCategory.ECONOMIC_GENERAL,
			FeatureCategory.ECONOMIC_RESOURCES,
			FeatureCategory.ECONOMIC_VOLATILITY,
			FeatureCategory.ECONOMIC_INDEXES,
			FeatureCategory.ECONOMIC_CURRENCIES,
		])
		x_training, y_training = get_features(start, split, data, feature_categories)
		x_validation, y_validation = get_features(split, end, data, feature_categories)
		execute_algorithm(x_training, x_validation, y_training, y_validation, algorithm, True, feature_categories, results)
		generate_stats(symbol, results, hyperparameters=True)

def execute_algorithm(
		x_training: pd.DataFrame,
		x_validation: pd.DataFrame,
		y_training: pd.DataFrame,
		y_validation: pd.DataFrame,
		algorithm: Algorithm,
		optimize: bool,
		feature_categories: frozenset[FeatureCategory] | None,
		results: TrainingResults
) -> None:
	match algorithm:
		case Algorithm.LIGHTGBM:
			train_lightgbm(x_training, x_validation, y_training, y_validation, optimize, feature_categories, results)
		case Algorithm.CATBOOST:
			train_catboost(x_training, x_validation, y_training, y_validation, optimize, feature_categories, results)
		case Algorithm.XGBOOST:
			train_xgboost(x_training, x_validation, y_training, y_validation, optimize, feature_categories, results)
		case _:
			raise Exception("Unknown algorithm specified")

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