from typing import Final
import os
from glob import glob
from enum import Enum
from itertools import islice
import sys
import random
from statistics import median
from collections import defaultdict
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from series import TimeSeries
from ohlc import OHLC
from config import Configuration

class TrainingData:
	ohlc_series: TimeSeries[OHLC]
	fred_data: dict[str, TimeSeries[float]]

	def __init__(self, symbol: str):
		continuous_contract_path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, f"{symbol}.csv")
		self.ohlc_series = TimeSeries.read_ohlc_csv(continuous_contract_path)
		fred_path = os.path.join(Configuration.FRED_DIRECTORY, "*.csv")
		fred_paths = glob(fred_path)
		self.fred_data = {}
		for path in fred_paths:
			base_name = os.path.basename(path)
			fred_symbol = os.path.splitext(base_name)[0]
			self.fred_data[fred_symbol] = TimeSeries.read_csv(path)

class PostProcessing(Enum):
	# Apply no post-processing, directly use values from .csv file
	NOMINAL: Final[int] = 0
	# Generate two features, the nominal value f(t) and the delta f(t) - f(t - 1)
	NOMINAL_AND_DIFFERENCE: Final[int] = 1
	# Calculate f(t) / f(t - 1) - 1 for the most recent data point
	RATE_OF_CHANGE: Final[int] = 2

def get_rate_of_change(new_value: float | int, old_value: float | int):
	rate = float(new_value) / float(old_value) - 1.0
	# Limit the value to reduce the impact of outliers
	rate = min(rate, 1.0)
	return rate

def get_features(start: pd.Timestamp, end: pd.Timestamp, data: TrainingData, balanced_samples: bool = False, shuffle: bool = False) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.int8]]:
	assert start < end
	# Offset 0 for the current day, offset 1 for yesterday, to calculate the binary label p(t) / p(t - 1) > 1
	# Offset 1 also serves as a reference for momentum features
	offsets = [0, 1]
	# Number of days to look into the past to calculate relative returns, i.e. p(t - 1) / p(t - n) - 1
	momentum_offsets = [
		2,
		3,
		# 5,
		# 10,
		25,
		50,
		# 100,
		150,
		# 200,
		250,
		# 500
	]
	momentum_feature_names = [f"Momentum ({x} days)" for x in momentum_offsets]
	feature_names = []
	offsets += momentum_offsets
	max_offset = max(offsets)
	unbalanced_features: defaultdict[int, list[list[float]]] = defaultdict(list)
	# Skip initial OHLC time series keys to make sure that there is enough past data to calculate momentum
	ohlc_series_offset = islice(data.ohlc_series, max_offset, None)
	ohlc_keys = [t for t in ohlc_series_offset if start <= t < end]
	for time in ohlc_keys:
		time_feature_names, time_features = get_time_features(time)
		records: list[OHLC] = data.ohlc_series.get(time, offsets=offsets)
		close_values = [ohlc.close for ohlc in records]
		today = close_values[0]
		yesterday = close_values[1]
		momentum_close_values = close_values[2:]
		momentum_features = [get_rate_of_change(yesterday, close) for close in momentum_close_values]
		fred_feature_names, fred_features = get_fred_features(time, data)
		features = time_features + momentum_features + fred_features
		feature_names = time_feature_names + momentum_feature_names + fred_feature_names
		# Create a simple binary label for the most recent returns (i.e. comparing today and yesterday)
		label_rate = get_rate_of_change(today, yesterday)
		label = 1 if label_rate > 0 else 0
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
	return feature_names, np_features, np_labels

def get_time_features(time: pd.Timestamp) -> tuple[list[str], list[float]]:
	time_feature_names = [
		"Month",
		"Day of the Month",
		"Day of the Week",
		"Week of the Year"
	]
	time_features = [
		time.month,
		time.day,
		time.dayofweek,
		time.week,
	]
	return time_feature_names, time_features

def get_fred_features(time: pd.Timestamp, data: TrainingData) -> tuple[list[str], list[float]]:
	yesterday = time - pd.Timedelta(days=1)
	# FRED economic data
	fred_config = [
		# Initial unemployment claims (ICSA), nominal value, weekly
		("Initial Unemployment Claims", "ICSA", PostProcessing.RATE_OF_CHANGE),
		# Unemployment Rate (UNRATE), percentage, monthly
		("Unemployment Rate", "UNRATE", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity (T10Y2Y), percentage, daily
		# Excluded due to the data starting in 2020
		# ("10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity", "T10Y2Y", PostProcessing.NOMINAL),
		# 30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US), percentage, weekly
		("30-Year Fixed Rate Mortgage", "MORTGAGE30US", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Federal Funds Effective Rate (FEDFUNDS), percentage, monthly
		("Federal Funds Effective Rate", "FEDFUNDS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# M2 money supply (M2SL), nominal, monthly
		("M2 Money Supply", "M2SL", PostProcessing.RATE_OF_CHANGE),
		# Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL), nominal, weekly
		("Consumer Price Index for Urban Consumers", "CPIAUCSL", PostProcessing.RATE_OF_CHANGE),
		# Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level (WALCL), nominal, weekly
		("Total Assets (Less Eliminations from Consolidation)", "WALCL", PostProcessing.RATE_OF_CHANGE),
		# Real Gross Domestic Product (GDPC1), nominal, quarterly
		("Real Gross Domestic Product", "GDPC1", PostProcessing.RATE_OF_CHANGE),
		# All Employees, Total Nonfarm (PAYEMS), nominal, monthly
		("All Employees, Total Nonfarm", "PAYEMS", PostProcessing.RATE_OF_CHANGE),
		# Job Openings: Total Nonfarm (JTSJOL), nominal, monthly
		("Job Openings: Total Nonfarm", "JTSJOL", PostProcessing.RATE_OF_CHANGE),
	]
	feature_names = []
	features = []
	for config in fred_config:
		name, symbol, post_processing = config
		match post_processing:
			case PostProcessing.NOMINAL:
				value = data.fred_data[symbol].get(yesterday)
				features.append(value)
				feature_names.append(name)
			case PostProcessing.NOMINAL_AND_DIFFERENCE:
				values = data.fred_data[symbol].get(yesterday, count=2)
				nominal_value = values[0]
				difference = values[0] - values[1]
				features += [nominal_value, difference]
				feature_names += [
					name,
					f"{name} (Delta)"
				]
			case PostProcessing.RATE_OF_CHANGE:
				values = data.fred_data[symbol].get(yesterday, count=2)
				rate = get_rate_of_change(values[0], values[1])
				features.append(rate)
				feature_names.append(f"{name} (Rate of Change)")
	return feature_names, features

def train(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < split < end
	data = TrainingData(symbol)
	feature_names, x_train, y_train = get_features(start, split, data, balanced_samples=True)
	_, x_validation, y_validation = get_features(split, end, data)
	scaler = StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_validation = scaler.transform(x_validation)

	# Statistics
	precision_values = []
	roc_auc_values = []
	f1_scores = []
	heatmap_data = []
	best_model = None
	best_score = None

	# Iterate over hyperparameters
	# num_leaves_values = [31, 60, 90, 120, 180, 255]
	num_leaves_values = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 50, 55, 60]
	# num_iterations_values = [75, 100, 200, 300, 500, 1000]
	num_iterations_values = [200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 10000]
	inner_iterations = 1
	for num_leaves in num_leaves_values:
		heatmap_row = []
		for num_iterations in num_iterations_values:
			samples = []
			for _ in range(inner_iterations):
				params = {
					"verbosity": -1,
					"objective": "binary",
					# "metric": ["binary_logloss", "auc"],
					# "metric": "binary_logloss",
					"metric": "average_precision",
					"num_iterations": num_iterations,
					"num_leaves": num_leaves,
				}
				train_dataset = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
				validation_dataset = lgb.Dataset(x_validation, label=y_validation, feature_name=feature_names, reference=train_dataset)
				model = lgb.train(params, train_dataset, valid_sets=[validation_dataset])

				print(f"num_leaves: {num_leaves}, num_iterations: {num_iterations}")
				# print_metrics("Training", model, x_train, y_train)
				print_metrics("Validation", model, x_validation, y_validation)

				predictions = model.predict(x_validation)
				predictions = binary_predictions(predictions)
				precision = precision_score(y_validation, predictions)
				precision_values.append(precision)
				roc_auc = roc_auc_score(y_validation, predictions)
				roc_auc_values.append(roc_auc)
				f1 = f1_score(y_validation, predictions)
				f1_scores.append(f1)
				samples.append(f1)

				if best_score is None or precision > best_score:
					best_score = precision
					best_model = model

			sample = median(samples)
			heatmap_row.append(sample)
		heatmap_data.append(heatmap_row)

	median_precision = median(precision_values)
	random_precision = get_random_precision(y_validation)
	label_distribution = get_label_distribution(y_validation)
	median_roc_auc = median(roc_auc_values)
	median_f1 = median(f1_scores)
	print(f"Median precision: {median_precision:.1%}")
	print(f"Random precision: {random_precision:.1%}")
	print(f"Positive labels: {label_distribution[1]:.1%}")
	print(f"Negative labels: {label_distribution[0]:.1%}")
	print(f"Median ROC-AUC: {median_roc_auc:.3f}")
	print(f"Median F1: {median_f1:.3f}")

	# SHAP summary
	explainer = shap.TreeExplainer(best_model)
	shap_values = explainer(x_train)
	shap_values.feature_names = feature_names
	shap.summary_plot(shap_values, x_train, max_display=25, show=False)
	plt.gcf().canvas.manager.set_window_title("SHAP Summary Plot")
	plt.show()

	# Show heatmap of hyperparameters
	x_tick_labels = [str(x) for x in num_iterations_values]
	y_tick_labels = [str(x) for x in num_leaves_values]
	seaborn.heatmap(heatmap_data, xticklabels=x_tick_labels, yticklabels=y_tick_labels, cbar_kws={"label": "F1 Score"})
	plt.xlabel("Iterations")
	plt.ylabel("Leaves")
	plt.gca().invert_yaxis()
	plt.gcf().canvas.manager.set_window_title("Hyperparameters Heatmap")
	plt.show()

def binary_predictions(predictions: npt.NDArray[np.float64]) -> npt.NDArray[np.int8]:
	return (predictions > 0.5).astype(dtype=np.int8)

def get_random_precision(y_validation: npt.NDArray[np.int8]):
	values = [0, 1]
	choices = random.choices(values, k=len(y_validation))
	hits = 0
	for a, b in zip(choices, y_validation):
		if a == b:
			hits += 1
	precision = float(hits) / len(y_validation)
	return precision

def get_label_distribution(y_validation: npt.NDArray[np.int8]) -> defaultdict[int, float]:
	output = defaultdict(float)
	for label in y_validation:
		output[label] += 1.0
	for label in output:
		output[label] /= len(y_validation)
	return output

def print_metrics(title, model, x, y):
	predictions = model.predict(x)
	predictions = binary_predictions(predictions)
	precision = precision_score(y, predictions)
	roc_auc = roc_auc_score(y, predictions)
	f1 = f1_score(y, predictions)
	matrix = confusion_matrix(y, predictions)
	print(f"\t{title}:")
	print(f"\t\tPrecision: {precision:.1%}")
	print(f"\t\tROC-AUC: {roc_auc:.3f}")
	print(f"\t\tF1: {f1:.3f}")
	print(f"\t\t[{matrix[0][0]:6d} {matrix[0][1]:6d}]")
	print(f"\t\t[{matrix[1][0]:6d} {matrix[1][1]:6d}]")

def main() -> None:
	if len(sys.argv) != 5:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbol> <start date> <split date> <end date>")
		return
	symbol = sys.argv[1]
	start = pd.Timestamp(sys.argv[2])
	split = pd.Timestamp(sys.argv[3])
	end = pd.Timestamp(sys.argv[4])
	train(symbol, start, split, end)

main()