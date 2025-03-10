import os
import sys
import random
from statistics import median
from collections import defaultdict
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, precision_score, confusion_matrix
import seaborn
import matplotlib.pyplot as plt
from series import TimeSeries
from ohlc import OHLC
from config import Configuration

class TrainingData:
	ohlc_series: TimeSeries[OHLC]

	def __init__(self, symbol: str):
		path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, f"{symbol}.csv")
		self.ohlc_series = TimeSeries.read_ohlc_csv(path)

def get_rate_of_change(new_value: float | int, old_value: float | int):
	rate = float(new_value) / float(old_value) - 1.0
	# Limit the value to reduce the impact of outliers
	rate = min(rate, 1.0)
	return rate

def get_features(data: TrainingData, start: pd.Timestamp, end: pd.Timestamp, balanced_samples: bool, shuffle: bool) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
	assert start < end
	offsets = [
		0,
		1,
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
	max_offset = max(offsets)
	i = 0
	unbalanced_features: defaultdict[int, list[list[float]]] = defaultdict(list)
	for time in data.ohlc_series:
		if time >= end:
			break
		if i > max_offset and start <= time:
			records: list[OHLC] = data.ohlc_series.get(time, offsets=offsets)
			close_values = [ohlc.close for ohlc in records]
			today = close_values[0]
			yesterday = close_values[1]
			momentum_close_values = close_values[2:]
			momentum_rates = [get_rate_of_change(yesterday, close) for close in momentum_close_values]
			features = momentum_rates
			# Create a simple binary label for the most recent returns (i.e. comparing today and yesterday)
			label_rate = get_rate_of_change(today, yesterday)
			label = 1 if label_rate > 0 else 0
			unbalanced_features[label].append(features)
		i += 1

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
	return np_features, np_labels

def train(symbol: str, start: pd.Timestamp, split: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < split < end
	data = TrainingData(symbol)
	x_train, y_train = get_features(data, start, split, True, True)
	x_validation, y_validation = get_features(data, split, end, False, False)
	scaler = StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_validation = scaler.transform(x_validation)

	# Statistics
	precision_values = []
	roc_auc_values = []
	f1_scores = []

	heatmap_data = []

	# Iterate over hyperparameters
	num_leaves_values = [31, 60, 90, 120]
	num_iterations_values = [50, 100, 200, 300]
	inner_iterations = 50
	for num_leaves in num_leaves_values:
		heatmap_row = []
		for num_iterations in num_iterations_values:
			samples = []
			for _ in range(inner_iterations):
				params = {
					"verbosity": -1,
					"objective": "binary",
					"metric": ["binary_logloss", "auc"],
					"num_iterations": num_iterations,
					"num_leaves": num_leaves,
				}
				train_dataset = lgb.Dataset(x_train, label=y_train)
				validation_dataset = lgb.Dataset(x_validation, label=y_validation, reference=train_dataset)
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
			sample = median(samples)
			heatmap_row.append(sample)
		heatmap_data.append(heatmap_row)

	median_precision = median(precision_values)
	median_roc_auc = median(roc_auc_values)
	median_f1 = median(f1_scores)
	print(f"Median precision: {median_precision:.1%}")
	print(f"Median ROC-AUC: {median_roc_auc:.3f}")
	print(f"Median F1: {median_f1:.3f}")

	# Show heatmap of hyperparameters
	x_tick_labels = [str(x) for x in num_iterations_values]
	y_tick_labels = [str(x) for x in num_leaves_values]
	seaborn.heatmap(heatmap_data, xticklabels=x_tick_labels, yticklabels=y_tick_labels, cbar_kws={"label": "F1 Score"})
	plt.xlabel("Iterations")
	plt.ylabel("Leaves")
	plt.gca().invert_yaxis()
	plt.gcf().canvas.manager.set_window_title("Hyperparameters Heatmap")
	plt.show()

def binary_predictions(predictions: np.ndarray[np.float64]) -> np.ndarray[np.int8]:
	return (predictions > 0.5).astype(dtype=np.int8) # type: ignore

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