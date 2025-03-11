from typing import Final
import os
from pathlib import Path
from glob import glob
from enum import Enum
from itertools import islice, product
import sys
import random
from statistics import stdev, mean
from collections import defaultdict
import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from tqdm import tqdm
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
	# Delta: f(t) - f(t - 1)
	DIFFERENCE: Final[int] = 1
	# Generate two features, the nominal value f(t) and the delta f(t) - f(t - 1)
	NOMINAL_AND_DIFFERENCE: Final[int] = 2
	# Calculate f(t) / f(t - 1) - 1 for the most recent data point
	RATE_OF_CHANGE: Final[int] = 3

def get_rate_of_change(new_value: float | int, old_value: float | int):
	rate = float(new_value) / float(old_value) - 1.0
	# Limit the value to reduce the impact of outliers
	rate = min(rate, 1.0)
	return rate

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

def get_technical_features(time: pd.Timestamp, days_since_high_map: dict[pd.Timestamp, int], data: TrainingData) -> tuple[list[str], list[float], int]:
	# Offset 0 for the current day, offset 1 for yesterday, to calculate the binary label p(t) / p(t - 1) > 1
	# Offset 1 also serves as a reference for momentum features
	offsets = [0, 1]
	# Number of days to look into the past to calculate relative returns, i.e. p(t - 1) / p(t - n) - 1
	momentum_offsets = [
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
	moving_averages = [
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
		5,
		10,
		20,
		40,
		60,
		120
	]
	technical_feature_names = [f"Momentum ({x} Days)" for x in momentum_offsets]
	technical_feature_names += [f"Price Minus Moving Average ({x} Days)" for x in moving_averages]
	offsets += momentum_offsets
	# Technically days_since_x should be part of this calculation
	ohlc_count = max(max(offsets), max(moving_averages)) + 1
	records: list[OHLC] = data.ohlc_series.get(time, count=ohlc_count)
	today = records[0].close
	# Truncate records to prevent any further access to data from the future
	records = records[1:]
	close_values = [ohlc.close for ohlc in records]
	yesterday = close_values[0]
	momentum_close_values = [close_values[i - 1] for i in momentum_offsets]
	technical_features = [get_rate_of_change(yesterday, close) for close in momentum_close_values]
	for moving_average_days in moving_averages:
		moving_average_values = close_values[:moving_average_days]
		assert len(moving_average_values) == moving_average_days
		# Calculate price minus simple moving average
		moving_average = sum(moving_average_values) / moving_average_days
		moving_average_feature = yesterday - moving_average
		technical_features.append(moving_average_feature)
	days_since_x_feature_names, days_since_x_features = get_days_since_x_features(time, records, days_since_high_map)
	technical_feature_names += days_since_x_feature_names
	technical_features += days_since_x_features
	# Daily volatility
	technical_feature_names += [f"Volatility ({x} Days)" for x in volatility_days]
	technical_features += [get_daily_volatility(close_values, x) for x in volatility_days]
	# Create a simple binary label for the most recent returns (i.e. comparing today and yesterday)
	label_rate = get_rate_of_change(today, yesterday)
	label = 1 if label_rate > 0 else 0
	return technical_feature_names, technical_features, label

def get_days_since_x_features(time: pd.Timestamp, records: list[OHLC], days_since_high_map: dict[pd.Timestamp, int]) -> tuple[list[str], list[float]]:
	days_since_x = [
		20,
		40,
		60,
		120
	]
	technical_feature_names: list[str] = []
	technical_features: list[float] = []
	high_values = [ohlc.high for ohlc in records]
	low_values = [ohlc.low for ohlc in records]
	# Days since last all-time high
	technical_feature_names.append("Days Since Last All-Time High")
	days_since_all_time_high = days_since_high_map[time]
	technical_features.append(days_since_all_time_high)
	# Days since last high within the last n days
	technical_feature_names += [f"Days Since High ({x} Days)" for x in days_since_x]
	for days in days_since_x:
		values = high_values[:days]
		index = values.index(max(values))
		technical_features.append(index)
	# Days since last low within the last n days
	technical_feature_names += [f"Days Since Low ({x} Days)" for x in days_since_x]
	for days in days_since_x:
		values = low_values[:days]
		index = values.index(min(values))
		technical_features.append(index)
	return technical_feature_names, technical_features

def get_days_since_high_map(data: TrainingData) -> dict[pd.Timestamp, int]:
	high_time = None
	high = None
	days_since_high_map = {}
	for ohlc in data.ohlc_series.values():
		if high is not None:
			days_since_high_map[ohlc.time] = (ohlc.time - high_time).days
		if high is None or ohlc.high > high:
			high_time = ohlc.time
			high = ohlc.high
	return days_since_high_map

def get_daily_volatility(close_values: list[float], days: int) -> float:
	values = close_values[:days]
	returns = [get_rate_of_change(a, b) for a, b in zip(values, values[1:])]
	volatility = stdev(returns)
	return volatility

def get_economic_features(time: pd.Timestamp, data: TrainingData) -> tuple[list[str], list[float]]:
	yesterday = time - pd.Timedelta(days=1)
	# FRED economic data
	fred_config = [
		# Initial unemployment claims (ICSA), nominal value, weekly
		("Initial Unemployment Claims", "ICSA", PostProcessing.RATE_OF_CHANGE),
		# Unemployment Rate (UNRATE), percentage, monthly
		("Unemployment Rate", "UNRATE", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis (DGS10), percentage, daily
		("10-Year T-Note Yield", "DGS10", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity (T10Y3M), percentage, daily
		("10-Year T-Note Minus 3-Month T-Bill Yield", "T10Y3M", PostProcessing.NOMINAL),
		# 30-Year Fixed Rate Mortgage Average in the United States (MORTGAGE30US), percentage, weekly
		("30-Year Fixed Rate Mortgage", "MORTGAGE30US", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# 15-Year Fixed Rate Mortgage Average in the United States (MORTGAGE15US), percentage, weekly
		("15-Year Fixed Rate Mortgage Average", "MORTGAGE15US", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations (RRPONTSYD), nominal, daily
		("Overnight Reverse Repurchase Agreements", "RRPONTSYD", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Federal Funds Effective Rate (FEDFUNDS), percentage, monthly
		("Federal Funds Effective Rate", "FEDFUNDS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Federal Funds Effective Rate (DFF)
		("Federal Funds Effective Rate (Daily)", "DFF", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# M2 money supply (M2SL), nominal, monthly
		("M2 Supply", "M2SL", PostProcessing.RATE_OF_CHANGE),
		# Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL), nominal, weekly
		("Consumer Price Index", "CPIAUCSL", PostProcessing.RATE_OF_CHANGE),
		# Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level (WALCL), nominal, weekly
		("Total Assets", "WALCL", PostProcessing.RATE_OF_CHANGE),
		# Real Gross Domestic Product (GDPC1), nominal, quarterly
		("Real Gross Domestic Product", "GDPC1", PostProcessing.RATE_OF_CHANGE),
		# All Employees, Total Nonfarm (PAYEMS), nominal, monthly
		("All Employees", "PAYEMS", PostProcessing.RATE_OF_CHANGE),
		# Job Openings: Total Nonfarm (JTSJOL), nominal, monthly
		("Job Openings", "JTSJOL", PostProcessing.RATE_OF_CHANGE),
		# Producer Price Index by Commodity: Final Demand: Finished Goods Less Foods and Energy (WPSFD4131), nominal, monthly
		("Finished Goods Less Foods and Energy", "WPSFD4131", PostProcessing.RATE_OF_CHANGE),
		# Producer Price Index by Commodity: Final Demand: Finished Goods (WPSFD49207), nominal, monthly
		("Finished Goods", "WPSFD49207", PostProcessing.RATE_OF_CHANGE),
		# Advance Retail Sales: Retail Trade and Food Services (RSAFS), nominal, monthly
		("Retail Trade and Food Services", "RSAFS", PostProcessing.RATE_OF_CHANGE),
		# Imports of Goods and Services (IMPGS), nominal, quarterly
		("Imports of Goods and Services", "IMPGS", PostProcessing.RATE_OF_CHANGE),
		# Exports of Goods and Services (EXPGS), nominal, quarterly
		("Exports of Goods and Services", "EXPGS", PostProcessing.RATE_OF_CHANGE),
		# Average Weekly Earnings of All Employees, Total Private (CES0500000011), nominal, monthly
		("Average Weekly Earnings", "CES0500000011", PostProcessing.RATE_OF_CHANGE),
		# Nonfarm Business Sector: Labor Productivity (Output per Hour) for All Workers (PRS85006092), percentage, quarterly
		("Labor Productivity", "PRS85006092", PostProcessing.NOMINAL),
		# Nonfarm Business Sector: Unit Labor Costs for All Workers (PRS85006112), percentage, quarterly
		("Unit Labor Costs", "PRS85006112", PostProcessing.NOMINAL),
		# Manufacturers' New Orders: Durable Goods (DGORDER), nominal, monthly
		("Durable Goods New Orders", "DGORDER", PostProcessing.RATE_OF_CHANGE),
		# Manufacturers' New Orders: Total Manufacturing (AMTMNO), nominal, monthly
		("Total Manufacturing New Orders", "AMTMNO", PostProcessing.RATE_OF_CHANGE),
		# New One Family Houses Sold: United States (HSN1F), nominal, monthly
		("New One Family Houses Sold", "HSN1F", PostProcessing.RATE_OF_CHANGE),
		# New Privately-Owned Housing Units Started: Total Units (HOUST), nominal, monthly
		("New Privately-Owned Housing Units Started", "HOUST", PostProcessing.RATE_OF_CHANGE),
		# Industrial Production: Total Index (INDPRO), nominal, monthly
		("Industrial Production - Total Index", "INDPRO", PostProcessing.RATE_OF_CHANGE),
		# Personal Income (PI), nominal, monthly
		("Personal Income", "PI", PostProcessing.RATE_OF_CHANGE),
		# Personal Consumption Expenditures (PCE), nominal, monthly
		("Personal Consumption Expenditures", "PCE", PostProcessing.RATE_OF_CHANGE),
		# Trade Balance: Goods and Services, Balance of Payments Basis (BOPGSTB), nominal, monthly
		("Trade Balance - Goods and Services", "BOPGSTB", PostProcessing.RATE_OF_CHANGE),
		# University of Michigan: Consumer Sentiment (UMCSENT), nominal, monthly
		("University of Michigan Consumer Sentiment", "UMCSENT", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma (DCOILWTICO), nominal, daily
		("Crude Oil - West Texas Intermediate", "DCOILWTICO", PostProcessing.RATE_OF_CHANGE),
		# Crude Oil Prices: Brent - Europe (DCOILBRENTEU), nominal, daily
		("Crude Oil - Brent", "DCOILBRENTEU", PostProcessing.RATE_OF_CHANGE),
		# US Regular All Formulations Gas Price (GASREGW), nominal, weekly
		("Gas Price", "GASREGW", PostProcessing.RATE_OF_CHANGE),
		# Henry Hub Natural Gas Spot Price (DHHNGSP), nominal, daily
		("Henry Hub Natural Gas Spot Price", "DHHNGSP", PostProcessing.RATE_OF_CHANGE),
		# Global price of LNG, Asia (PNGASJPUSDM), nominal, monthly
		("Global Price of LNG - Asia", "PNGASJPUSDM", PostProcessing.RATE_OF_CHANGE),
		# Average Price: Electricity per Kilowatt-Hour in U.S. City Average (APU000072610), nominal, monthly
		("Price of Electricity", "APU000072610", PostProcessing.DIFFERENCE),
		# Global price of Copper (PCOPPUSDM), nominal, monthly
		("Global Price of Copper", "PCOPPUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Energy index (PNRGINDEXM), nominal, monthly
		("Global Price of Energy Index", "PNRGINDEXM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Natural gas, EU (PNGASEUUSDM), nominal, monthly
		("Global Price of Natural Gas - EU", "PNGASEUUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Aluminum (PALUMUSDM), nominal, monthly
		("Global Price of Aluminum", "PALUMUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Corn (PMAIZMTUSDM), nominal, monthly
		("Global Price of Corn", "PMAIZMTUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Soybeans (PSOYBUSDM), nominal, monthly
		("Global Price of Soybeans", "PSOYBUSDM", PostProcessing.RATE_OF_CHANGE),
		# Global price of Food index (PFOODINDEXM), nominal, monthly
		("Global Price of Food index", "PFOODINDEXM", PostProcessing.RATE_OF_CHANGE),
		# CBOE Volatility Index: VIX (VIXCLS), nominal, daily
		("CBOE Volatility Index", "VIXCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE S&P 500 3-Month Volatility Index (VXVCLS), nominal, daily
		("CBOE S&P 500 3-Month Volatility Index", "VXVCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE Gold ETF Volatility Index (GVZCLS), nominal, daily
		("CBOE Gold ETF Volatility Index", "GVZCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE Crude Oil ETF Volatility Index (OVXCLS), nominal, daily
		("CBOE Crude Oil ETF Volatility Index", "OVXCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE NASDAQ 100 Volatility Index (VXNCLS), nominal, daily
		("CBOE NASDAQ 100 Volatility Index", "VXNCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
		# CBOE DJIA Volatility Index (VXDCLS), nominal, daily
		("CBOE DJIA Volatility Index", "VXDCLS", PostProcessing.NOMINAL_AND_DIFFERENCE),
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
			case PostProcessing.DIFFERENCE:
				values = data.fred_data[symbol].get(yesterday, count=2)
				difference = values[0] - values[1]
				features.append(difference)
				feature_names.append(f"{name} (Delta)")
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
	x_train, y_train = get_features(start, split, data)
	x_validation, y_validation = get_features(split, end, data)

	# Scale features to improve model performance (or so they say)
	if Configuration.ENABLE_SCALING:
		scaler = StandardScaler()
		scaler.fit(x_train)
		x_train_scaled = scaler.transform(x_train)
		x_validation_scaled = scaler.transform(x_validation)
		x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
		x_validation = pd.DataFrame(x_validation_scaled, columns=x_validation.columns)

	# Statistics
	precision_values = []
	roc_auc_values = []
	f1_scores = []
	heatmap_data = []
	max_precision = None
	max_f1 = None
	best_model_parameters = None
	best_model = None
	best_model_precision = None
	best_model_f1 = None

	# Iterate over hyperparameters
	num_leaves_values = [20, 30, 40, 50]
	min_data_in_leaf_values = [10, 15, 20, 25, 30]
	# max_depth_values = [-1, 10, 20, 50]
	max_depth_values = [-1]
	num_iterations_values = [75, 100, 150, 200, 250]
	parameter_f1: defaultdict[str, defaultdict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
	combinations = list(product(num_leaves_values, min_data_in_leaf_values, max_depth_values, num_iterations_values))
	for num_leaves, min_data_in_leaf, max_depth, num_iterations in tqdm(combinations, desc="Evaluating hyperparameters", colour="green"):
		params = {
			"objective": "binary",
			# "metric": ["binary_logloss", "auc"],
			"metric": "binary_logloss",
			# "metric": "average_precision",
			"verbosity": -1,
			"num_leaves": num_leaves,
			"min_data_in_leaf": min_data_in_leaf,
			"max_depth": max_depth,
			"num_iterations": num_iterations,
		}
		train_dataset = lgb.Dataset(x_train, label=y_train)
		validation_dataset = lgb.Dataset(x_validation, label=y_validation, reference=train_dataset)
		model = lgb.train(params, train_dataset, valid_sets=[validation_dataset])

		model_parameters = {
			"num_leaves": num_leaves,
			"min_data_in_leaf": min_data_in_leaf,
			"max_depth": max_depth,
			"num_iterations": num_iterations,
		}

		predictions = model.predict(x_validation)
		predictions = binary_predictions(predictions)
		precision = precision_score(y_validation, predictions)
		precision_values.append(precision)
		roc_auc = roc_auc_score(y_validation, predictions)
		roc_auc_values.append(roc_auc)
		f1 = f1_score(y_validation, predictions)
		f1_scores.append(f1)

		if best_model is None or f1 > best_model_f1:
			best_model_precision = precision
			best_model_f1 = f1
			best_model_parameters = model_parameters
			best_model = model
		max_f1 = max(f1, max_f1 if max_f1 is not None else f1)
		max_precision = max(precision, max_precision if max_precision is not None else precision)
		for name, value in model_parameters.items():
			parameter_f1[name][value].append(f1)

	print(f"Number of samples in training data: {x_train.shape[0]}")
	print(f"Number of samples in validation data: {x_validation.shape[0]}")
	print(f"Number of features: {x_train.shape[1]}")

	mean_precision = mean(precision_values)
	label_distribution = get_label_distribution(y_validation)
	mean_roc_auc = mean(roc_auc_values)
	mean_f1 = mean(f1_scores)
	print(f"Mean precision: {mean_precision:.1%}")
	print(f"Positive labels: {label_distribution[1]:.1%}")
	print(f"Negative labels: {label_distribution[0]:.1%}")
	print(f"Mean ROC-AUC: {mean_roc_auc:.3f}")
	print(f"Mean F1 score: {mean_f1:.3f}")
	print(f"Maximum precision: {max_precision:.1%}")
	print(f"Maximum F1 score: {max_f1:.3f}")

	print("Mean F1 scores of hyperparameters:")
	for name, values in parameter_f1.items():
		print(f"\t{name}:")
		for value, f1_values in values.items():
			print(f"\t\t{value}: {mean(f1_values):.3f}")

	print(f"Best model precision: {best_model_precision:.1%}")
	print(f"Best model F1 score: {best_model_f1:.3f}")
	print("Best hyperparameters:")
	print(best_model_parameters)

	# Render SHAP summary
	explainer = shap.TreeExplainer(best_model)
	x_all = pd.concat([x_train, x_validation], ignore_index=True)
	shap_values = explainer(x_all)
	shap.summary_plot(shap_values, x_all, max_display=30, show=False, plot_size=(12, 12))
	save_plot(symbol, "SHAP Summary")

	# Evaluate certain features only
	selected_features = [
		"Momentum (2 Days)",
		"Crude Oil - West Texas Intermediate (Rate of Change)",
		"Henry Hub Natural Gas Spot Price (Rate of Change)",
		"Crude Oil - Brent (Rate of Change)",
		"Global Price of Energy Index (Rate of Change)",
		"Global Price of Natural Gas - EU (Rate of Change)",
		"Price of Electricity (Delta)",
		"CBOE Crude Oil ETF Volatility Index",
		"CBOE Crude Oil ETF Volatility Index (Delta)"
	]
	shap.summary_plot(shap_values[:, selected_features], x_all[selected_features], show=False, plot_size=(12, 6))
	save_plot(symbol, "SHAP Summary (selection)")

	# Mean feature importance values
	df = pd.DataFrame({
		"Feature": x_all.columns,
		"Mean Absolute SHAP": np.mean(np.abs(shap_values.values), axis=0)
	})
	df = df.sort_values(by="Mean Absolute SHAP", ascending=False)
	csv_path = os.path.join(Configuration.PLOT_DIRECTORY, symbol, f"Feature Importance.csv")
	df.to_csv(csv_path, index=False)

	# SHAP dependence plots
	if Configuration.GENERATE_DEPENDENCE_PLOTS:
		# Workaround for too many PyPlot figures being created for some reason
		# The plt.close() in save_plot doesn't seem to do the trick
		plt.rcParams["figure.max_open_warning"] = 1000
		for feature in tqdm(range(x_all.shape[1]), desc="Generating dependence plots", colour="green"):
			plt.figure(figsize=(14, 8))
			shap.dependence_plot(feature, shap_values.values, x_all, show=False)
			save_plot(symbol, f"Dependence", x_all.columns[feature])

def save_plot(*tokens: str) -> None:
	directories = tokens[:-1]
	name = tokens[-1]
	directory = os.path.join(Configuration.PLOT_DIRECTORY, *directories)
	path = Path(directory)
	path.mkdir(parents=True, exist_ok=True)
	plot_path = os.path.join(directory, f"{name}.png")
	plt.savefig(plot_path)
	plt.close()

def binary_predictions(predictions: npt.NDArray[np.float64]) -> npt.NDArray[np.int8]:
	return (predictions > 0.5).astype(dtype=np.int8)

def get_label_distribution(y_validation: pd.DataFrame) -> defaultdict[int, float]:
	output = defaultdict(float)
	for label in y_validation.iloc[:, 0]:
		output[label] += 1.0
	for label in output:
		output[label] /= y_validation.shape[0]
	return output

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