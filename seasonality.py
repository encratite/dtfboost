import calendar
from collections import defaultdict

import pandas as pd

def add_seasonality_features(time: pd.Timestamp, features: defaultdict[str, list[float]]):
	# Binary dummy variables
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