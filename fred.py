import pandas as pd

from data import TrainingData
from enums import PostProcessing
from technical import get_rate_of_change
from feature import Feature
from fred_config import FRED_CONFIG

def get_fred_features(start: pd.Timestamp, time: pd.Timestamp, data: TrainingData) -> list[Feature]:
	features: list[Feature] = []
	for feature_name, symbol, post_processing, feature_category, feature_frequency, upload_time in FRED_CONFIG:
		series = data.fred_data[symbol]
		if start < series.start - pd.Timedelta(days=361):
			# Skip features that weren't available at the time
			# This is a hack to partially integrate series for which only more recent vintages have been made available
			continue
		# Questionable way to simulate the lag of FRED releases
		days_offset = 1 if "PM" in upload_time else 0
		time_with_offset = time
		if days_offset > 0:
			time_with_offset -= pd.Timedelta(days=days_offset)
		match post_processing:
			case PostProcessing.NOMINAL:
				feature_value = series.get(time_with_offset)
				feature = Feature(feature_name, feature_category, feature_value)
				features.append(feature)
			case PostProcessing.DIFFERENCE:
				feature_name = f"{feature_name} (Delta)"
				values = series.get(time_with_offset, count=2)
				feature_value = values[0] - values[1]
				feature = Feature(feature_name, feature_category, feature_value)
				features.append(feature)
			case PostProcessing.NOMINAL_AND_DIFFERENCE:
				values = series.get(time_with_offset, count=2)
				nominal_value = values[0]
				difference = values[0] - values[1]
				nominal_feature = Feature(feature_name, feature_category, nominal_value)
				difference_feature = Feature(f"{feature_name} (Delta)", feature_category, difference)
				features += [
					nominal_feature,
					difference_feature
				]
			case PostProcessing.RATE_OF_CHANGE:
				values = series.get(time_with_offset, count=2)
				feature_value = get_rate_of_change(values[0], values[1])
				feature = Feature(feature_name, feature_category, feature_value)
				features.append(feature)
				days_values = [30, 60, 90, 360]
				for days in days_values:
					then = time_with_offset - pd.Timedelta(days=days)
					value = series.get(then)
					feature_value = get_rate_of_change(values[0], value)
					feature = Feature(f"{feature_name} ({days} Days)", feature_category, feature_value)
					features.append(feature)
	return features