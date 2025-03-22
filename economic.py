import pandas as pd

from data import TrainingData
from enums import FeatureCategory
from feature import Feature
from technical import get_rate_of_change

def get_barchart_features(time: pd.Timestamp, data: TrainingData) -> list[Feature]:
	# Big stock indexes and currency pairs that do not require any additional post-processing
	symbols = [
		("DAX Stock Index", "$DAX", FeatureCategory.ECONOMIC_INDEXES),
		# ("Dow Jones Industrial Average", "$DJX", FeatureCategory.ECONOMIC_INDEXES),
		# ("NASDAQ Composite", "$NASX", FeatureCategory.ECONOMIC_INDEXES),
		("Nikkei 225 Index", "$NKY", FeatureCategory.ECONOMIC_INDEXES),
		("S&P 500 Index", "$SPX", FeatureCategory.ECONOMIC_INDEXES),
		("Euro Stoxx 50 Index", "$STXE", FeatureCategory.ECONOMIC_INDEXES),
		("AUD/USD Exchange Rate", "^AUDUSD", FeatureCategory.ECONOMIC_CURRENCIES),
		("CAD/USD Exchange Rate", "^CADUSD", FeatureCategory.ECONOMIC_CURRENCIES),
		("CHF/USD Exchange Rate", "^CHFUSD", FeatureCategory.ECONOMIC_CURRENCIES),
		("EUR/USD Exchange Rate", "^EURUSD", FeatureCategory.ECONOMIC_CURRENCIES),
		("USD/JPY Exchange Rate", "^USDJPY", FeatureCategory.ECONOMIC_CURRENCIES),
		("Silver Spot", "^XAGUSD", FeatureCategory.ECONOMIC_RESOURCES),
		("Gold Spot", "^XAUUSD", FeatureCategory.ECONOMIC_RESOURCES),
	]
	features = []
	for feature_name, symbol, feature_category in symbols:
		ohlc_series = data.barchart_data[symbol]
		records = ohlc_series.get(time, count=2)
		feature_value = get_rate_of_change(records[0].close, records[1].close)
		feature = Feature(feature_name, feature_category, feature_value)
		features.append(feature)

		days_values = [20, 40, 120, 250]
		for days in days_values:
			then = time - pd.Timedelta(days=days)
			then_record = ohlc_series.get(then)
			feature_value = get_rate_of_change(records[0].close, then_record.close)
			feature = Feature(f"{feature_name} ({days} Days)", feature_category, feature_value)
			features.append(feature)
	return features