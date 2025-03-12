import pandas as pd

from data import TrainingData
from fred import get_fred_features
from technical import get_rate_of_change
from enums import FeatureCategory
from feature import Feature

def get_economic_features(time: pd.Timestamp, data: TrainingData) -> list[Feature]:
	yesterday = time - pd.Timedelta(days=1)
	features = get_fred_features(yesterday, data)
	features += get_barchart_features(yesterday, data)
	return features

def get_barchart_features(yesterday: pd.Timestamp, data: TrainingData) -> list[Feature]:
	# Big stock indexes and currency pairs that do not require any additional post-processing
	symbols = [
		("DAX Stock Index", "$DAX", FeatureCategory.ECONOMIC_INDEXES),
		("Dow Jones Industrial Average", "$DJX", FeatureCategory.ECONOMIC_INDEXES),
		("NASDAQ Composite", "$NASX", FeatureCategory.ECONOMIC_INDEXES),
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
		records = ohlc_series.get(yesterday, count=2)
		feature_value = get_rate_of_change(records[0].close, records[1].close)
		feature = Feature(feature_name, feature_category, feature_value)
		features.append(feature)
	return features