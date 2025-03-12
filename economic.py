import pandas as pd

from data import TrainingData
from fred import get_fred_features
from technical import get_rate_of_change

def get_economic_features(time: pd.Timestamp, data: TrainingData) -> tuple[list[str], list[float]]:
	yesterday = time - pd.Timedelta(days=1)
	fred_feature_names, fred_features = get_fred_features(yesterday, data)
	barchart_feature_names, barchart_features = get_barchart_features(yesterday, data)
	feature_names = fred_feature_names + barchart_feature_names
	features = fred_features + barchart_features
	return feature_names, features

def get_barchart_features(yesterday: pd.Timestamp, data: TrainingData) -> tuple[list[str], list[float]]:
	# Big stock indexes and currency pairs that do not require any additional post-processing
	symbols = [
		("DAX Stock Index", "$DAX"),
		("Dow Jones Industrial Average", "$DJX"),
		("NASDAQ Composite", "$NASX"),
		("Nikkei 225 Index", "$NKY"),
		("S&P 500 Index", "$SPX"),
		("Euro Stoxx 50 Index", "$STXE"),
		("AUD/USD Exchange Rate", "^AUDUSD"),
		("CAD/USD Exchange Rate", "^CADUSD"),
		("CHF/USD Exchange Rate", "^CHFUSD"),
		("EUR/USD Exchange Rate", "^EURUSD"),
		("USD/JPY Exchange Rate", "^USDJPY"),
		("Silver Spot", "^XAGUSD"),
		("Gold Spot", "^XAUUSD"),
	]
	feature_names = []
	features = []
	for feature_name, symbol in symbols:
		ohlc_series = data.barchart_data[symbol]
		records = ohlc_series.get(yesterday, count=2)
		rate_of_change = get_rate_of_change(records[0].close, records[1].close)
		feature_names.append(feature_name)
		features.append(rate_of_change)
	return feature_names, features