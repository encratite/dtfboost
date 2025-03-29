import os
import re
from glob import glob

import pandas as pd

from config import Configuration
from enums import FeatureFrequency
from fred_config import FRED_CONFIG
from ohlc import OHLC
from series import TimeSeries

class TrainingData:
	ohlc_series: TimeSeries[OHLC]
	barchart_data: dict[str, TimeSeries[OHLC]]
	fred_data: dict[str, TimeSeries[float]]

	def __init__(self, symbol: str):
		continuous_contract_path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, f"{symbol}.csv")
		self.ohlc_series = TimeSeries.read_ohlc_csv(continuous_contract_path)
		self._load_barchart_data()
		self._load_fred_data()

	def _load_barchart_data(self):
		# Load Barchart index/currency data
		path = os.path.join(Configuration.BARCHART_DIRECTORY, "*.csv")
		paths = glob(path)
		pattern = re.compile(r"^([$^].+?)\.D1\.csv$")
		self.barchart_data = {}
		for path in paths:
			base_name = os.path.basename(path)
			match = pattern.match(base_name)
			if match is not None:
				symbol = match[1]
				self.barchart_data[symbol] = TimeSeries.read_ohlc_csv(path)

	def _load_fred_data(self):
		# Load FRED economic data
		self.fred_data = {}
		for _feature_name, seid, _post_processing, _feature_category, _feature_frequency, _upload_time in FRED_CONFIG:
			path = os.path.join(Configuration.FRED_DIRECTORY, f"{seid}.csv")
			is_daily = _feature_frequency == FeatureFrequency.DAILY
			self.fred_data[seid] = TimeSeries.read_csv(path, is_daily)

class RegressionDataset:
	start: pd.Timestamp
	split: pd.Timestamp
	end: pd.Timestamp
	x_training: list[list[float]]
	y_training: list[float]
	x_validation: list[list[float]]
	y_validation: list[float]
	training_times: list[pd.Timestamp]
	validation_times: list[pd.Timestamp]
	delta_validation: list[float]

	def __init__(
		self,
		start: pd.Timestamp,
		split: pd.Timestamp,
		end: pd.Timestamp,
		x_training: list[list[float]],
		y_training: list[float],
		x_validation: list[list[float]],
		y_validation: list[float],
		training_times: list[pd.Timestamp],
		validation_times: list[pd.Timestamp],
		deltas_validation: list[float]
	):
		self.start = start
		self.split = split
		self.end = end
		self.x_training = x_training
		self.y_training = y_training
		self.x_validation = x_validation
		self.y_validation = y_validation
		self.training_times = training_times
		self.validation_times = validation_times
		self.delta_validation = deltas_validation