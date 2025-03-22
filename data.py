import os
import re
from glob import glob

from config import Configuration
from ohlc import OHLC
from series import TimeSeries
from fred_config import FRED_CONFIG
from enums import FeatureFrequency

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