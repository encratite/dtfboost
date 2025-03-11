import os
from glob import glob

from config import Configuration
from ohlc import OHLC
from series import TimeSeries

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