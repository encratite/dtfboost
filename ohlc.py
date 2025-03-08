from typing import Any
import pandas as pd
from globex import GlobexCode

class OHLC:
	globex_code: GlobexCode
	time: pd.Timestamp
	open: float
	high: float
	low: float
	close: float
	volume: int
	open_interest: int

	def __init__(self, row: Any):
		self.globex_code = GlobexCode(row.symbol)
		self.time = row.time
		self.open = row.open
		self.high = row.high
		self.low = row.low
		self.close = row.close
		self.volume	= row.volume
		self.open_interest = row.open_interest

	def __repr__(self):
		return f"[{self.globex_code}] {self.time} {self.close}"