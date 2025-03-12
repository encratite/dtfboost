from typing import Any

import pandas as pd

from globex import GlobexCode

class OHLC:
	globex_code: GlobexCode | None
	time: pd.Timestamp
	open: float
	high: float
	low: float
	close: float
	volume: int
	open_interest: int | None

	def __init__(self, row: Any):
		if GlobexCode.is_globex_code(row.symbol):
			self.globex_code = GlobexCode(row.symbol)
		else:
			self.globex_code = None
		self.time = pd.Timestamp(row.time)
		self.open = row.open
		self.high = row.high
		self.low = row.low
		self.close = row.close
		self.volume = row.volume
		if hasattr(row, "open_interest"):
			self.open_interest = row.open_interest
		else:
			self.open_interest = None

	def __repr__(self):
		return f"[{self.globex_code}] {self.time} {self.close}"