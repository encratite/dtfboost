from __future__ import annotations
import re
from functools import total_ordering

@total_ordering
class GlobexCode:
	symbol: str
	root: str
	month: str
	year: int

	def __init__(self, symbol):
		pattern = re.compile("^([A-Z0-9]{2,})([FGHJKMNQUVXZ])([0-9]{2})$")
		match = pattern.match(symbol)
		if match is None:
			raise Exception("Invalid Globex code")
		self.symbol = symbol
		self.root = match[1]
		self.month = match[2]
		year = int(match[3])
		if year < 70:
			year += 1900
		else:
			year += 2000
		self.year = year

	def _get_tuple(self) -> tuple[str, int, str]:
		return self.root, self.year, self.month

	def __eq__(self, other: GlobexCode) -> bool:
		return self._get_tuple() == other._get_tuple()

	def __lt__(self, other: GlobexCode) -> bool:
		return self._get_tuple() < other._get_tuple() # type: ignore

	def __hash__(self) -> int:
		globex_tuple = self._get_tuple()
		output = hash(globex_tuple)
		return output

	def __repr__(self) -> str:
		return self.symbol