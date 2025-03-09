from typing import Any
import os
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
import numpy.typing as npt
from series import TimeSeries
from ohlc import OHLC
from config import Configuration

def get_rate_of_change(new_value: float | int, old_value: float | int):
	rate = float(new_value) / float(old_value) - 1.0
	# Limit the value to reduce the impact of outliers
	rate = min(rate, 1.0)
	return rate

def get_features(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
	assert start < end
	path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, f"{symbol}.csv")
	ohlc_series = TimeSeries.read_ohlc_csv(path)
	offsets = [
		0,
		1,
		2,
		10,
		25,
		50,
		150,
		250
	]
	max_offset = max(offsets)
	i = 0
	unbalanced_features: defaultdict[int, list[list[float | int]]] = defaultdict(list)
	for time in ohlc_series:
		if time >= end:
			break
		if i > max_offset and start <= time:
			records: list[OHLC] = ohlc_series.get(time, offsets=offsets)
			close_values = list(map(lambda ohlc: ohlc.close, records))
			today = close_values[0]
			yesterday = close_values[1]
			momentum_close_values = close_values[2:]
			momentum_rates = [get_rate_of_change(yesterday, close) for close in momentum_close_values]
			features = momentum_rates
			# Create a simple binary label for the most recent returns (i.e. comparing today and yesterday)
			label_rate = get_rate_of_change(today, yesterday)
			label = 1 if label_rate > 0 else 0
			unbalanced_features[label].append(features)
		i += 1
	for label, label_features in unbalanced_features.items():
		print(f"Label {label}: {len(label_features)}")
		for f in label_features[0:5]:
			print(f)

def main() -> None:
	if len(sys.argv) != 4:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbol> <start ISO date> <end ISO date>")
		return
	symbol = sys.argv[1]
	start = pd.Timestamp(sys.argv[2])
	end = pd.Timestamp(sys.argv[3])
	get_features(symbol, start, end)

main()