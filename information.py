import os
import sys
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from config import Configuration
from series import TimeSeries
from technical import get_rate_of_change

def analyze(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
	path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, f"{symbol}.csv")
	ohlc_series = TimeSeries.read_ohlc_csv(path)
	time_range = [t for t in ohlc_series if start <= t < end]
	momentum_days = [
		2,
		3,
		5,
		10,
		25,
		50,
		100,
		150,
		200,
		250
	]
	future_days_values = [
		1,
		2,
		3,
		4,
		5,
		25,
		50,
		100,
		150,
		200,
		250
	]
	series_count = max(momentum_days)

	heatmap_data = []
	column_labels = [str(x) for x in momentum_days]
	row_labels = [str(x) for x in future_days_values]

	for future_days in future_days_values:
		future_returns_values = []
		momentum_values = defaultdict(list)
		for time in time_range:
			future_time = time + pd.Timedelta(days=future_days)
			future = ohlc_series.get(future_time)
			records = ohlc_series.get(time, count=series_count)
			today = records[0]
			future_returns = get_rate_of_change(future.close, today.close)
			future_returns_values.append(future_returns)
			for days in momentum_days:
				then = records[days - 1]
				momentum = get_rate_of_change(today.close, then.close)
				momentum_values[days].append(momentum)

		r_values = [spearmanr(future_returns_values, momentum_values[days]).statistic for days in momentum_days] # type: ignore
		heatmap_data.append(r_values)

	heatmap_data.reverse()
	row_labels.reverse()
	heatmap_df = pd.DataFrame(heatmap_data, columns=column_labels)
	plt.figure(figsize=(10, 8))
	ax = sns.heatmap(heatmap_df, xticklabels=column_labels, yticklabels=row_labels, annot=True)
	plt.title(f"{symbol} Momentum/Returns IC", pad=15)
	ax.set_xlabel("Momentum Offset (Days)", labelpad=10)
	ax.set_ylabel("Future Returns Offset (Days)", labelpad=10)
	plt.show()

def main() -> None:
	if len(sys.argv) != 4:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbol> <start date> <end date>")
		return
	symbol = sys.argv[1]
	start = pd.Timestamp(sys.argv[2])
	end = pd.Timestamp(sys.argv[3])
	analyze(symbol, start, end)

main()