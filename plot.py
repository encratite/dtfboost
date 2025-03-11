import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from config import Configuration

def main() -> None:
	if len(sys.argv) != 2:
		print("Usage:")
		print(f"python {sys.argv[0]} <symbol of continuous contract>")
		return
	symbol = sys.argv[1]
	path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, f"{symbol}.csv")
	time_column = "time"
	df = pd.read_csv(path, parse_dates=[time_column])
	plt.figure(figsize=(12, 8))
	plt.gcf().canvas.manager.set_window_title(f"{symbol} Chart")
	plt.title(symbol)
	plt.plot(df[time_column], df["close"], label="Adjusted close")
	plt.plot(df[time_column], df["unadjusted_close"], label="Unadjusted close")
	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.legend()
	x_axis = plt.gca().xaxis
	x_axis.set_major_locator(mdates.YearLocator())
	x_axis.set_major_formatter(mdates.DateFormatter("%Y"))
	time_min = df[time_column].min()
	time_max = df[time_column].max()
	plt.xlim(time_min, time_max)
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.show()

main()