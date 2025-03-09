import os
import pandas as pd
from config import Configuration
from series import TimeSeries

def test_time_series() -> None:
	path = os.path.join(Configuration.FRED_DIRECTORY, "AMTMNO.csv")
	series = TimeSeries.read_csv(path)
	times = [
		"1992-01-01",
		"1992-02-01",
		"1992-02-02",
		"1992-03-01",
		"2025-01-01",
		"2025-01-02"
	]
	for time_string in times:
		time = pd.Timestamp(time_string)
		try:
			value = series.get(time)
			print(f"series.get({time}): {value}")
		except Exception as e:
			print(f"series.get({time}): {e}")

def test_ohlc_time_series() -> None:
	path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, "ES.csv")
	series = TimeSeries.read_ohlc_csv(path)

	# Single values
	times = [
		"2008-01-01",
		"2008-01-02",
		"2008-01-05",
		"2026-03-06"
	]
	for time_string in times:
		time = pd.Timestamp(time_string)
		try:
			value = series.get(time)
			print(f"series.get({time}): {value}")
		except Exception as e:
			print(f"series.get({time}): {e}")

	# Multiple values
	time = pd.Timestamp("2008-01-10")
	counts = [
		5,
		10
	]
	for count in counts:
		print(f"series.get({time}):")
		try:
			values = series.get(time, count=count)
			i = 1
			for value in values:
				print(f"{i}. {value}")
				i += 1
		except Exception as e:
			print(e)

# test_time_series()
test_ohlc_time_series()