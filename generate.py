import os
from collections import defaultdict
from typing import Any, cast

import pandas as pd

from config import Configuration
from globex import GlobexCode
from ohlc import OHLC

def generate_continuous_contract(symbol: str) -> None:
	input_path = os.path.join(Configuration.BARCHART_DIRECTORY, f"{symbol}.D1.csv")
	print(f"Processing {input_path}")
	df: pd.DataFrame = pd.read_csv(input_path, parse_dates=["time"])
	contracts_per_day: dict[pd.Timestamp, list[OHLC]] = defaultdict(list)
	contract_ranges: dict[GlobexCode, tuple[pd.Timestamp, pd.Timestamp]] = {}
	for row in df.itertuples():
		row = cast(Any, row)
		ohlc = OHLC(row)
		contracts_per_day[row.time].append(ohlc)
		if row.symbol in contract_ranges:
			first, last = contract_ranges[ohlc.globex_code]
			contract_ranges[ohlc.globex_code] = (min(row.time, first), max(row.time, last))
		else:
			contract_ranges[ohlc.globex_code] = (row.time, row.time)
	current_globex_code: GlobexCode | None = None
	ohlc_offsets: list[tuple[OHLC, float]] = []
	# Skip old records
	pairs = [(time, records) for time, records in contracts_per_day.items() if time >= Configuration.CUTOFF_DATE]
	# Select first Globex code by open interest
	_, first_records = next(iter(pairs))
	first_record = get_ohlc_by_open_interest(first_records)
	current_globex_code = first_record.globex_code
	# Calculate rollovers and keep track of the offsets
	for time, records in pairs:
		current_ohlc = next((x for x in records if x.globex_code == current_globex_code), None)
		if current_ohlc is None:
			raise Exception(f"Unable to find a record for current contract {current_globex_code} at {time.date()}")
		_, last_contract_day = contract_ranges[current_globex_code]

		def is_rollover_target(x: OHLC) -> bool:
			last_day = time == last_contract_day
			enough_open_interest = x.open_interest > current_ohlc.open_interest > 0
			return x.globex_code > current_globex_code and (last_day or enough_open_interest)

		filtered_records = [x for x in records if is_rollover_target(x)]
		offset = 0
		if len(filtered_records) > 0:
			new_ohlc = get_ohlc_by_open_interest(filtered_records)
			if new_ohlc.globex_code > current_globex_code:
				# Roll over into new contract
				offset = new_ohlc.close - current_ohlc.close
				# print(f"{time.date()} {current_globex_code} -> {new_ohlc.globex_code}: {offset:+.2f}")
				current_globex_code = new_ohlc.globex_code
				current_ohlc = new_ohlc
		ohlc_offsets.append((current_ohlc, offset))
	# Calculate global offset from differences between contracts, in reverse
	global_offset = 0
	for i in reversed(range(len(ohlc_offsets))):
		ohlc, offset = ohlc_offsets[i]
		ohlc_offsets[i] = ohlc, global_offset
		global_offset += offset
	# Generate DataFrame using the Panama canal method
	df_dict = defaultdict(list)
	precision = 2
	for ohlc, offset in ohlc_offsets:
		df_dict["time"].append(ohlc.time.date())
		df_dict["symbol"].append(repr(ohlc.globex_code))
		df_dict["open"].append(round(ohlc.open + offset, precision))
		df_dict["high"].append(round(ohlc.high + offset, precision))
		df_dict["low"].append(round(ohlc.low + offset, precision))
		df_dict["close"].append(round(ohlc.close + offset, precision))
		df_dict["unadjusted_close"].append(ohlc.close)
		df_dict["volume"].append(ohlc.volume)
		df_dict["open_interest"].append(ohlc.open_interest)
	df = pd.DataFrame(df_dict)
	output_path = os.path.join(Configuration.CONTINUOUS_CONTRACT_DIRECTORY, f"{symbol}.csv")
	df.to_csv(output_path, index=False)

def get_ohlc_by_open_interest(records: list[OHLC]) -> OHLC:
	return max(records, key=lambda x: x.open_interest)

def main() -> None:
	symbols = [
		"ES",
		"GC",
		"CL",
		"NG",
		"ZC",
		"ZN",
		"ZT",
		"VI",
	]

	for symbol in symbols:
		generate_continuous_contract(symbol)

main()