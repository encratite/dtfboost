import io
import os
import urllib.parse
from itertools import islice
from zipfile import ZipFile

import pandas as pd
import requests

from config import Configuration
from fred import FRED_CONFIG

def query_fred(path: str, parameters: dict[str, str]) -> requests.Response:
	url = f"https://api.stlouisfed.org/fred/{path}?"
	parameters["api_key"] = Configuration.FRED_API_KEY
	is_first = True
	for name, value in parameters.items():
		if is_first:
			is_first = False
		else:
			url += "&"
		url += f"{urllib.parse.quote(name)}={urllib.parse.quote(value)}"
	response = requests.get(url)
	response.raise_for_status()
	return response

def get_csv_from_zip(response: requests.Response) -> list[str]:
	if response.status_code == 200:
		zip_bytes = io.BytesIO(response.content)
		with ZipFile(zip_bytes, "r") as zf:
			for file_name in zf.namelist():
				with zf.open(file_name) as f:
					_name, extension = os.path.splitext(file_name)
					if extension == ".csv":
						content = f.read().decode()
						lines = content.splitlines()
						return lines
	raise Exception("Unable to find .csv file in .zip file")

def download_observations(seid: str) -> None:
	path = get_csv_path(seid)
	if os.path.exists(path):
		print(f"File already exists: {path}")
		return
	base_parameters = {
		"series_id": seid,
		"file_type": "json"
	}
	series = query_fred("series", base_parameters).json()
	frequency = series["seriess"][0]["frequency"]
	if "Daily" in frequency:
		observations_parameters = {
			"series_id": seid,
			"units": "lin",
			"file_type": "csv",
			"output_type": "1",
			"frequency": "d",
			"observation_start": "2000-01-01"
		}
		print(f"Downloading daily data for {seid}")
		observations = query_fred("series/observations", observations_parameters)
		csv_lines = get_csv_from_zip(observations)
	else:
		vintage_dates_dict: dict = query_fred("series/vintagedates", base_parameters).json()
		vintage_dates = vintage_dates_dict["vintage_dates"]
		vintage_dates = [x for x in vintage_dates if pd.Timestamp(x).year >= 2000]
		# FRED supposedly limits you to 1000 vintages at a time with output_type 4
		# In practice it seems to be a little over 600?
		iterator = iter(vintage_dates)
		vintage_chunks = iter(lambda: list(islice(iterator, 600)), [])
		csv_header = None
		csv_lines = []
		chunk_id = 1
		for chunk in vintage_chunks:
			observations_parameters = {
				"series_id": seid,
				"units": "lin",
				"file_type": "csv",
				"output_type": "4",
				"vintage_dates": ",".join(chunk)
			}
			print(f"Downloading chunk {chunk_id} for {seid}")
			observations = query_fred("series/observations", observations_parameters)
			content = get_csv_from_zip(observations)
			csv_header = content[0]
			csv_lines += content[1:]
			chunk_id += 1
		csv_lines = [csv_header] + csv_lines
	write_lines(seid, csv_lines)

def get_csv_path(seid: str) -> str:
	path = os.path.join(Configuration.FRED_DIRECTORY, f"{seid}.csv")
	return path

def write_lines(seid: str, csv_lines: list[str]) -> None:
	tokens = csv_lines[1].split(",")
	time = pd.Timestamp(tokens[0])
	if time.year > 2000:
		print(f"Warning: vintages of {seid} start at {time}")
	path = get_csv_path(seid)
	with open(path, "w+") as file:
		file.writelines(line + "\n" for line in csv_lines)
		print(f"Wrote {path}")

def download_fred_files() -> None:
	for _feature_name, seid, _post_processing, _feature_category, _feature_frequency, _upload_time in FRED_CONFIG:
		download_observations(seid)

download_fred_files()