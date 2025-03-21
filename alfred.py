import os
import requests
import urllib.parse
from lxml import html

from config import Configuration

def download_observations(seid: str) -> None:
	url = f"https://alfred.stlouisfed.org/series/downloaddata?seid={urllib.parse.quote(seid)}"
	headers = {
		"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
	}
	session = requests.session()
	response = session.get(url, headers=headers)
	response.raise_for_status()
	tree = html.fromstring(response.text)
	start_date = tree.xpath("//input[@id='form_obs_start_date']/@value")[0]
	end_date = tree.xpath("//input[@id='form_obs_end_date']/@value")[0]
	vintage_dates = tree.xpath("//select[@id='form_selected_vintage_dates']/option/text()")
	data = [
		("form[units]", "lin"),
		("form[obs_start_date]", start_date),
		("form[form_obs_end_date]", end_date),
	]
	for vintage_date in vintage_dates:
		data.append(("form[selected_vintage_dates][]", vintage_date))
	data += [
		("form[entered_vintage_dates]", ""),
		("form[file_type]", 4),
		("form[file_format]", "csv"),
		("form[download_data]", "")
	]
	headers["Referer"] = url
	print(session.cookies)
	response = session.post(url, data=data, headers=headers)
	response.raise_for_status()
	print(response.headers["Content-Type"])

def save_zip_file(seid: str, response: requests.Response) -> None:
	if response.status_code == 200:
		path = os.path.join(Configuration.ALFRED_DIRECTORY, f"{seid}.zip")
		with open(path, "wb") as file:
			for chunk in response.iter_content(chunk_size=4096):
				file.write(chunk)
		print(f"Downloaded {path}")

download_observations("UNRATE")