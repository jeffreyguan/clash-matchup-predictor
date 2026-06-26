"""
Download one season of the s1m0n38/clash-royale-games dataset into data/kaggle/.

The dataset stores one CSV per day under a per-season folder named by its date
range, e.g. 20231002-20231106/20231002-20231106/YYYYMMDD.csv. Single-file
downloads arrive zipped, so each day is unzipped and the .zip removed.

Credentials are read from the repo .env (KAGGLE_API_TOKEN) and set before
importing kaggle, which authenticates on import.

Run from anywhere:  venv/bin/python data/fetch_kaggle.py
"""

import os
import zipfile
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent
DEST = DATA_DIR / "kaggle"
DATASET = "s1m0n38/clash-royale-games"

# Season 53. The folder is named by this exact range (also the inner subfolder).
SEASON_FOLDER = "20231002-20231106"
START = date(2023, 10, 2)
END = date(2023, 11, 6)  # inclusive


def load_token():
    env = DATA_DIR.parent / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("KAGGLE_API_TOKEN") and "=" in line:
            os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
            return
    raise SystemExit("KAGGLE_API_TOKEN not found in .env")


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main():
    load_token()
    # Import after the token is set: the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    DEST.mkdir(parents=True, exist_ok=True)

    got, missing = 0, 0
    for d in daterange(START, END):
        day = d.strftime("%Y%m%d")
        csv_path = DEST / f"{day}.csv"
        if csv_path.exists():
            got += 1
            print(f"  {day}: already present")
            continue

        remote = f"{SEASON_FOLDER}/{SEASON_FOLDER}/{day}.csv"
        try:
            api.dataset_download_file(DATASET, remote, path=str(DEST), quiet=True)
        except Exception as e:
            missing += 1
            print(f"  {day}: skipped ({type(e).__name__})")
            continue

        zip_path = DEST / f"{day}.csv.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(DEST)
            zip_path.unlink()
        got += 1
        print(f"  {day}: ok")

    print(f"\nDownloaded {got} day(s), {missing} missing -> {DEST}")


if __name__ == "__main__":
    main()
