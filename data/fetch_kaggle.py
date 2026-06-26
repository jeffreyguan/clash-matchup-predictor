"""
Download the s1m0n38/clash-royale-games dataset into data/kaggle/.

Uses the Kaggle Python API. Credentials are read from the repo .env
(KAGGLE_API_TOKEN) and set before importing kaggle, which authenticates
on import. Run from anywhere:  venv/bin/python data/fetch_kaggle.py
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent
DEST = DATA_DIR / "kaggle"
DATASET = "s1m0n38/clash-royale-games"

# Set to a single file path within the dataset to grab just that one,
# or leave None to download the whole dataset.
SINGLE_FILE = None


def load_token():
    env = DATA_DIR.parent / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("KAGGLE_API_TOKEN") and "=" in line:
            os.environ["KAGGLE_API_TOKEN"] = line.split("=", 1)[1].strip().strip("\"'")
            return
    raise SystemExit("KAGGLE_API_TOKEN not found in .env")


def main():
    load_token()
    # Import after the token is set: the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    DEST.mkdir(parents=True, exist_ok=True)
    if SINGLE_FILE:
        api.dataset_download_file(DATASET, SINGLE_FILE, path=str(DEST))
        print(f"Downloaded {SINGLE_FILE} -> {DEST}")
    else:
        api.dataset_download_files(DATASET, path=str(DEST), unzip=True)
        print(f"Downloaded full dataset -> {DEST}")


if __name__ == "__main__":
    main()
