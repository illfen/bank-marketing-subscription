from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "bank_marketing.zip"
EXTRACTED_DIR = RAW_DIR / "bank"
FULL_CSV = EXTRACTED_DIR / "bank-full.csv"


def ensure_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not FULL_CSV.exists():
        if not ZIP_PATH.exists():
            urlretrieve(DATA_URL, ZIP_PATH)

        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(EXTRACTED_DIR)

        nested_zip = EXTRACTED_DIR / "bank.zip"
        if nested_zip.exists() and not FULL_CSV.exists():
            with zipfile.ZipFile(nested_zip) as archive:
                archive.extractall(EXTRACTED_DIR)

    if not FULL_CSV.exists():
        raise FileNotFoundError(f"dataset was not found after extraction: {FULL_CSV}")

    return FULL_CSV


def load_bank_marketing() -> pd.DataFrame:
    csv_path = ensure_dataset()
    return pd.read_csv(csv_path, sep=";")


def dataset_profile(frame: pd.DataFrame) -> dict[str, object]:
    target_counts = frame["y"].value_counts().to_dict()
    return {
        "n_samples": int(len(frame)),
        "n_features_with_target": int(frame.shape[1]),
        "positive_count": int(target_counts.get("yes", 0)),
        "negative_count": int(target_counts.get("no", 0)),
        "positive_rate": round(float((frame["y"] == "yes").mean()), 6),
        "columns": list(frame.columns),
    }
