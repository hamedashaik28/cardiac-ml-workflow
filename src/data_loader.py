"""Data loading and validation module for the UCI Cleveland Heart Disease dataset.

This module handles:
1. Downloading the raw dataset from the official UCI repository with local caching.
2. Graceful fallback when network access is unavailable, instructing manual placement.
3. Parsing missing values ('?' -> NaN).
4. Converting the multi-class angiographic diagnosis (0-4) into a binary target (0 vs 1).
5. Ensuring target or target-derived columns are never present in the feature matrix.
"""

from pathlib import Path
from typing import Tuple, Optional
import urllib.request
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Official UCI Cleveland Heart Disease dataset URL
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Standard 13 clinical predictors in exact Cleveland sequence
FEATURE_NAMES = [
    "age",       # Age in years
    "sex",       # Sex (1 = male; 0 = female)
    "cp",        # Chest pain type (1, 2, 3, 4)
    "trestbps",  # Resting blood pressure in mm Hg
    "chol",      # Serum cholesterol in mg/dl
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",   # Resting electrocardiographic results (0, 1, 2)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise induced angina (1 = yes; 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # Slope of peak exercise ST segment (1, 2, 3)
    "ca",        # Number of major vessels (0-3) colored by fluoroscopy
    "thal"       # Thallium scan status (3 = normal; 6 = fixed defect; 7 = reversible defect)
]

RAW_COLUMNS = FEATURE_NAMES + ["num"]

# Human-readable metadata dictionary for UI and documentation
CLINICAL_DICTIONARY = {
    "age": {
        "label": "Age",
        "description": "Patient age in years",
        "type": "numerical",
        "unit": "years"
    },
    "sex": {
        "label": "Sex",
        "description": "Biological sex at birth",
        "type": "categorical",
        "categories": {0: "Female", 1: "Male"}
    },
    "cp": {
        "label": "Chest Pain Type",
        "description": "Clinical category of chest discomfort",
        "type": "categorical",
        "categories": {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-Anginal Pain",
            4: "Asymptomatic"
        }
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "description": "Resting blood pressure upon hospital admission",
        "type": "numerical",
        "unit": "mm Hg"
    },
    "chol": {
        "label": "Serum Cholesterol",
        "description": "Serum cholesterol level",
        "type": "numerical",
        "unit": "mg/dl"
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "description": "Elevated fasting blood sugar indicator",
        "type": "categorical",
        "categories": {0: "<= 120 mg/dl (Normal)", 1: "> 120 mg/dl (Elevated)"}
    },
    "restecg": {
        "label": "Resting Electrocardiogram (ECG)",
        "description": "Resting ECG findings",
        "type": "categorical",
        "categories": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy (LVH)"
        }
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "description": "Peak heart rate during treadmill exercise testing",
        "type": "numerical",
        "unit": "beats/min"
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "description": "Angina provoked by exercise stress test",
        "type": "categorical",
        "categories": {0: "No", 1: "Yes"}
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "description": "ST depression induced by exercise relative to resting state",
        "type": "numerical",
        "unit": "mm"
    },
    "slope": {
        "label": "ST Segment Slope",
        "description": "Slope of the peak exercise ST segment",
        "type": "categorical",
        "categories": {
            1: "Upsloping",
            2: "Flat",
            3: "Downsloping"
        }
    },
    "ca": {
        "label": "Fluoroscopy Colored Vessels (ca)",
        "description": "Number of major coronary vessels (0-3) visualized by fluoroscopy",
        "type": "discrete",
        "categories": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"}
    },
    "thal": {
        "label": "Thallium Stress Scintigraphy (Thal)",
        "description": "Myocardial perfusion scintigraphy status",
        "type": "categorical",
        "categories": {
            3: "Normal",
            6: "Fixed Defect (Prior Infarction)",
            7: "Reversible Defect (Ischemia)"
        }
    },
    "target": {
        "label": "Heart Disease Diagnosis",
        "description": "Presence of coronary artery disease (>50% angiographic stenosis)",
        "type": "binary",
        "categories": {0: "Absent (Negative)", 1: "Present (Positive)"}
    }
}


def download_uci_dataset(destination_path: Path, url: str = UCI_DATA_URL) -> Path:
    """Download the Cleveland dataset from UCI repository if not already present.

    Parameters
    ----------
    destination_path : Path
        Target file path to save raw data.
    url : str
        Source URL of the processed Cleveland data.

    Returns
    -------
    Path
        Path to the downloaded or existing file.

    Raises
    ------
    RuntimeError
        If download fails and no local file exists.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and destination_path.stat().st_size > 0:
        logger.info("Found cached raw dataset at: %s", destination_path)
        return destination_path

    logger.info("Downloading dataset from: %s", url)
    try:
        urllib.request.urlretrieve(url, destination_path)
        logger.info("Successfully downloaded raw data to: %s", destination_path)
        return destination_path
    except Exception as exc:
        error_msg = (
            f"Failed to automatically download dataset from {url}.\n"
            f"Reason: {exc}\n"
            f"MANUAL FALLBACK INSTRUCTIONS:\n"
            f"1. Download the file manually from:\n   {url}\n"
            f"2. Place the file at:\n   {destination_path.resolve()}\n"
            f"3. Re-run your command."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from exc


def load_heart_disease_data(
    raw_path: Optional[str] = "data/raw/processed.cleveland.data",
    force_download: bool = False
) -> pd.DataFrame:
    """Load, parse, validate, and convert the UCI Cleveland dataset.

    Parameters
    ----------
    raw_path : str, optional
        Path to raw data file.
    force_download : bool, default False
        Whether to re-download even if file exists locally.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 13 clinical features and binary 'target' column.
    """
    path = Path(raw_path)
    if force_download and path.exists():
        path.unlink()

    # Step 1: Ensure raw file is downloaded / present
    download_uci_dataset(path)

    # Step 2: Read CSV; '?' represents missing values in raw Cleveland data
    df_raw = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values="?",
        skipinitialspace=True
    )

    logger.info("Raw dataset loaded. Shape: %s", df_raw.shape)

    # Step 3: Validate expected columns and rows
    expected_cols = set(RAW_COLUMNS)
    actual_cols = set(df_raw.columns)
    if not expected_cols.issubset(actual_cols):
        missing = expected_cols - actual_cols
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Step 4: Binary target transformation
    # Raw 'num' uses 0 for absence and 1, 2, 3, 4 for presence (>50% diameter stenosis).
    # Convert to binary: 0 = Absence, 1 = Presence.
    df_clean = df_raw.copy()
    raw_target = pd.to_numeric(df_clean["num"], errors="coerce")
    if raw_target.isna().any():
        raise ValueError("Target column 'num' contains invalid or missing values.")

    df_clean["target"] = (raw_target > 0).astype(int)

    # Step 5: Strictly drop 'num' to ensure no label leakage
    df_clean = df_clean.drop(columns=["num"])

    # Cast numeric feature types appropriately
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Categorical and discrete features (may contain NaNs in 'ca' and 'thal')
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    logger.info(
        "Data validation complete. Final columns: %s. Target balance: %s",
        list(df_clean.columns),
        dict(df_clean["target"].value_counts())
    )

    return df_clean


def get_feature_and_target_matrices(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from the binary target (y).

    Ensures that the target is never included in X.
    """
    if "target" not in df.columns:
        raise KeyError("Column 'target' is missing from DataFrame.")

    X = df[FEATURE_NAMES].copy()
    y = df["target"].copy()

    # Integrity verification
    assert "target" not in X.columns, "Data leakage: 'target' found inside feature matrix X!"
    assert "num" not in X.columns, "Data leakage: raw 'num' found inside feature matrix X!"
    assert list(X.columns) == FEATURE_NAMES, "Feature columns must match exact 13 clinical predictors."

    return X, y


if __name__ == "__main__":
    df = load_heart_disease_data()
    X, y = get_feature_and_target_matrices(df)
    print(f"Data successfully loaded. Samples: {len(df)}, Features: {X.shape[1]}")
    print(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    # Cache processed dataset for reference
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "heart_cleveland_clean.csv", index=False)
    print(f"Cleaned dataset saved to {processed_dir / 'heart_cleveland_clean.csv'}")
