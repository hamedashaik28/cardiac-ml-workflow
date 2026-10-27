"""Unit tests for data loading, parsing, and target conversion."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import (
    load_heart_disease_data,
    get_feature_and_target_matrices,
    FEATURE_NAMES,
    RAW_COLUMNS,
)


def test_load_heart_disease_data_schema():
    """Verify that loaded dataset has expected dimensions and column names."""
    df = load_heart_disease_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 303, f"Expected 303 rows in Cleveland dataset, got {len(df)}"

    expected_cols = FEATURE_NAMES + ["target"]
    assert list(df.columns) == expected_cols, "Cleaned dataset columns do not match expected schema."


def test_binary_target_conversion():
    """Ensure that the multi-class angiographic diagnosis is converted to 0 and 1 only."""
    df = load_heart_disease_data()
    unique_targets = set(df["target"].unique())
    assert unique_targets == {0, 1}, f"Target must only contain 0 and 1, got {unique_targets}"

    # Cleveland benchmark: 164 absence (0) and 139 presence (1)
    counts = df["target"].value_counts().to_dict()
    assert counts[0] == 164, f"Expected 164 negative cases, found {counts[0]}"
    assert counts[1] == 139, f"Expected 139 positive cases, found {counts[1]}"


def test_no_target_leakage_in_features():
    """Verify that neither 'target' nor 'num' exist in feature matrix X."""
    df = load_heart_disease_data()
    X, y = get_feature_and_target_matrices(df)

    assert "target" not in X.columns, "Leakage detected: 'target' present in X!"
    assert "num" not in X.columns, "Leakage detected: 'num' present in X!"
    assert len(X.columns) == 13, f"Expected exactly 13 features, got {len(X.columns)}"
    assert list(X.columns) == FEATURE_NAMES
    assert len(y) == len(X)


def test_missing_values_parsed_as_nan():
    """Confirm that '?' symbols in raw data are correctly parsed as NaNs in X."""
    df = load_heart_disease_data()
    # In the Cleveland dataset, 'ca' has 4 missing and 'thal' has 2 missing values
    assert df["ca"].isna().sum() == 4, f"Expected 4 missing in ca, got {df['ca'].isna().sum()}"
    assert df["thal"].isna().sum() == 2, f"Expected 2 missing in thal, got {df['thal'].isna().sum()}"
