"""Unit tests for scikit-learn preprocessing and leakage prevention."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    create_preprocessor,
    build_pipeline,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    get_feature_names_out,
)
from src.data_loader import FEATURE_NAMES


@pytest.fixture
def sample_cardiac_df():
    """Create a synthetic mini-DataFrame replicating cardiac feature schema with intentional NaNs."""
    return pd.DataFrame({
        "age": [55.0, 60.0, np.nan, 45.0],
        "sex": [1.0, 0.0, 1.0, 0.0],
        "cp": [4.0, 3.0, 2.0, 1.0],
        "trestbps": [140.0, 120.0, 130.0, 110.0],
        "chol": [240.0, 200.0, 280.0, np.nan],
        "fbs": [0.0, 1.0, 0.0, 0.0],
        "restecg": [0.0, 1.0, 2.0, 0.0],
        "thalach": [150.0, 160.0, 130.0, 170.0],
        "exang": [0.0, 1.0, 0.0, 1.0],
        "oldpeak": [1.5, 0.0, 2.3, 0.5],
        "slope": [2.0, 1.0, 3.0, 1.0],
        "ca": [0.0, np.nan, 2.0, 0.0],
        "thal": [3.0, 7.0, np.nan, 3.0],
    })


def test_create_preprocessor_structure():
    """Verify that create_preprocessor returns a valid ColumnTransformer."""
    preprocessor = create_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_imputation_and_scaling(sample_cardiac_df):
    """Ensure that missing values are completely imputed and output contains no NaNs."""
    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(sample_cardiac_df)

    assert not np.isnan(transformed).any(), "Preprocessed output contains unhandled NaNs!"
    assert transformed.shape[0] == len(sample_cardiac_df)


def test_no_data_leakage_between_train_and_test():
    """Verify that training set statistics are not altered when transforming test data."""
    train_df = pd.DataFrame({
        "age": [40.0, 50.0, 60.0],
        "sex": [1.0, 0.0, 1.0],
        "cp": [1.0, 2.0, 3.0],
        "trestbps": [120.0, 130.0, 140.0],
        "chol": [200.0, 220.0, 240.0],
        "fbs": [0.0, 0.0, 1.0],
        "restecg": [0.0, 0.0, 1.0],
        "thalach": [160.0, 150.0, 140.0],
        "exang": [0.0, 1.0, 0.0],
        "oldpeak": [0.0, 1.0, 2.0],
        "slope": [1.0, 2.0, 3.0],
        "ca": [0.0, 1.0, 0.0],
        "thal": [3.0, 7.0, 3.0],
    })

    test_df = pd.DataFrame({
        "age": [80.0, 90.0, 100.0],  # Extreme values
        "sex": [1.0, 1.0, 0.0],
        "cp": [4.0, 4.0, 4.0],
        "trestbps": [180.0, 190.0, 200.0],
        "chol": [350.0, 400.0, 450.0],
        "fbs": [1.0, 1.0, 1.0],
        "restecg": [2.0, 2.0, 2.0],
        "thalach": [100.0, 90.0, 80.0],
        "exang": [1.0, 1.0, 1.0],
        "oldpeak": [3.5, 4.0, 5.0],
        "slope": [3.0, 3.0, 3.0],
        "ca": [3.0, 3.0, 3.0],
        "thal": [7.0, 6.0, 7.0],
    })

    preprocessor = create_preprocessor()
    preprocessor.fit(train_df)

    # Recorded train scaler mean for 'age' (mean of 40, 50, 60 is 50.0)
    train_scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    age_idx = NUMERIC_FEATURES.index("age")
    train_mean_age = train_scaler.mean_[age_idx]
    assert np.isclose(train_mean_age, 50.0), f"Expected train mean age 50.0, got {train_mean_age}"

    # Transform test set
    _ = preprocessor.transform(test_df)

    # Scaler mean must remain unchanged after transforming test set
    assert np.isclose(train_scaler.mean_[age_idx], 50.0), "Data leakage: scaler mean altered by test set!"


def test_build_pipeline():
    """Verify that build_pipeline creates a complete two-step Pipeline."""
    clf = LogisticRegression()
    pipeline = build_pipeline(clf)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
