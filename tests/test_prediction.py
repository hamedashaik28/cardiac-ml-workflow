"""Unit tests for inference, input schema validation, and risk prediction output."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.predict import CardiacPredictor, VALIDATION_RANGES
from src.data_loader import FEATURE_NAMES
from src.preprocessing import build_pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def valid_patient_input():
    """A realistic clinical patient sample."""
    return {
        "age": 58,
        "sex": 1,
        "cp": 3,
        "trestbps": 135,
        "chol": 230,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.8,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }


def test_validate_input_dict_valid(valid_patient_input):
    """Confirm valid input dictionary is correctly formatted into a DataFrame."""
    df_sample = CardiacPredictor.validate_input_dict(valid_patient_input)
    assert isinstance(df_sample, pd.DataFrame)
    assert len(df_sample) == 1
    assert list(df_sample.columns) == FEATURE_NAMES
    assert df_sample["age"].iloc[0] == 58.0


def test_validate_input_dict_missing_key(valid_patient_input):
    """Ensure missing features trigger a descriptive ValueError."""
    del valid_patient_input["age"]
    with pytest.raises(ValueError, match="missing required clinical features"):
        CardiacPredictor.validate_input_dict(valid_patient_input)


def test_validate_input_dict_out_of_bounds(valid_patient_input):
    """Ensure values exceeding clinical boundaries trigger ValueError."""
    # Blood pressure of 350 mm Hg is physiologically invalid
    valid_patient_input["trestbps"] = 350
    with pytest.raises(ValueError, match="outside clinical boundary"):
        CardiacPredictor.validate_input_dict(valid_patient_input)


def test_validate_input_dict_invalid_thal(valid_patient_input):
    """Ensure thal codes outside {3, 6, 7} are rejected."""
    valid_patient_input["thal"] = 5
    with pytest.raises(ValueError, match="Thal scan value must be 3"):
        CardiacPredictor.validate_input_dict(valid_patient_input)


def test_predictor_inference_and_probabilities(valid_patient_input, tmp_path):
    """Test full pipeline inference using a trained pipeline artifact or fixture."""
    # Create and fit a minimal pipeline on dummy data to test predictor methods
    dummy_X = pd.DataFrame([valid_patient_input, valid_patient_input])
    dummy_y = np.array([0, 1])

    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(dummy_X, dummy_y)

    temp_model_path = tmp_path / "test_pipeline.joblib"
    import joblib
    joblib.dump(pipeline, temp_model_path)

    predictor = CardiacPredictor(model_path=str(temp_model_path))
    result = predictor.predict_risk(valid_patient_input)

    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in ["Heart Disease Present", "Heart Disease Absent"]
    assert 0.0 <= result["probability_disease"] <= 1.0
    assert 0.0 <= result["probability_absence"] <= 1.0
    assert np.isclose(result["probability_disease"] + result["probability_absence"], 1.0)
    assert result["risk_tier"] in [
        "Low Statistical Risk",
        "Moderate / Intermediate Risk",
        "Elevated / High Statistical Risk"
    ]
    assert isinstance(result["contributing_factors"], list)
