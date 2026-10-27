"""Inference, input validation, and risk qualification module.

Ensures:
1. Exact feature schema matching (13 features in correct clinical sequence).
2. Input range and type validation with beginner-friendly, descriptive errors.
3. Loading fitted pipeline without retraining.
4. Output of binary prediction, calibrated probability score, and clinical risk tier.
5. Highlighting key contributing clinical factors for interpretability.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import FEATURE_NAMES, CLINICAL_DICTIONARY

# Clinical parameter boundaries for validation
VALIDATION_RANGES = {
    "age": (18, 110, "years"),
    "sex": (0, 1, "0=Female, 1=Male"),
    "cp": (1, 4, "1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic"),
    "trestbps": (70, 260, "mm Hg"),
    "chol": (80, 700, "mg/dl"),
    "fbs": (0, 1, "0=Normal, 1=Elevated (>120 mg/dl)"),
    "restecg": (0, 2, "0=Normal, 1=ST-T Abnormality, 2=LVH"),
    "thalach": (50, 240, "bpm"),
    "exang": (0, 1, "0=No, 1=Yes"),
    "oldpeak": (0.0, 10.0, "mm"),
    "slope": (1, 3, "1=Upsloping, 2=Flat, 3=Downsloping"),
    "ca": (0, 3, "0, 1, 2, or 3 vessels"),
    "thal": (3, 7, "3=Normal, 6=Fixed Defect, 7=Reversible Defect"),
}


class CardiacPredictor:
    """Predictor class that manages pipeline loading, schema validation, and inference."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Default to best pipeline in models directory
            model_path = str(Path("models/best_cardiac_pipeline.joblib"))
        self.model_path = Path(model_path)
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        """Load the persisted scikit-learn pipeline."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {self.model_path.resolve()}.\n"
                f"Please execute the training pipeline first: python src/train.py"
            )
        return joblib.load(self.model_path)

    @staticmethod
    def validate_input_dict(input_dict: Dict[str, Any]) -> pd.DataFrame:
        """Validate input features and return a 1-row DataFrame matching the training schema.

        Parameters
        ----------
        input_dict : Dict[str, Any]
            Dictionary containing 13 clinical predictors.

        Returns
        -------
        pd.DataFrame
            Validated 1-row DataFrame with correct column names and order.

        Raises
        ------
        ValueError
            If features are missing or values fall outside physiologically valid bounds.
        """
        # 1. Check for missing features
        missing_keys = set(FEATURE_NAMES) - set(input_dict.keys())
        if missing_keys:
            raise ValueError(f"Input is missing required clinical features: {sorted(list(missing_keys))}")

        # 2. Check bounds and data types
        validated_row = {}
        for feat in FEATURE_NAMES:
            raw_val = input_dict[feat]
            try:
                val = float(raw_val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature '{feat}' must be a valid numeric value, received: {raw_val}") from exc

            if np.isnan(val):
                raise ValueError(f"Feature '{feat}' cannot be NaN or empty.")

            min_val, max_val, note = VALIDATION_RANGES[feat]
            if feat == "thal":
                if int(val) not in [3, 6, 7]:
                    raise ValueError(f"Thal scan value must be 3 (Normal), 6 (Fixed), or 7 (Reversible), received: {val}")
            elif not (min_val <= val <= max_val):
                raise ValueError(
                    f"Feature '{feat}' value {val} is outside clinical boundary [{min_val}, {max_val}] ({note})."
                )

            validated_row[feat] = val

        # 3. Assemble single-row DataFrame in the exact training sequence
        df_input = pd.DataFrame([validated_row], columns=FEATURE_NAMES)
        return df_input

    def predict_risk(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clinical risk inference for a single patient sample.

        Parameters
        ----------
        input_dict : Dict[str, Any]
            Dictionary of 13 clinical predictors.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing binary prediction, probability, risk tier,
            and qualitative factors.
        """
        df_sample = self.validate_input_dict(input_dict)

        # Binary classification (0 = Absent, 1 = Present)
        prediction = int(self.pipeline.predict(df_sample)[0])

        # Risk probability
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(df_sample)[0]
            prob_absence = float(probabilities[0])
            prob_presence = float(probabilities[1])
        else:
            prob_presence = 1.0 if prediction == 1 else 0.0
            prob_absence = 1.0 - prob_presence

        # Clinical Risk Tiering
        if prob_presence < 0.35:
            risk_tier = "Low Statistical Risk"
            risk_color = "#28a745"  # Green
            clinical_implication = (
                "Patient profile demonstrates low statistical concordance with coronary stenosis "
                "in the Cleveland cohort. Standard preventative lifestyle counsel applies."
            )
        elif prob_presence < 0.65:
            risk_tier = "Moderate / Intermediate Risk"
            risk_color = "#fd7e14"  # Orange
            clinical_implication = (
                "Intermediate risk profile. Clinical ambiguity warrants standard follow-up "
                "or secondary non-invasive functional testing (e.g., echocardiography, stress ECG review)."
            )
        else:
            risk_tier = "Elevated / High Statistical Risk"
            risk_color = "#dc3545"  # Red
            clinical_implication = (
                "High statistical probability of significant coronary artery stenosis (>50% narrowing). "
                "Prompt clinical correlation and cardiology evaluation recommended."
            )

        # Identify contributing risk factors
        key_factors = []
        if df_sample["cp"].values[0] == 4:
            key_factors.append("Asymptomatic chest pain (statistically associated with silent ischemia/higher risk in cohort)")
        if df_sample["thal"].values[0] == 7:
            key_factors.append("Reversible thallium perfusion defect (suggestive of myocardial ischemia)")
        elif df_sample["thal"].values[0] == 6:
            key_factors.append("Fixed thallium defect (prior myocardial infarction)")
        if df_sample["ca"].values[0] > 0:
            key_factors.append(f"{int(df_sample['ca'].values[0])} major coronary vessel(s) visualized under fluoroscopy")
        if df_sample["oldpeak"].values[0] >= 2.0:
            key_factors.append(f"Significant exercise-induced ST depression ({df_sample['oldpeak'].values[0]} mm)")
        if df_sample["exang"].values[0] == 1:
            key_factors.append("Exercise-induced angina provoked during stress test")
        if df_sample["age"].values[0] >= 60:
            key_factors.append(f"Advanced age ({int(df_sample['age'].values[0])} years)")
        if df_sample["trestbps"].values[0] >= 140:
            key_factors.append(f"Stage 2 hypertensive resting blood pressure ({int(df_sample['trestbps'].values[0])} mm Hg)")

        return {
            "prediction": prediction,
            "prediction_label": "Heart Disease Present" if prediction == 1 else "Heart Disease Absent",
            "probability_disease": round(prob_presence, 4),
            "probability_absence": round(prob_absence, 4),
            "risk_percentage": round(prob_presence * 100, 1),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "clinical_implication": clinical_implication,
            "contributing_factors": key_factors,
            "input_features": input_dict,
        }
