"""Preprocessing module using scikit-learn ColumnTransformer and Pipeline.

Key engineering safeguards:
1. Complete prevention of data leakage: all imputation parameters (medians, modes)
   and scaling statistics (means, standard deviations) are computed exclusively on
   training folds.
2. Robust missing value handling:
   - Numerical features: median imputation.
   - Categorical features: most-frequent imputation.
3. Feature scaling: StandardScaler applied to numerical features for gradient-based
   models (Logistic Regression).
4. One-Hot Encoding: OneHotEncoder with handle_unknown='ignore' so live user inputs
   or unseen categories in deployment never cause runtime exceptions.
"""

from typing import List
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Clinical feature groupings
NUMERIC_FEATURES: List[str] = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES: List[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


def create_preprocessor() -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer for cardiac predictors.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor encapsulating numeric and categorical sub-pipelines.
    """
    # Numerical sub-pipeline: Median Imputation -> Standard Scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical sub-pipeline: Mode Imputation -> One-Hot Encoding
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # ColumnTransformer orchestrates feature transformations without leakage
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drop any unlisted columns (safeguard against accidental target leakage)
    )

    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """Build a complete, end-to-end scikit-learn Pipeline with preprocessing and classifier.

    Parameters
    ----------
    classifier : BaseEstimator
        scikit-learn classifier (e.g., LogisticRegression, RandomForestClassifier).

    Returns
    -------
    Pipeline
        Full scikit-learn pipeline ready for fit/predict.
    """
    preprocessor = create_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve the names of all transformed features from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted ColumnTransformer instance.

    Returns
    -------
    List[str]
        List of expanded feature names.
    """
    try:
        # scikit-learn 1.0+ get_feature_names_out
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        names = list(NUMERIC_FEATURES)
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
            names.extend(list(cat_names))
        except Exception:
            names.extend([f"cat_{i}" for i in range(15)])
        return names
