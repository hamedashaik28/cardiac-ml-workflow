"""Model training, cross-validation, selection, and artifact persistence.

Rigorous Machine Learning Methodology:
1. Fixed random seed (42) for complete reproducibility.
2. Stratified train/test split (80/20) preserving positive:negative ratio.
3. 5-fold Stratified Cross-Validation executed exclusively on the training set
   to select the optimal architecture without test set contamination.
4. Preprocessing encapsulated in scikit-learn Pipeline so imputer and scaler
   statistics are computed strictly per training fold.
5. Held-out test set evaluated exactly ONCE for unbiased generalization estimate.
6. Persistence of complete fitted pipelines, metadata, metrics, and figures.
"""

import sys
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is in sys.path whether executed as module or direct script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_heart_disease_data,
    get_feature_and_target_matrices,
    FEATURE_NAMES,
)
from src.preprocessing import build_pipeline, get_feature_names_out
from src.evaluate import (
    calculate_clinical_metrics,
    plot_confusion_matrix,
    plot_roc_curves_comparison,
    plot_feature_importance_rf,
    generate_eda_figures,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def run_training_pipeline(
    raw_data_path: str = "data/raw/processed.cleveland.data",
    models_dir: str = "models",
    reports_dir: str = "reports"
):
    """Execute the end-to-end training and evaluation pipeline."""
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    figures_path = reports_path / "figures"

    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load and Validate Data
    # ---------------------------------------------------------
    logger.info("Step 1: Loading UCI Cleveland Heart Disease dataset...")
    df = load_heart_disease_data(raw_path=raw_data_path)
    X, y = get_feature_and_target_matrices(df)

    logger.info("Generating EDA figures for presentation and reporting...")
    generate_eda_figures(df, figures_path)

    # ---------------------------------------------------------
    # 2. Stratified Train / Held-Out Test Split
    # ---------------------------------------------------------
    logger.info("Step 2: Splitting data (80%% train / 20%% test, stratified, seed=%d)...", RANDOM_SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    logger.info("Train set shape: %s (Positive class: %d)", X_train.shape, int(y_train.sum()))
    logger.info("Held-out test set shape: %s (Positive class: %d)", X_test.shape, int(y_test.sum()))

    # ---------------------------------------------------------
    # 3. Define Candidate Architectures with Full Pipelines
    # ---------------------------------------------------------
    # Model A: Logistic Regression (L2 regularized, max_iter=1000)
    # Model B: Random Forest Classifier (100 trees, max_depth=5, min_samples_split=4, seed=42)
    candidate_pipelines = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver="lbfgs")
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=4,
                random_state=RANDOM_SEED
            )
        )
    }

    # ---------------------------------------------------------
    # 4. Training-Only 5-Fold Stratified Cross-Validation
    # ---------------------------------------------------------
    logger.info("Step 3: Performing 5-Fold Stratified Cross-Validation on training data only...")
    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_scorers = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results_summary = {}
    for name, pipeline in candidate_pipelines.items():
        logger.info("Running Cross-Validation for: %s", name)
        cv_out = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=cv_scorers,
            return_train_score=False
        )

        cv_results_summary[name] = {
            "cv_accuracy_mean": float(np.mean(cv_out["test_accuracy"])),
            "cv_accuracy_std": float(np.std(cv_out["test_accuracy"])),
            "cv_balanced_accuracy_mean": float(np.mean(cv_out["test_balanced_accuracy"])),
            "cv_balanced_accuracy_std": float(np.std(cv_out["test_balanced_accuracy"])),
            "cv_precision_mean": float(np.mean(cv_out["test_precision"])),
            "cv_precision_std": float(np.std(cv_out["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_out["test_recall"])),
            "cv_recall_std": float(np.std(cv_out["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_out["test_f1"])),
            "cv_f1_std": float(np.std(cv_out["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_out["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_out["test_roc_auc"])),
        }

    # ---------------------------------------------------------
    # 5. Model Selection Based on CV Performance
    # ---------------------------------------------------------
    # Primary selection criteria: Cross-validation ROC-AUC
    best_model_name = max(
        cv_results_summary.keys(),
        key=lambda k: cv_results_summary[k]["cv_roc_auc_mean"]
    )
    logger.info("Best model selected via 5-Fold CV ROC-AUC: %s", best_model_name)

    # ---------------------------------------------------------
    # 6. Fit Final Pipelines on Full Training Set & Evaluate Once on Test Set
    # ---------------------------------------------------------
    logger.info("Step 4: Fitting final pipelines on full training set and evaluating on held-out test set...")
    fitted_pipelines = {}
    test_metrics_summary = {}

    for name, pipeline in candidate_pipelines.items():
        # Fit on 100% of training data
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Save individual model pipeline
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, models_path / f"pipeline_{slug}.joblib")

        # Held-out test predictions
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = calculate_clinical_metrics(y_test.values, y_test_pred, y_test_proba)
        test_metrics_summary[name] = metrics

        # Generate and save confusion matrix figure
        cm_path = figures_path / f"confusion_matrix_{slug}.png"
        plot_confusion_matrix(y_test.values, y_test_pred, model_name=f"{name} (Held-out Test)", save_path=cm_path)

    # Plot overlaid ROC curves for test set
    roc_fig_path = figures_path / "roc_curves_comparison.png"
    plot_roc_curves_comparison(fitted_pipelines, X_test, y_test.values, save_path=roc_fig_path)

    # Plot Random Forest feature importances
    rf_pipeline = fitted_pipelines["Random Forest"]
    transformed_features = get_feature_names_out(rf_pipeline.named_steps["preprocessor"])
    fi_path = figures_path / "feature_importance_random_forest.png"
    plot_feature_importance_rf(rf_pipeline, transformed_features, top_n=12, save_path=fi_path)

    # ---------------------------------------------------------
    # 7. Save Best Model Pipeline & Comprehensive Comparison Table
    # ---------------------------------------------------------
    best_pipeline = fitted_pipelines[best_model_name]
    best_pipeline_path = models_path / "best_cardiac_pipeline.joblib"
    joblib.dump(best_pipeline, best_pipeline_path)
    logger.info("Saved best pipeline to: %s", best_pipeline_path)

    # Build side-by-side comparison dataframe
    comparison_rows = []
    for name in candidate_pipelines.keys():
        cv = cv_results_summary[name]
        t = test_metrics_summary[name]
        row = {
            "Model": name,
            "CV ROC-AUC (Mean +/- Std)": f"{cv['cv_roc_auc_mean']:.3f} +/- {cv['cv_roc_auc_std']:.3f}",
            "CV Recall / Sens (Mean)": f"{cv['cv_recall_mean']:.3f}",
            "CV F1-Score (Mean)": f"{cv['cv_f1_mean']:.3f}",
            "CV Accuracy (Mean)": f"{cv['cv_accuracy_mean']:.3f}",
            "Test ROC-AUC": round(t["roc_auc"], 4),
            "Test Recall (Sensitivity)": round(t["recall_sensitivity"], 4),
            "Test Specificity": round(t["specificity"], 4),
            "Test Precision": round(t["precision"], 4),
            "Test F1-Score": round(t["f1_score"], 4),
            "Test Accuracy": round(t["accuracy"], 4),
            "Test Balanced Accuracy": round(t["balanced_accuracy"], 4),
            "Test False Negatives (Missed)": t["false_negatives"],
            "Test False Positives": t["false_positives"],
        }
        comparison_rows.append(row)

    df_comparison = pd.DataFrame(comparison_rows)
    comparison_csv_path = reports_path / "model_comparison.csv"
    df_comparison.to_csv(comparison_csv_path, index=False, encoding="utf-8")
    logger.info("Saved model comparison table to: %s", comparison_csv_path)

    # ---------------------------------------------------------
    # 8. Save Model Metadata JSON
    # ---------------------------------------------------------
    metadata = {
        "project": "End-to-End Machine Learning Workflow for Cardiac Health Risk Prediction",
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "selected_model": best_model_name,
        "feature_names_input": FEATURE_NAMES,
        "numeric_features": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical_features": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "target_mapping": {"0": "Absence of Heart Disease", "1": "Presence of Heart Disease"},
        "cv_results": cv_results_summary,
        "test_results": test_metrics_summary,
    }

    metadata_path = models_path / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metrics_summary_path = reports_path / "metrics_summary.json"
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump({"cv": cv_results_summary, "test": test_metrics_summary}, f, indent=2)

    logger.info("Training pipeline completed successfully.")
    return df_comparison


if __name__ == "__main__":
    df_comp = run_training_pipeline()
    print("\n--- FINAL MODEL COMPARISON TABLE ---")
    print(df_comp.to_string(index=False))
