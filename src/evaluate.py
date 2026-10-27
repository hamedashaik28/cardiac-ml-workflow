"""Comprehensive clinical evaluation metrics and visualization module.

Calculates:
- Accuracy
- Balanced Accuracy
- Sensitivity / Recall (clinical priority: avoiding missed cardiac cases)
- Specificity (True Negative Rate)
- Precision (Positive Predictive Value)
- F1-Score
- ROC-AUC (Area Under the ROC Curve)
- Confusion Matrix (TN, FP, FN, TP counts and clinical consequences)
- Brier Score (probability calibration quality)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    brier_score_loss,
)

# Clinical styling defaults for clean, readable plots
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute comprehensive healthcare-oriented performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).
    y_proba : np.ndarray, optional
        Predicted probability of the positive class (class 1).

    Returns
    -------
    Dict[str, float]
        Dictionary of formatted metric names and scalar scores.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
            metrics["brier_score"] = float("nan")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Generate an annotated clinical confusion matrix figure.

    Explains True Negatives, False Positives (over-triage),
    False Negatives (under-triage / high clinical risk), and True Positives.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)

    labels = [
        [f"True Neg (TN)\n{tn}\n({tn/total:.1%})", f"False Pos (FP)\n{fp}\n({fp/total:.1%})\n[Over-triage]"],
        [f"False Neg (FN)\n{fn}\n({fn/total:.1%})\n[CRITICAL RISK]", f"True Pos (TP)\n{tp}\n({tp/total:.1%})"]
    ]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm,
        annot=np.array(labels),
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted: Absence (0)", "Predicted: Disease (1)"],
        yticklabels=["Actual: Absence (0)", "Actual: Disease (1)"],
        ax=ax,
        linewidths=1.5,
        linecolor="#EEEEEE"
    )
    ax.set_title(f"Clinical Confusion Matrix - {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Actual Clinical Diagnosis", fontsize=11, fontweight="bold")
    ax.set_xlabel("Pipeline Predicted Classification", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_roc_curves_comparison(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot overlaid ROC curves comparing multiple fitted pipelines on held-out test data."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, (name, pipeline) in enumerate(models_dict.items()):
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_val = roc_auc_score(y_test, y_proba)
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")

    # Chance baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Test Set: ROC Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#F9F9F9", edgecolor="#DDDDDD", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_feature_importance_rf(
    pipeline,
    feature_names: List[str],
    top_n: int = 12,
    save_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """Plot Gini feature importances from a Random Forest classifier within a pipeline."""
    try:
        rf = pipeline.named_steps["classifier"]
        importances = rf.feature_importances_
        if len(importances) != len(feature_names):
            feature_names = [f"feat_{i}" for i in range(len(importances))]

        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
        sns.barplot(
            data=feat_df,
            x="Importance",
            y="Feature",
            palette="Blues_r",
            ax=ax,
            hue="Feature",
            legend=False
        )
        ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Clinical Predictor (Transformed)", fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

        return fig
    except Exception as exc:
        print(f"Could not plot feature importance: {exc}")
        return None


def generate_eda_figures(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create publication-ready exploratory figures for the dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    counts = df["target"].value_counts()
    pcts = (counts / len(df) * 100).round(1)
    bars = ax.bar(["Absence (0)", "Presence (1)"], counts.values, color=["#2b5c8f", "#d95f02"], width=0.55)
    for bar, pct in zip(bars, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{int(bar.get_height())} ({pct}%)",
            ha="center",
            va="bottom",
            fontweight="bold"
        )
    ax.set_ylim(0, max(counts.values) + 25)
    ax.set_ylabel("Patient Count", fontsize=11, fontweight="bold")
    ax.set_title("Target Distribution: Coronary Artery Disease Status", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    p1 = output_dir / "eda_target_distribution.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    generated.append(p1)

    # 2. Correlation Heatmap (numerical features + target)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "target"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.8, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap: Key Continuous Predictors & Target", fontsize=12, fontweight="bold", pad=10)
    p2 = output_dir / "eda_correlation_heatmap.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    generated.append(p2)

    # 3. Age vs Maximum Heart Rate by Target
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sns.scatterplot(
        data=df,
        x="age",
        y="thalach",
        hue="target",
        palette={0: "#2b5c8f", 1: "#d95f02"},
        alpha=0.85,
        s=55,
        ax=ax
    )
    ax.set_title("Age vs. Maximum Heart Rate (thalach) by Cardiac Risk", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Age (years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Max Heart Rate Achieved (bpm)", fontsize=11, fontweight="bold")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["No Disease (0)", "Disease Present (1)"], title="Diagnosis")
    ax.grid(True, linestyle=":", alpha=0.5)
    p3 = output_dir / "eda_age_thalach_by_target.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    generated.append(p3)

    # 4. Chest Pain Type vs Target
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    cp_map = {1: "1: Typical", 2: "2: Atypical", 3: "3: Non-Anginal", 4: "4: Asymptomatic"}
    df_plot = df.copy()
    df_plot["cp_label"] = df_plot["cp"].map(cp_map)
    cp_order = ["1: Typical", 2, 3, 4]
    sns.countplot(
        data=df_plot,
        x="cp_label",
        hue="target",
        palette={0: "#2b5c8f", 1: "#d95f02"},
        ax=ax
    )
    ax.set_title("Chest Pain Type Distribution by Cardiac Diagnosis", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Chest Pain Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=11, fontweight="bold")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["No Disease (0)", "Disease Present (1)"], title="Diagnosis")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    p4 = output_dir / "eda_chest_pain_by_target.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    generated.append(p4)

    return generated
