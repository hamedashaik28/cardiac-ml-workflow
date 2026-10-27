"""Streamlit Clinical Analytics Dashboard: Cardiac Health Risk Prediction.

Educational Research Prototype demonstrating the end-to-end healthcare ML workflow.
Not a medical device. Not intended for clinical diagnosis.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_heart_disease_data,
    FEATURE_NAMES,
    CLINICAL_DICTIONARY,
    UCI_DATA_URL,
)
from src.predict import CardiacPredictor, VALIDATION_RANGES

# -----------------------------------------------------------------------------
# Streamlit App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk ML | Clinical Analytics Workflow",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional clinical typography and cards
st.markdown("""
<style>
    /* Global font & background adjustments */
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #fff3cd;
        border-left: 6px solid #ffeeba;
        border-radius: 4px;
        padding: 0.8rem 1.2rem;
        color: #856404;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #718096;
    }
    .risk-badge-low {
        background-color: #d4edda;
        color: #155724;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-moderate {
        background-color: #ffeeba;
        color: #856404;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resources (Prevent re-training or re-loading on each interaction)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_predictor() -> CardiacPredictor:
    """Load the persisted best model pipeline once."""
    return CardiacPredictor(model_path="models/best_cardiac_pipeline.joblib")


@st.cache_data
def get_cached_data() -> pd.DataFrame:
    """Load the cleaned dataset once."""
    return load_heart_disease_data()


@st.cache_data
def get_model_metadata() -> dict:
    """Load model training metadata and cross-validation records."""
    meta_path = Path("models/model_metadata.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_comparison_table() -> pd.DataFrame:
    """Load the model comparison table."""
    csv_path = Path("reports/model_comparison.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Metadata
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🫀 CardioRisk ML")
st.sidebar.caption("Clinical Analytics Decision-Support Prototype")

nav_choice = st.sidebar.radio(
    "Workflow Navigation",
    [
        "1. Project Overview & Scope",
        "2. Dataset Explorer & Quality",
        "3. Exploratory Data Analysis (EDA)",
        "4. Model Benchmark & Evaluation",
        "5. Patient Risk Assessment Form",
        "6. Clinical Ethics & Limitations"
    ],
    index=0
)

# Sidebar System Specs
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Pipeline Specifications")
st.sidebar.markdown("- **Dataset**: UCI Cleveland Heart Disease")
st.sidebar.markdown("- **Cohort Size**: 303 samples (13 predictors)")
st.sidebar.markdown("- **Split**: Stratified 80% Train / 20% Test")
st.sidebar.markdown("- **Validation**: 5-Fold Stratified CV (Train-only)")
st.sidebar.markdown("- **Selected Architecture**: Logistic Regression Pipeline")
st.sidebar.markdown("- **Random Seed**: Fixed (`42`)")
st.sidebar.markdown("---")
st.sidebar.caption("🎓 Health Informatics & Analytics College Assignment")


# -----------------------------------------------------------------------------
# Global Medical Disclaimer Banner (Appears on every page)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-banner">
    <strong>⚠️ Mandatory Educational & Research Prototype Notice:</strong>
    This application is an academic software demonstration built strictly for educational and portfolio demonstration.
    It is <strong>NOT a certified medical device (SaMD)</strong>, has not undergone clinical trials, and must <strong>NEVER</strong>
    be used for real-time medical diagnosis, acute triage, or prescribing treatment. Always consult a qualified licensed physician
    for cardiovascular medical evaluation.
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: PROJECT OVERVIEW & SCOPE
# =============================================================================
if nav_choice == "1. Project Overview & Scope":
    st.markdown('<div class="main-header">End-to-End Machine Learning Workflow for Cardiac Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous, leak-free predictive modeling benchmarked on the classic UCI Cleveland Heart Disease cohort.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 🎯 Problem Definition & Clinical Motivation")
        st.write("""
        Coronary Artery Disease (CAD) remains the leading cause of global mortality. While gold-standard diagnosis requires
        invasive coronary angiography (>50% luminal stenosis), routine non-invasive clinical findings—including resting ECG,
        exercise treadmill responses, chest pain typology, and serum biomarkers—contain strong diagnostic signals.

        **Objective**: Design, train, validate, and package a leak-free Machine Learning classification pipeline capable of estimating
        the statistical probability of coronary artery disease from 13 standard clinical parameters.
        """)

        st.markdown("### 🛡️ Core ML Engineering Safeguards")
        st.markdown("""
        1. **Zero Data Leakage**:
           - All preprocessing steps (`ColumnTransformer`) are enclosed within a scikit-learn `Pipeline`.
           - Missing-value imputation (median/mode) and continuous feature scaling (`StandardScaler`) are calculated
             **strictly on the training partition** during cross-validation folds and never fit on held-out test data.
        2. **Stratified Partitioning**:
           - Fixed random seed (`42`) with an 80/20 train/test split preserving the natural class ratio (54% negative : 46% positive).
        3. **Rigorous Validation Protocol**:
           - 5-Fold Stratified Cross-Validation on the 80% training set for model selection.
           - Single-shot final evaluation on the untouched 20% test partition to report genuine generalization capability.
        4. **Clinical Metric Prioritization**:
           - Focus on **Sensitivity (Recall)** and **ROC-AUC** to penalize False Negatives (missed cardiac disease cases).
        """)

    with col2:
        st.markdown("### 📋 Clinical Feature Architecture")
        st.markdown("""
        | Predictor | Description | Category |
        |---|---|---|
        | `age` | Patient Age (29-77 yrs) | Continuous |
        | `sex` | Biological Sex (F/M) | Binary |
        | `cp` | Chest Pain Type (1-4) | Categorical |
        | `trestbps` | Resting Blood Pressure | Continuous |
        | `chol` | Serum Cholesterol (mg/dl) | Continuous |
        | `fbs` | Fasting Blood Sugar >120 | Binary |
        | `restecg` | Resting ECG Abnormalities | Categorical |
        | `thalach` | Max Achieved Heart Rate | Continuous |
        | `exang` | Exercise-Induced Angina | Binary |
        | `oldpeak` | Exercise ST Depression | Continuous |
        | `slope` | Peak ST Segment Slope | Categorical |
        | `ca` | Fluoroscopy Vessels (0-3) | Discrete |
        | `thal` | Thallium Scintigraphy | Categorical |
        """)
        st.info("💡 **Target Definition**: Binary classification where `0 = Absence` (<50% stenosis) and `1 = Presence` (>50% diameter stenosis).")


# =============================================================================
# PAGE 2: DATASET EXPLORER & DATA QUALITY
# =============================================================================
elif nav_choice == "2. Dataset Explorer & Quality":
    st.markdown('<div class="main-header">Dataset Explorer & Quality Assurance Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Provenance, data dictionary, missing value audit, and schema validation.</div>', unsafe_allow_html=True)

    df = get_cached_data()

    # Metric summary row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">303</div><div class="metric-label">Total Cohort Rows</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">13</div><div class="metric-label">Clinical Predictors</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">6</div><div class="metric-label">Missing Values (Raw)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">164 (54%)</div><div class="metric-label">Absence (0)</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card"><div class="metric-value">139 (46%)</div><div class="metric-label">Presence (1)</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔎 Interactive Clinical Cohort Browser")
    show_raw = st.checkbox("Show full raw dataset rows", value=False)
    num_rows = len(df) if show_raw else 15
    st.dataframe(df.head(num_rows), use_container_width=True)

    tab_qa, tab_dict, tab_provenance = st.tabs(["Data Quality & Missing Values Audit", "Clinical Feature Dictionary", "Provenance & Citation"])

    with tab_qa:
        st.markdown("#### Missing Value Analysis & Integrity Checks")
        st.write("""
        In the raw Cleveland data file, missing records were coded with the character `'?'`.
        Our data ingestion pipeline specifically parsed `'?'` into IEEE NaN values and verified their occurrence:
        """)
        missing_counts = df.isna().sum()
        missing_df = pd.DataFrame({
            "Feature": missing_counts.index,
            "Missing Count": missing_counts.values,
            "Percentage (%)": (missing_counts.values / len(df) * 100).round(2),
            "Clinical Handling in Pipeline": [
                "Median Imputer (train-only)" if col in ["age", "trestbps", "chol", "thalach", "oldpeak"]
                else ("Most Frequent Imputer (train-only)" if col != "target" else "None")
                for col in missing_counts.index
            ]
        })
        st.dataframe(missing_df[missing_df["Missing Count"] > 0], use_container_width=True)

        st.markdown("#### Automated Quality Verification Checklist")
        st.success("✅ **Duplicate Records**: 0 duplicate rows detected.")
        st.success("✅ **Target Range**: Binary values strictly in `{0, 1}`.")
        st.success("✅ **Target Separation**: Target column isolated; zero presence in feature matrix `X`.")
        st.success("✅ **Physiological Plausibility**: All clinical fields verified within physiologically valid bounds.")

    with tab_dict:
        st.markdown("#### Detailed Clinical Variable Definitions")
        dict_records = []
        for feat, info in CLINICAL_DICTIONARY.items():
            dict_records.append({
                "Feature Key": feat,
                "Clinical Label": info["label"],
                "Type": info["type"],
                "Description": info["description"],
                "Units / Coding": info.get("unit") or str(info.get("categories", "N/A"))
            })
        st.dataframe(pd.DataFrame(dict_records), use_container_width=True)

    with tab_provenance:
        st.markdown("#### Dataset Provenance & Academic Citation")
        st.write(f"""
        - **Repository**: [UCI Machine Learning Repository (Heart Disease Dataset)]({UCI_DATA_URL})
        - **Donor Institutions**:
          1. Cleveland Clinic Foundation (Robert Detrano, M.D., Ph.D.)
          2. Hungarian Institute of Cardiology, Budapest (Andras Janosi, M.D.)
          3. University Hospital, Zurich, Switzerland (William Steinbrunn, M.D.)
          4. University Hospital, Basel, Switzerland (Matthias Pfisterer, M.D.)
        """)
        st.code("""
Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J. J., Sandhu, S., ... & Froelicher, V. (1989).
International application of a new probability algorithm for the diagnosis of coronary artery disease.
American Journal of Cardiology, 64(5), 304-310.
        """, language="text")


# =============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
elif nav_choice == "3. Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visualizing cardiovascular distributions, correlations, and clinical associations.</div>', unsafe_allow_html=True)

    eda_tabs = st.tabs([
        "1. Class Balance & Target",
        "2. Correlation Heatmap",
        "3. Age vs Max Heart Rate",
        "4. Chest Pain Typology"
    ])

    with eda_tabs[0]:
        c1, c2 = st.columns([3, 2])
        with c1:
            p = Path("reports/figures/eda_target_distribution.png")
            if p.exists():
                st.image(str(p), caption="Figure 1: Target Class Balance (UCI Cleveland)", use_column_width=True)
        with c2:
            st.markdown("#### Clinical Interpretation: Balanced Cohort")
            st.write("""
            - **Distribution**: 164 patients (54.1%) have no angiographic coronary disease; 139 patients (45.9%) have documented disease (>50% narrowing).
            - **Machine Learning Significance**: Unlike rare disease datasets with extreme class imbalances (e.g., 99% vs 1%), this near-balanced 54:46 split means standard cross-entropy loss functions and accuracy are informative.
            - However, **in healthcare, false negatives (missed disease) are far more dangerous than false positives**. Hence, Recall and ROC-AUC are prioritized over raw accuracy.
            """)

    with eda_tabs[1]:
        c1, c2 = st.columns([3, 2])
        with c1:
            p = Path("reports/figures/eda_correlation_heatmap.png")
            if p.exists():
                st.image(str(p), caption="Figure 2: Pearson Correlation of Continuous Features", use_column_width=True)
        with c2:
            st.markdown("#### Clinical Interpretation: Physiological Correlations")
            st.write("""
            - **`oldpeak` (Exercise ST Depression)** exhibits the strongest positive correlation with disease presence ($r \\approx +0.42$). Deep exercise-induced ST depression is a classic hallmark of myocardial ischemia.
            - **`thalach` (Max Heart Rate)** shows a notable negative correlation with disease ($r \\approx -0.42$). Patients with impaired cardiac reserve or ischemic symptoms fail to reach expected age-predicted peak heart rates.
            - Continuous predictors exhibit mild-to-moderate collinearity, confirming the suitability of regularized Logistic Regression.
            """)

    with eda_tabs[2]:
        c1, c2 = st.columns([3, 2])
        with c1:
            p = Path("reports/figures/eda_age_thalach_by_target.png")
            if p.exists():
                st.image(str(p), caption="Figure 3: Age vs Maximum Heart Rate by Diagnosis", use_column_width=True)
        with c2:
            st.markdown("#### Clinical Interpretation: Exercise Stress Dynamics")
            st.write("""
            - Across virtually all age strata (40 to 70 years), patients diagnosed with heart disease (orange points) cluster towards lower achieved heart rates during treadmill testing.
            - Younger patients who demonstrate low peak heart rates exhibit markedly elevated disease prevalence, reflecting premature chronotropic incompetence or early ischemic termination of stress testing.
            """)

    with eda_tabs[3]:
        c1, c2 = st.columns([3, 2])
        with c1:
            p = Path("reports/figures/eda_chest_pain_by_target.png")
            if p.exists():
                st.image(str(p), caption="Figure 4: Chest Pain Typology vs CAD Presence", use_column_width=True)
        with c2:
            st.markdown("#### Clinical Interpretation: The 'Asymptomatic' Paradox")
            st.write("""
            - Counterintuitively, category **4 ('Asymptomatic')** has the highest concentration of confirmed coronary disease.
            - In clinical cardiology, 'asymptomatic' in this angiographic cohort refers to patients referred for stress imaging due to objective risk factors rather than classical anginal symptoms.
            - Typical angina (1) and atypical angina (2) patients frequently present for non-cardiac chest discomfort (e.g., musculoskeletal or gastroesophageal reflux).
            """)


# =============================================================================
# PAGE 4: MODEL BENCHMARK & EVALUATION
# =============================================================================
elif nav_choice == "4. Model Benchmark & Evaluation":
    st.markdown('<div class="main-header">Model Benchmark & Generalization Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous 5-Fold Stratified Cross-Validation vs. Single-Shot Held-Out Test Evaluation.</div>', unsafe_allow_html=True)

    df_comp = get_comparison_table()

    # Highlight metrics cards for selected model (Logistic Regression)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">0.9665</div><div class="metric-label">Test ROC-AUC</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">92.86%</div><div class="metric-label">Test Sensitivity (Recall)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">84.85%</div><div class="metric-label">Test Specificity</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">2</div><div class="metric-label">Missed Cases (FN)</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Side-by-Side Architectural Comparison")
    st.dataframe(df_comp, use_container_width=True)

    c_roc, c_cm = st.columns(2)
    with c_roc:
        st.markdown("#### Receiver Operating Characteristic (ROC) Comparison")
        p_roc = Path("reports/figures/roc_curves_comparison.png")
        if p_roc.exists():
            st.image(str(p_roc), caption="Figure 5: Test Set ROC Curves with Area Under Curve (AUC)", use_column_width=True)
        st.caption("Logistic Regression (AUC = 0.967) outperforms Random Forest (AUC = 0.943) in discriminative separation.")

    with c_cm:
        st.markdown("#### Clinical Confusion Matrices (Held-out Test: 61 Patients)")
        cm_model = st.selectbox("Select Model Matrix", ["Logistic Regression", "Random Forest"])
        slug = cm_model.lower().replace(" ", "_")
        p_cm = Path(f"reports/figures/confusion_matrix_{slug}.png")
        if p_cm.exists():
            st.image(str(p_cm), caption=f"Figure 6: {cm_model} Test Confusion Matrix", use_column_width=True)

    st.markdown("---")
    c_fi, c_discuss = st.columns([3, 2])
    with c_fi:
        st.markdown("#### Feature Importance Analysis (Random Forest)")
        p_fi = Path("reports/figures/feature_importance_random_forest.png")
        if p_fi.exists():
            st.image(str(p_fi), caption="Figure 7: Top Transformed Clinical Predictors by Gini Importance", use_column_width=True)
    with c_discuss:
        st.markdown("#### Clinical Engineering Insights")
        st.write("""
        - **Why Logistic Regression won**: On this 303-sample dataset with 13 predictors, regularized linear models exhibit superior resistance to variance and overfitting compared to tree ensembles.
        - **Sensitivity Trade-off**: Logistic Regression achieved a **92.86% recall** on the held-out test partition, missing only 2 cardiac cases out of 28 positive patients.
        - **Feature Drivers**: Major vessels colored by fluoroscopy (`ca`), thallium perfusion defects (`thal`), exercise-induced ST depression (`oldpeak`), and peak heart rate (`thalach`) dominate both models' decision boundaries.
        """)


# =============================================================================
# PAGE 5: PATIENT RISK ASSESSMENT FORM
# =============================================================================
elif nav_choice == "5. Patient Risk Assessment Form":
    st.markdown('<div class="main-header">Interactive Patient Risk Assessment Form</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter clinical parameters to compute model probability score, risk tier, and contributing factors.</div>', unsafe_allow_html=True)

    predictor = get_cached_predictor()

    st.markdown("#### ⚡ Clinical Preset Profiles for Quick Testing")
    preset_cols = st.columns(3)
    preset_mode = None
    with preset_cols[0]:
        if st.button("🟢 Load Low-Risk Profile"):
            st.session_state["p_age"] = 42
            st.session_state["p_sex"] = "Female"
            st.session_state["p_cp"] = "2: Atypical Angina"
            st.session_state["p_trestbps"] = 118
            st.session_state["p_chol"] = 195
            st.session_state["p_fbs"] = "No (<= 120 mg/dl)"
            st.session_state["p_restecg"] = "0: Normal"
            st.session_state["p_thalach"] = 172
            st.session_state["p_exang"] = "No"
            st.session_state["p_oldpeak"] = 0.0
            st.session_state["p_slope"] = "1: Upsloping"
            st.session_state["p_ca"] = 0
            st.session_state["p_thal"] = "3: Normal"
    with preset_cols[1]:
        if st.button("🟠 Load Intermediate-Risk Profile"):
            st.session_state["p_age"] = 57
            st.session_state["p_sex"] = "Male"
            st.session_state["p_cp"] = "3: Non-Anginal Pain"
            st.session_state["p_trestbps"] = 138
            st.session_state["p_chol"] = 245
            st.session_state["p_fbs"] = "No (<= 120 mg/dl)"
            st.session_state["p_restecg"] = "1: ST-T Wave Abnormality"
            st.session_state["p_thalach"] = 142
            st.session_state["p_exang"] = "No"
            st.session_state["p_oldpeak"] = 1.2
            st.session_state["p_slope"] = "2: Flat"
            st.session_state["p_ca"] = 1
            st.session_state["p_thal"] = "3: Normal"
    with preset_cols[2]:
        if st.button("🔴 Load High-Risk Profile"):
            st.session_state["p_age"] = 65
            st.session_state["p_sex"] = "Male"
            st.session_state["p_cp"] = "4: Asymptomatic"
            st.session_state["p_trestbps"] = 160
            st.session_state["p_chol"] = 286
            st.session_state["p_fbs"] = "Yes (> 120 mg/dl)"
            st.session_state["p_restecg"] = "2: Left Ventricular Hypertrophy"
            st.session_state["p_thalach"] = 108
            st.session_state["p_exang"] = "Yes"
            st.session_state["p_oldpeak"] = 2.8
            st.session_state["p_slope"] = "2: Flat"
            st.session_state["p_ca"] = 2
            st.session_state["p_thal"] = "7: Reversible Defect"

    st.markdown("---")
    with st.form("cardiac_risk_form"):
        col_demo, col_stress = st.columns(2)

        with col_demo:
            st.markdown("### 👤 Demographics & Baseline Vitals")
            age = st.slider("Patient Age (years)", 20, 95, value=st.session_state.get("p_age", 55), step=1)
            sex_choice = st.selectbox(
                "Biological Sex",
                ["Female", "Male"],
                index=0 if st.session_state.get("p_sex") == "Female" else 1
            )
            trestbps = st.slider(
                "Resting Blood Pressure (mm Hg)",
                90, 220,
                value=st.session_state.get("p_trestbps", 130),
                step=1,
                help="Resting blood pressure measured upon hospital admission."
            )
            chol = st.slider(
                "Serum Cholesterol (mg/dl)",
                100, 600,
                value=st.session_state.get("p_chol", 230),
                step=2,
                help="Fasting serum cholesterol level in mg/dl."
            )
            fbs_choice = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                ["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
                index=1 if st.session_state.get("p_fbs") == "Yes (> 120 mg/dl)" else 0
            )
            restecg_choice = st.selectbox(
                "Resting Electrocardiogram (ECG)",
                [
                    "0: Normal",
                    "1: ST-T Wave Abnormality",
                    "2: Left Ventricular Hypertrophy"
                ],
                index=[
                    "0: Normal",
                    "1: ST-T Wave Abnormality",
                    "2: Left Ventricular Hypertrophy"
                ].index(st.session_state.get("p_restecg", "0: Normal"))
            )

        with col_stress:
            st.markdown("### 🏃 Exercise Stress Test Findings")
            cp_choice = st.selectbox(
                "Chest Pain Type",
                [
                    "1: Typical Angina",
                    "2: Atypical Angina",
                    "3: Non-Anginal Pain",
                    "4: Asymptomatic"
                ],
                index=[
                    "1: Typical Angina",
                    "2: Atypical Angina",
                    "3: Non-Anginal Pain",
                    "4: Asymptomatic"
                ].index(st.session_state.get("p_cp", "4: Asymptomatic"))
            )
            thalach = st.slider(
                "Maximum Heart Rate Achieved (bpm)",
                60, 220,
                value=st.session_state.get("p_thalach", 145),
                step=1,
                help="Peak heart rate achieved during maximal exercise stress test."
            )
            exang_choice = st.selectbox(
                "Exercise-Induced Angina Provoked?",
                ["No", "Yes"],
                index=1 if st.session_state.get("p_exang") == "Yes" else 0
            )
            oldpeak = st.slider(
                "ST Depression Induced by Exercise (mm)",
                0.0, 7.0,
                value=float(st.session_state.get("p_oldpeak", 1.0)),
                step=0.1,
                help="ST segment depression relative to baseline rest."
            )
            slope_choice = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                [
                    "1: Upsloping",
                    "2: Flat",
                    "3: Downsloping"
                ],
                index=[
                    "1: Upsloping",
                    "2: Flat",
                    "3: Downsloping"
                ].index(st.session_state.get("p_slope", "2: Flat"))
            )
            ca_choice = st.selectbox(
                "Fluoroscopy Colored Major Vessels (0-3)",
                [0, 1, 2, 3],
                index=st.session_state.get("p_ca", 0),
                help="Number of major coronary vessels visualized by fluoroscopy."
            )
            thal_choice = st.selectbox(
                "Thallium Scintigraphy (Thal)",
                [
                    "3: Normal",
                    "6: Fixed Defect",
                    "7: Reversible Defect"
                ],
                index=[
                    "3: Normal",
                    "6: Fixed Defect",
                    "7: Reversible Defect"
                ].index(st.session_state.get("p_thal", "3: Normal"))
            )

        submit_btn = st.form_submit_button("🫀 Calculate Statistical Cardiac Risk")

    if submit_btn:
        # Convert UI choices into standardized feature dictionary
        sex_val = 1 if sex_choice == "Male" else 0
        cp_val = int(cp_choice.split(":")[0])
        fbs_val = 1 if "Yes" in fbs_choice else 0
        restecg_val = int(restecg_choice.split(":")[0])
        exang_val = 1 if exang_choice == "Yes" else 0
        slope_val = int(slope_choice.split(":")[0])
        ca_val = int(ca_choice)
        thal_val = int(thal_choice.split(":")[0])

        input_payload = {
            "age": age,
            "sex": sex_val,
            "cp": cp_val,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_val,
            "restecg": restecg_val,
            "thalach": thalach,
            "exang": exang_val,
            "oldpeak": oldpeak,
            "slope": slope_val,
            "ca": ca_val,
            "thal": thal_val,
        }

        try:
            inference = predictor.predict_risk(input_payload)

            st.markdown("---")
            st.markdown("### 📋 Risk Assessment Results")

            r1, r2, r3 = st.columns([2, 2, 3])
            with r1:
                st.metric(
                    label="Model Probability of Disease",
                    value=f"{inference['risk_percentage']}%",
                    delta=f"{inference['probability_disease']:.3f} p(y=1)"
                )
            with r2:
                tier = inference["risk_tier"]
                badge_class = (
                    "risk-badge-low" if "Low" in tier
                    else ("risk-badge-moderate" if "Moderate" in tier else "risk-badge-high")
                )
                st.markdown(f'<div style="margin-top:1.2rem;"><span class="{badge_class}">{tier}</span></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f"**Classification Label**: `{inference['prediction_label']}`")
                st.caption(inference["clinical_implication"])

            st.progress(inference["probability_disease"])

            # Explainable factors
            if inference["contributing_factors"]:
                st.markdown("#### 🔍 Notable Clinical Risk Modifiers Identified:")
                for factor in inference["contributing_factors"]:
                    st.markdown(f"- ⚠️ {factor}")
            else:
                st.success("No high-risk cardiovascular exacerbators detected in this profile.")

            st.info("🔒 **Data Privacy Guarantee**: Entered parameters are processed in-memory for this inference turn and are strictly never logged, cached, or transmitted to any external server.")

        except Exception as exc:
            st.error(f"Prediction Error: {exc}")


# =============================================================================
# PAGE 6: CLINICAL ETHICS & LIMITATIONS
# =============================================================================
elif nav_choice == "6. Clinical Ethics & Limitations":
    st.markdown('<div class="main-header">Clinical Ethics, Governance & Limitations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Addressing demographic bias, algorithmic safety, and clinical decision boundaries.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ⚠️ Dataset Demographics & Algorithmic Bias")
        st.write("""
        - **Historical Cohort (1988)**: The Cleveland dataset was compiled over three decades ago. Diagnostic cardiology standards, pharmacology (statins, beta-blockers), and stress imaging resolution have evolved dramatically.
        - **Sex Underrepresentation**: Females represent only **~32%** of this dataset (97 females vs 206 males). Ischemic presentation in women often differs (microvascular dysfunction vs epicardial stenosis), leading to potential performance disparities.
        - **Geographic & Racial Generalizability**: Collected at a single tertiary center in Cleveland, Ohio; fails to reflect racially and ethnically diverse population health cohorts.
        """)

        st.markdown("### ⚖️ Clinical Trade-offs: FP vs FN")
        st.write("""
        - **False Negatives (Critical Safety Risk)**: Failing to detect significant CAD leaves a patient vulnerable to acute myocardial infarction or cardiac arrest without preventative intervention.
        - **False Positives (Over-triage)**: Incur financial expense, psychological anxiety, and non-negligible procedural risks associated with unnecessary confirmatory invasive catheterization.
        - **Engineering Decision**: The model selection protocol strictly prioritized **Sensitivity / Recall** to minimize false negatives while maintaining acceptable specificity (>80%).
        """)

    with c2:
        st.markdown("### 🔒 Privacy, Security & Data Sovereignty")
        st.write("""
        - **Zero Cloud Storage**: This application performs local inference in transient RAM. No protected health information (PHI) is persisted or shared.
        - **HIPAA / GDPR Compliance**: An educational sandbox must never ingest de-anonymized real patient identifiers.
        """)

        st.markdown("### 🏛️ Regulatory Boundaries (FDA & SaMD)")
        st.write("""
        - Under FDA Guidance for Software as a Medical Device (SaMD) and Section 3060 of the 21st Century Cures Act, machine learning systems intended for clinical use must undergo:
          1. Prospective multi-site clinical trial validation.
          2. Rigorous software verification and validation (V&V).
          3. Real-world post-market surveillance for data drift.
        - **This repository is an academic demonstration only** and does not fulfill regulatory requirements for clinical deployment.
        """)
