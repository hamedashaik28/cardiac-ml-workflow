# Dataset Documentation: UCI Heart Disease (Cleveland Database)

## 1. Provenance and Citation
- **Source**: UCI Machine Learning Repository
- **URL**: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Direct Data File**: [https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
- **Principal Investigators**:
  - Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
  - University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
  - University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
  - V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
- **Citation**:
  > Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J. J., Sandhu, S., ... & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.

---

## 2. Dataset Overview & Dimensions
- **Original Samples**: 303 patient records.
- **Attributes in Processed File**: 14 (13 clinical predictors + 1 diagnosis target).
- **Missing Value Representation**: Represented as `'?'` in raw records (found in `ca` [4 rows] and `thal` [2 rows], total 6 rows affected).

---

## 3. Clinical Data Dictionary

| # | Attribute | Name in Pipeline | Type | Unit / Description / Clinical Encoding |
|---|---|---|---|---|
| 1 | `age` | `age` | Numerical | Age in years (range: 29 – 77) |
| 2 | `sex` | `sex` | Categorical / Binary | 1 = male; 0 = female |
| 3 | `cp` | `cp` | Categorical | Chest pain type: <br>1 = Typical angina <br>2 = Atypical angina <br>3 = Non-anginal pain <br>4 = Asymptomatic |
| 4 | `trestbps` | `trestbps` | Numerical | Resting blood pressure in mm Hg upon hospital admission (range: 94 – 200) |
| 5 | `chol` | `chol` | Numerical | Serum cholesterol in mg/dl (range: 126 – 564) |
| 6 | `fbs` | `fbs` | Categorical / Binary | Fasting blood sugar > 120 mg/dl: <br>1 = true; 0 = false |
| 7 | `restecg` | `restecg` | Categorical | Resting electrocardiographic results: <br>0 = Normal <br>1 = Having ST-T wave abnormality (T wave inversions and/or ST elevation/depression > 0.05 mV) <br>2 = Showing probable or definite left ventricular hypertrophy by Estes' criteria |
| 8 | `thalach` | `thalach` | Numerical | Maximum heart rate achieved during exercise stress test (range: 71 – 202) |
| 9 | `exang` | `exang` | Categorical / Binary | Exercise induced angina: <br>1 = yes; 0 = no |
| 10 | `oldpeak` | `oldpeak` | Numerical | ST depression induced by exercise relative to rest (range: 0.0 – 6.2 mm) |
| 11 | `slope` | `slope` | Categorical | Slope of peak exercise ST segment: <br>1 = Upsloping <br>2 = Flat <br>3 = Downsloping |
| 12 | `ca` | `ca` | Categorical / Discrete | Number of major vessels (0–3) colored by fluoroscopy (contains `?` missing values) |
| 13 | `thal` | `thal` | Categorical | Thallium heart scan (scintigraphy): <br>3 = Normal <br>6 = Fixed defect <br>7 = Reversible defect (contains `?` missing values) |
| 14 | `num` / `target` | `target` | Binary Target | Angiographic disease status: <br>**0**: Absence (< 50% diameter stenosis) <br>**1**: Presence (> 50% diameter stenosis, converted from raw values 1, 2, 3, 4) |

---

## 4. Target Transformation Rule
The raw column `num` contains values `0, 1, 2, 3, 4` indicating severity of stenosis.
In this workflow, we convert `num` to a binary classification target:
$$\text{target} = \begin{cases} 0 & \text{if } \text{num} = 0 \quad (\text{No coronary artery disease}) \\ 1 & \text{if } \text{num} \ge 1 \quad (\text{Coronary artery disease present}) \end{cases}$$
The original target and any target-derived fields are strictly excluded from input feature matrices to avoid label leakage.

---

## 5. Automated Loading & Manual Fallback Instructions

The script `src/data_loader.py` attempts to automatically download the raw dataset from the official UCI repository over HTTPS and caches it to:
`data/raw/processed.cleveland.data`

### Manual Fallback:
If the automatic download is blocked by a firewall, offline environment, or proxy:
1. Open your browser and navigate to:
   `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
2. Save the file exactly as:
   `data/raw/processed.cleveland.data`
3. Rerun `python src/train.py`. The data loader checks for local file existence before initiating any network request.
