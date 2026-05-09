# Noisy-Data-Handler-ML-Pipeline
ML pipeline in Python &amp; Scikit-Learn to detect, clean, and handle noisy real-world datasets — featuring outlier detection (IQR, Z-Score, Isolation Forest), missing value imputation (KNN, MICE), feature engineering, and automated model evaluation reporting.
# 🧹 Noisy Data Handler — ML Pipeline

> A production-grade Python pipeline for preprocessing and cleaning noisy real-world datasets before feeding them into machine learning models.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Overview

Real-world datasets are messy — they contain missing values, outliers, duplicate records, inconsistent types, and skewed distributions. This project provides a **modular, extensible ML pipeline** that handles all forms of data noise before training, ensuring better model accuracy and robustness.

### Key Capabilities

| Module | Technique |
|--------|-----------|
| **Data Ingestion** | CSV, JSON, Excel, SQL support |
| **Imputation** | Mean, Median, Mode, KNN, Iterative |
| **Outlier Detection** | IQR, Z-Score, Isolation Forest, LOF |
| **Feature Engineering** | Encoding, Scaling, Polynomial, Interaction |
| **Pipeline Automation** | Scikit-Learn custom transformers |
| **Model Training** | RandomForest, GradientBoosting, LogReg |
| **Evaluation & Reporting** | Classification/Regression metrics + HTML report |

---

## 🗂️ Project Structure

```
noisy-data-handler/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Multi-format data ingestion
│   ├── preprocessor.py         # Core cleaning & preprocessing
│   ├── outlier_detector.py     # Outlier detection strategies
│   ├── imputer.py              # Advanced missing-value imputation
│   ├── feature_engineer.py     # Feature creation & transformation
│   ├── pipeline.py             # Full end-to-end sklearn pipeline
│   └── model.py                # Model training, tuning & evaluation
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_outlier_detector.py
│   ├── test_imputer.py
│   └── test_pipeline.py
│
├── config/
│   └── config.yaml             # Centralized configuration
│
├── data/
│   ├── raw/                    # Raw input datasets
│   └── processed/              # Cleaned output datasets
│
├── notebooks/
│   └── exploration.ipynb       # EDA notebook
│
├── reports/                    # Auto-generated HTML reports
├── main.py                     # CLI entry point
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/25Satyam/noisy-data-handler.git
cd noisy-data-handler

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run the full pipeline on your own CSV:

```bash
python main.py --input data/raw/your_dataset.csv --target target_column --task classification
```

### Run with a built-in demo dataset:

```bash
python main.py --demo --task classification
```

### Available CLI arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input CSV file | — |
| `--target` | Name of the target column | `target` |
| `--task` | `classification` or `regression` | `classification` |
| `--demo` | Use built-in noisy demo dataset | `False` |
| `--output` | Save cleaned data to path | `data/processed/` |
| `--report` | Generate HTML evaluation report | `True` |

---

## 🔬 Pipeline Stages

### 1. Data Loading
Supports CSV, JSON, Excel (.xlsx), and Parquet formats with automatic type inference.

### 2. Basic Cleaning
- Remove duplicate rows
- Fix mixed-type columns
- Standardize column names
- Drop columns exceeding missing-value threshold

### 3. Missing Value Imputation
- **Numeric:** Mean, Median, KNN Imputer, Iterative Imputer (MICE)
- **Categorical:** Mode, constant fill, most-frequent

### 4. Outlier Detection & Treatment
- **IQR Method** — clips values beyond Q1/Q3 ± 1.5×IQR
- **Z-Score** — flags points beyond 3 standard deviations
- **Isolation Forest** — tree-based anomaly detection
- **Local Outlier Factor (LOF)** — density-based detection

### 5. Feature Engineering
- Label encoding / One-Hot encoding for categoricals
- StandardScaler / MinMaxScaler / RobustScaler for numerics
- Log/sqrt transforms for skewed distributions
- Polynomial and interaction feature generation

### 6. Model Training
- Supports RandomForest, GradientBoosting, LogisticRegression
- GridSearchCV / RandomizedSearchCV for hyperparameter tuning
- Cross-validated scoring

### 7. Evaluation & Reporting
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Regression: MAE, MSE, RMSE, R²
- Auto-generates an HTML report in `reports/`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Example Results

After applying the pipeline on a noisy version of the Breast Cancer dataset:

| Metric | Before Cleaning | After Pipeline |
|--------|----------------|----------------|
| Accuracy | 0.81 | **0.96** |
| F1 Score | 0.79 | **0.95** |
| ROC-AUC | 0.83 | **0.98** |

---

## 🛠️ Configuration

Edit `config/config.yaml` to tune every stage without touching code:

```yaml
imputation:
  numeric_strategy: knn
  n_neighbors: 5

outlier_detection:
  method: isolation_forest
  contamination: 0.05

feature_engineering:
  scaling: robust
  encode_categoricals: true
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Satyam**
- GitHub: [@25Satyam](https://github.com/25Satyam)
- LinkedIn: [Satyam](https://linkedin.com/in/satyam-423376244/)
