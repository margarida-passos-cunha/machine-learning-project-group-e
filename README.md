# Crop Recommendation System — Group E

> Supervised multi-class classification for precision agriculture

---

## Group Members

| # | Name | Student ID |
|---|------|------------|
| 1 | <!-- Name --> | <!-- ID --> |
| 2 | <!-- Name --> | <!-- ID --> |
| 3 | <!-- Name --> | <!-- ID --> |
| 4 | <!-- Name --> | <!-- ID --> |
| 5 | <!-- Name --> | <!-- ID --> |
| 6 | <!-- Name --> | <!-- ID --> |

---

## Overview

This project builds a data-driven crop recommendation system that predicts the most suitable crop to grow given soil nutrient levels and environmental conditions. It is a **supervised multi-class classification** problem with **22 crop classes**.

The workflow covers the full ML pipeline: exploratory data analysis, domain-informed feature engineering, model training and hyperparameter tuning, cross-validated evaluation, and interpretability analysis with SHAP.

---

## Dataset

**Source:** `data/raw/Crop_recommendation.csv`

| Property | Value |
|----------|-------|
| Samples | 2,200 |
| Features | 7 (original) + 3 (engineered) = 10 |
| Target classes | 22 crops |
| Missing values | None |
| Duplicates | None |

**Input features:**

| Feature | Description |
|---------|-------------|
| `N` | Nitrogen content in soil |
| `P` | Phosphorus content in soil |
| `K` | Potassium content in soil |
| `temperature` | Temperature (°C) |
| `humidity` | Relative humidity (%) |
| `ph` | Soil pH value |
| `rainfall` | Rainfall (mm) |

**Target:** `label` — crop type (22 classes: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee)

---

## Project Structure

```
machine-learning-project-group-e/
├── data/
│   ├── raw/
│   │   └── Crop_recommendation.csv       # Original dataset
│   └── processed/
│       └── crop_features_v1.csv          # Dataset with engineered features
├── notebooks/
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb      # Feature Engineering & Selection
│   └── 03_modelling.ipynb                # Model Training & Evaluation
├── reports/
│   └── figures/                          # Output visualizations
├── src/
│   ├── utils.py
│   ├── preprocessing.py
│   └── modeling.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Notebooks

Run the notebooks in order:

### [01 — Exploratory Data Analysis](notebooks/01_eda.ipynb)

- Data loading and integrity checks
- Feature distributions (univariate and bivariate)
- Outlier detection and class balance assessment
- Correlation analysis and pairplots
- Output: 8 visualizations saved to `reports/figures/`

### [02 — Feature Engineering](notebooks/02_feature_engineering.ipynb)

Three agronomically motivated features are engineered and validated:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `moisture_index` | `log(1 + rainfall × humidity)` | Combines water availability signals; log reduces skew |
| `P_to_K` | `P / (K + ε)` | Nutrient balance matters more than absolute amounts |
| `ph_dev` | `\|pH − 6.5\|` | Deviation from the optimal pH zone (~6.5) for most crops |

**Ablation study results (Macro-F1):**

| Features | Macro-F1 |
|----------|----------|
| Base (7 features) | 0.9707 |
| + `moisture_index` | 0.9725 (+0.0018) |
| + `P_to_K` | 0.9713 (+0.0006) |
| + `ph_dev` | 0.9750 (+0.0043) |
| + all 3 engineered | **0.9795 (+0.0088)** |

Feature selection via SelectKBest (mutual information) confirms all 10 features are worth retaining.

### [03 — Modelling](notebooks/03_modelling.ipynb)

Four classifiers are trained, each in an untuned and a hyperparameter-tuned version:

| Model | Notes |
|-------|-------|
| Logistic Regression | Multi-class baseline; C tuned via GridSearchCV |
| Support Vector Machine (SVM) | RBF kernel; C and gamma tuned |
| Decision Tree | Interpretable baseline; depth and split criteria tuned |
| Random Forest | Best overall performer; trees, depth, and splits tuned |

**Evaluation protocol:**
- 80/20 train-test split
- `StandardScaler` applied within a `Pipeline`
- `RepeatedStratifiedKFold` cross-validation
- Metrics: accuracy, precision, recall, macro-F1, confusion matrix
- Interpretability: SHAP values for feature contribution analysis

**Key results:**
- All four models achieve **>97% accuracy**
- **Random Forest** is the best-performing model
- Top predictive features: humidity, rainfall, potassium (K)
- `ph_dev` is the strongest single engineered feature

---

## Setup & Usage

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap jupyter
```

### Running the project

```bash
# Clone the repository
git clone <repo-url>
cd machine-learning-project-group-e

# Launch Jupyter
jupyter notebook
```

Then open and run the notebooks in order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_modelling.ipynb`

Each notebook is self-contained and can be run with **Run All Cells**.

---

## Results Summary

| Model | Tuned | Macro-F1 |
|-------|-------|----------|
| Logistic Regression | No | ~0.97 |
| Logistic Regression | Yes | ~0.97+ |
| SVM | No | ~0.98 |
| SVM | Yes | ~0.98+ |
| Decision Tree | No | ~0.97 |
| Decision Tree | Yes | ~0.97+ |
| Random Forest | No | ~0.98 |
| Random Forest | Yes | **~0.98+** |

All models produce near-perfect diagonal confusion matrices across all 22 crop classes, indicating strong generalization.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
