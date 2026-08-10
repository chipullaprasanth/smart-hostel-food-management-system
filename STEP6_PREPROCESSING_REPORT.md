# STEP 6: PREPROCESSING REPORT FOR APPROVED DATASET

**Project Title:** Smart Hostel Food Waste Management System  
**Approved Dataset:** Messy Food Waste Prediction Dataset (`datasets/messy_food_waste/`)  
**Date:** August 10, 2026  
**Status:** Step 6 Completed (Preprocessing Pipeline Fitted & Saved — No Models Trained)

---

## 1. Executive Summary & Rules Integrity Notice

In Step 6, the approved **Messy Food Waste Prediction Dataset** (`train.csv` and `test.csv`) was preprocessed using a clean, reproducible scikit-learn pipeline saved to [`model/final_preprocessor.pkl`](file:///c:/Users/jalas/project/smart-hostel-food-management-system/model/final_preprocessor.pkl).

### Confirmation of Project Rules:
- **`Dataset Propely.csv` was NOT modified or merged.**
- **Step 3 baseline benchmark results on `Dataset Propely.csv` remain completely unchanged:**
  - Linear Regression: MAE = `1.2158 kg` | RMSE = `1.3916 kg` | $R^2 = 0.0019$ (`0.19%`)
  - Random Forest: MAE = `1.3190 kg` | RMSE = `1.5416 kg` | $R^2 = -0.2248$ (`-22.48%`)
  - Decision Tree: MAE = `1.6775 kg` | RMSE = `2.0476 kg` | $R^2 = -1.1609$ (`-116.09%`)
- **Step 3 pipeline artifact (`model/preprocessor.pkl`) was preserved untouched.**
- **`test.csv` was NOT used to fit preprocessing parameters.**
- **No synthetic data was generated.**
- **No machine learning models were trained in Step 6.**
- **Original CSV files were preserved unchanged.**

---

## 2. Dataset Dimensions Summary

| Dataset Split | Raw Input Shape | Processed Feature Matrix Shape | Target Variable Present? |
| :--- | :---: | :---: | :---: |
| **Training Data (`train.csv`)** | **(1000, 11)** | **(1000, 19)** | YES (`food_waste_kg`) |
| **Testing Data (`test.csv`)** | **(200, 10)** | **(200, 19)** | NO (Unseen evaluation test) |

---

## 3. Target Variable Analysis (`food_waste_kg`)

The target variable is `food_waste_kg` present in `train.csv`:
- **Measurement Unit:** Physical food waste in kilograms (kg)
- **Minimum Value:** `5.71 kg`
- **Maximum Value:** `42.72 kg`
- **Mean Value:** `23.89 kg`
- **Median Value:** `23.83 kg`
- **Standard Deviation:** `7.42 kg`
- **Missing Values:** `0` missing values
- **Target Leakage Note:** `food_waste_kg` was strictly excluded from feature matrix $X$.

---

## 4. Feature Engineering & Preprocessing Transformations

### 4.1 Date Feature Engineering
The ISO string column `date` (spanning `2024-01-01` to `2026-09-26`) was converted to datetime format. Five calendar predictors were extracted:
1. `year`: Integer (2024 to 2026)
2. `month`: Integer (1 to 12)
3. `day`: Integer (1 to 31)
4. `day_of_week`: Integer (0 = Monday to 6 = Sunday)
5. `is_weekend`: Binary (1 if `day_of_week` $\ge 5$, else 0)

### 4.2 String-to-Numeric Conversions
Raw text columns containing embedded numbers were parsed using regex numeric extraction:
- `meals_served`: Extracted string patterns like `"252 meals"` $\rightarrow$ `meals_served_numeric` (`252.0`).
- `kitchen_staff`: Extracted string patterns like `"8 staff"` $\rightarrow$ `kitchen_staff_numeric` (`8.0`).

### 4.3 Categorical Normalization
Leading/trailing whitespace was stripped and casing was normalized to title-case for categorical features:
- `staff_experience`: Normalized `"EXPERT"`, `"Beginner "`, `"Beginner"`, `"intermediate"` $\rightarrow$ 3 clean categories: `"Expert"`, `"Beginner"`, `"Intermediate"`.
- `waste_category`: Normalized `"dairy"`, `"Bakery"`, `"Rice"`, `"VEGETABLES"`, `"MeAt"` $\rightarrow$ 5 clean categories: `"Dairy"`, `"Bakery"`, `"Rice"`, `"Vegetables"`, `"Meat"`.

---

## 5. Missing-Value Imputation (`temperature_C`)

- **Missing Count:** `temperature_C` contained `80` missing values (8.0% missingness) in `train.csv`.
- **Imputation Strategy:** `SimpleImputer(strategy="median")`.
- **Fit Discipline:** The imputer was **FIT ONLY on `X_train_raw`** (training median: `24.82°C`). `test.csv` was transformed using the training-derived median.

---

## 6. Temperature Outlier Investigation

`temperature_C` contains extreme values in the training set:
- **`-10.0°C`**: 15 observations
- **`60.0°C`**: 15 observations

### Outlier Handling Decision:
These observations represent recorded sensor/environmental extremes in the dataset. Per instructions, these values were **retained without row deletion or arbitrary modification** to allow tree-based and linear models in Step 7 to evaluate realistic operational robustness. Robust median imputation and standard scaling prevent scaling distortion.

---

## 7. Categorical One-Hot Encoding

Categorical variables were encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` **FIT ONLY on training data**:
- **`staff_experience` (3 features):** `staff_experience_Beginner`, `staff_experience_Expert`, `staff_experience_Intermediate`.
- **`waste_category` (5 features):** `waste_category_Bakery`, `waste_category_Dairy`, `waste_category_Meat`, `waste_category_Rice`, `waste_category_Vegetables`.

Total One-Hot Encoded Features = `8 columns`.

---

## 8. Final Processed Feature List (19 Features)

Below is the complete list of 19 input features passed to the models:

### Numerical & Engineered Features (11 Features)
1. `meals_served_numeric`
2. `kitchen_staff_numeric`
3. `temperature_C`
4. `humidity_percent`
5. `past_waste_kg`
6. `year`
7. `month`
8. `day`
9. `day_of_week`
10. `is_weekend`
11. `special_event`

### One-Hot Encoded Categorical Features (8 Features)
12. `staff_experience_Beginner`
13. `staff_experience_Expert`
14. `staff_experience_Intermediate`
15. `waste_category_Bakery`
16. `waste_category_Dairy`
17. `waste_category_Meat`
18. `waste_category_Rice`
19. `waste_category_Vegetables`

---

## 9. Past Waste & Target Leakage Investigation

### Past Waste Analysis (`past_waste_kg`)
- `past_waste_kg` represents historical food waste from the *previous* operational cycle/day.
- It is a standard autoregressive input feature (range: `5.05 kg` to `44.99 kg`, mean: `25.23 kg`).
- **Leakage Status:** NO target leakage. It provides historical context without leaking current-meal target values.

### Explicit Leakage Checks Summary:
- **Direct Target Duplication:** Verified absent.
- **Target-derived Features:** Verified absent.
- **Test Set Data Leakage:** `preprocessor` was fitted **EXCLUSIVELY** on `X_train_raw`. Zero test set information was used during fitting.
- **Target Inclusion in X:** Verified `food_waste_kg` is completely excluded from feature matrix $X$.

---

## 10. Recommended Validation Strategy for Step 7

For model selection and hyperparameter tuning inside `train.csv` during Step 7:
- **Primary Recommendation:** **`TimeSeriesSplit(n_splits=5)`** or **5-Fold Cross-Validation** on `X_train_proc` and `y_train`.
- **Test Set Protection:** `test.csv` (200 records) remains completely untouched and unseen until final model evaluation in Step 7.

---

## 11. Reproducible Scikit-Learn Pipeline & Artifacts

The preprocessing pipeline is codified in [`preprocess_step6.py`](file:///c:/Users/jalas/project/smart-hostel-food-management-system/preprocess_step6.py):

```python
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
```

- **Saved Pipeline Artifact:** [`model/final_preprocessor.pkl`](file:///c:/Users/jalas/project/smart-hostel-food-management-system/model/final_preprocessor.pkl)
- **Step 3 Benchmark Pipeline Preserved:** [`model/preprocessor.pkl`](file:///c:/Users/jalas/project/smart-hostel-food-management-system/model/preprocessor.pkl)

---

## 12. Final Status Statement

STEP 6 PREPROCESSING COMPLETE — READY FOR STEP 7 MODEL TRAINING.
