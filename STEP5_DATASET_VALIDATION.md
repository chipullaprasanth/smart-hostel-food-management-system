# STEP 5: CANDIDATE DATASET VALIDATION REPORT

**Project Title:** Smart Hostel Food Waste Management System  
**Dataset Under Validation:** Messy Food Waste Prediction Dataset  
**Storage Path:** `datasets/messy_food_waste/`  
**Date:** August 10, 2026  
**Status:** Step 5 Completed (Dataset Downloaded and Validated — No Model Retraining Executed)

---

## 1. Executive Summary & Benchmark Integrity Notice

As required by project instructions, the Step 3 baseline benchmark results on `Dataset Propely.csv` remain **completely untouched and unchanged**:

### Step 3 Baseline Benchmark Results (UNTOUCHED):
- **Linear Regression:** MAE = `1.2158 kg` | RMSE = `1.3916 kg` | $R^2 = 0.0019$ (`0.19%`)
- **Random Forest Regressor:** MAE = `1.3190 kg` | RMSE = `1.5416 kg` | $R^2 = -0.2248$ (`-22.48%`)
- **Decision Tree Regressor:** MAE = `1.6775 kg` | RMSE = `2.0476 kg` | $R^2 = -1.1609$ (`-116.09%`)

The candidate dataset **Messy Food Waste Prediction Dataset** has been isolated in `datasets/messy_food_waste/`. `Dataset Propely.csv` was **NOT** modified or overwritten, no datasets were merged, no synthetic data was generated, and no machine learning models were trained.

---

## 2. 17-Point Detailed Dataset Inspection & Validation

### Point 1: Exact Filename(s)
- `datasets/messy_food_waste/train.csv`
- `datasets/messy_food_waste/test.csv`

### Point 2: Number of Records
- **Training Set (`train.csv`):** 1,000 records
- **Testing Set (`test.csv`):** 200 records

### Point 3: Number of Columns
- **Training Set (`train.csv`):** 11 columns
- **Testing Set (`test.csv`):** 10 columns (Target column `food_waste_kg` omitted for prediction evaluation)

### Point 4: Exact Column Names
- **`train.csv`:** `date`, `meals_served`, `kitchen_staff`, `temperature_C`, `humidity_percent`, `day_of_week`, `special_event`, `past_waste_kg`, `staff_experience`, `waste_category`, `food_waste_kg`
- **`test.csv`:** `date`, `meals_served`, `kitchen_staff`, `temperature_C`, `humidity_percent`, `day_of_week`, `special_event`, `past_waste_kg`, `staff_experience`, `waste_category`

### Point 5: Data Types
- `date`: `object` / `string`
- `meals_served`: `object` / `string` (Uncleaned, e.g. `"252 meals"`, `"498 meals"`)
- `kitchen_staff`: `object` / `string` (Uncleaned, e.g. `"8 staff"`, `"11 staff"`)
- `temperature_C`: `float64`
- `humidity_percent`: `float64`
- `day_of_week`: `int64` (0 = Monday to 6 = Sunday)
- `special_event`: `int64` (Binary 0/1)
- `past_waste_kg`: `float64`
- `staff_experience`: `object` / `string` (Categorical, mixed casing)
- `waste_category`: `object` / `string` (Categorical, mixed casing)
- `food_waste_kg`: `float64` (Target variable)

### Point 6: Missing-Value Count (Every Column in `train.csv`)
| Column Name | Missing Count | Missing Percentage |
| :--- | :---: | :---: |
| `date` | 0 | 0.0% |
| `meals_served` | 0 | 0.0% |
| `kitchen_staff` | 0 | 0.0% |
| `temperature_C` | **80** | **8.0%** |
| `humidity_percent` | 0 | 0.0% |
| `day_of_week` | 0 | 0.0% |
| `special_event` | 0 | 0.0% |
| `past_waste_kg` | 0 | 0.0% |
| `staff_experience` | 0 | 0.0% |
| `waste_category` | 0 | 0.0% |
| `food_waste_kg` | 0 | 0.0% |

### Point 7: Duplicate-Row Count
- `0` duplicate rows found in `train.csv`.

### Point 8: Summary Statistics for Numerical Columns (Train Set)
| Column Name | Minimum | Maximum | Mean | Median |
| :--- | :---: | :---: | :---: | :---: |
| `temperature_C` | `-10.00°C` | `60.00°C` | `24.90°C` | `24.82°C` |
| `humidity_percent` | `30.10%` | `89.99%` | `60.48%` | `61.08%` |
| `day_of_week` | `0` | `6` | `3.00` | `3.00` |
| `special_event` | `0` | `1` | `0.15` | `0.00` |
| `past_waste_kg` | `5.05 kg` | `44.99 kg` | `25.23 kg` | `25.47 kg` |
| **`food_waste_kg` (Target)** | **`5.71 kg`** | **`42.72 kg`** | **`23.89 kg`** | **`23.83 kg`** |

### Point 9: Unique Values for Categorical Columns
- **`date`:** 1,000 unique daily string timestamps.
- **`meals_served`:** 398 unique string entries (e.g. `"252 meals"`, `"498 meals"`).
- **`kitchen_staff`:** 30 unique string entries (e.g. `"8 staff"`, `"11 staff"`).
- **`staff_experience`:** 4 unique entries (`"EXPERT"`, `"Beginner "`, `"Beginner"`, `"intermediate"`).
- **`waste_category`:** 5 unique entries (`"dairy"`, `"Bakery"`, `"Rice"`, `"VEGETABLES"`, `"MeAt"`).

### Point 10: Date / Time Range
- Date range: `2024-01-01` to `2026-09-26` (1,000 consecutive daily records).

### Point 11: Exact Target Column
- **`food_waste_kg`**

### Point 12: Target Measurement Verification
- **YES.** `food_waste_kg` is a continuous physical quantity measuring daily food waste in kilograms.

### Point 13: `Meals_Served` Presence Verification
- **YES.** `meals_served` is present as an attendance proxy feature. String parsing (regex extraction) will be required in Step 6 to convert formatted strings like `"252 meals"` into numeric counts.

### Point 14: Temperature & Humidity Presence Verification
- **YES.** Both `temperature_C` and `humidity_percent` are present as environmental weather predictors.

### Point 15: Target Leakage Verification
- **NO TARGET LEAKAGE.** No feature in `train.csv` is derived directly from `food_waste_kg` (such as `cost_loss` or `waste_ratio`). `past_waste_kg` represents historical waste from the prior cycle, which is a standard autoregressive input feature.

### Point 16: Operational Suitability for Hostel/Canteen Food Waste
- **YES.** This dataset directly captures cafeteria/mess attendance (`meals_served`), staff capacity (`kitchen_staff`), environmental weather conditions (`temperature_C`, `humidity_percent`), special events (`special_event`), and item categories (`waste_category`), making it highly suitable for predicting food waste in an institutional setting.

### Point 17: Suspicious, Synthetic, or Data Cleaning Artifacts
- The dataset intentionally contains "messy" data quality artifacts designed to test data preprocessing pipelines:
  1. String-formatted numbers (`"252 meals"`, `"8 staff"`).
  2. Mixed text casing and whitespace (`"VEGETABLES"`, `"MeAt"`, `"Beginner "`).
  3. Extreme temperature outliers (`-10.0°C` and `60.0°C`).
  4. 8.0% missing values in `temperature_C`.

These issues do not prevent training but will require robust cleaning in Step 6.

---

## 3. Final Validation Verdict

```text
FINAL VERDICT: A. APPROVED FOR ML TRAINING
```

**Reason for Approval:**  
The dataset satisfies all feature requirements identified in Step 4A and Step 4B. It provides attendance headcount proxies (`meals_served`), environmental weather predictors (`temperature_C`, `humidity_percent`), kitchen staffing, and daily food waste in kilograms (`food_waste_kg`) across 1,000 records.

---

## 4. Constraint Compliance Statement

- **`Dataset Propely.csv` was NOT modified or overwritten.**
- **No datasets were merged.**
- **No synthetic data was generated.**
- **No student attendance or weather numbers were fabricated.**
- **No ML models were trained in Step 5.**
- **Step 3 baseline metrics remain unchanged.**

*Stopped as instructed after completing dataset download, 17-point validation, and report creation. Waiting for Step 6.*
