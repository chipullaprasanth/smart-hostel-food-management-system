# STEP 2: DATASET PREPROCESSING REPORT

**Project Title:** Smart Hostel Food Waste Management System  
**Selected Dataset:** `Dataset Propely.csv`  
**Date:** August 10, 2026  
**Status:** Step 2 Completed (Preprocessing & Pipeline Construction Only — No ML Models Trained Yet)

---

## 1. Executive Summary

In Step 2, `Dataset Propely.csv` was processed strictly for machine learning model training in compliance with all project safety and engineering constraints:
- **No synthetic data** was generated.
- **No duplicate records** were added.
- **The original CSV file was left untouched and unmodified.**
- **No student attendance/headcount data was fabricated.**
- **`Cost_Loss` was identified as target leakage and completely removed.**
- **No machine learning models were trained.**

A reproducible scikit-learn preprocessing pipeline was created in [`preprocess.py`](file:///c:/Users/jalas/project/smart-hostel-food-management-system/preprocess.py).

---

## 2. Record & Schema Summary

| Parameter | Value |
| :--- | :--- |
| **Original Records** | 2,600 |
| **Final Processed Records** | 2,600 |
| **Target Variable** | `Waste_Weight_kg` (Continuous numerical target in kg) |
| **Raw Input Columns** | `Date`, `Meal`, `Canteen_Section`, `Food_Category`, `Unit_Price_per_kg` |
| **Excluded Columns (Leakage)** | `Cost_Loss` (Removed due to direct mathematical derivation from target) |
| **Final Feature Matrix Shape** | **(2600, 16)** (2,600 rows × 16 numerical input features) |

---

## 3. Date Feature Engineering

The original ISO string column `Date` (spanning `2025-06-11` to `2025-08-10`) was parsed into datetime format. Four calendar features were engineered to capture daily and weekly mess waste cycles:

1. **`DayOfWeek`**: Integer (0 = Monday, 1 = Tuesday, ..., 6 = Sunday).
2. **`Month`**: Integer (6 = June, 7 = July, 8 = August).
3. **`Day`**: Integer (Day of the month, 1 to 31).
4. **`IsWeekend`**: Binary indicator (1 if `DayOfWeek` $\ge$ 5, else 0).

*Note: The raw `Date` string column was removed from the input matrix prior to model encoding.*

---

## 4. Feature Taxonomy & Categorical Encoding

### Categorical Features
- **`Meal`**: 3 categories (`Breakfast`, `Lunch`, `Dinner`)
- **`Canteen_Section`**: 4 categories (`A`, `B`, `C`, `D`)
- **`Food_Category`**: 4 categories (`Meat`, `Rice`, `Soup`, `Vegetables`)

### Encoding Method
Categorical features were transformed using **One-Hot Encoding (`sklearn.preprocessing.OneHotEncoder`)** with `sparse_output=False` and `handle_unknown='ignore'`.

This expands the 3 categorical variables into **11 binary indicator columns**:
1. `Meal_Breakfast`
2. `Meal_Dinner`
3. `Meal_Lunch`
4. `Canteen_Section_A`
5. `Canteen_Section_B`
6. `Canteen_Section_C`
7. `Canteen_Section_D`
8. `Food_Category_Meat`
9. `Food_Category_Rice`
10. `Food_Category_Soup`
11. `Food_Category_Vegetables`

### Numerical & Temporal Features (5 features)
12. `Unit_Price_per_kg` (Continuous numerical price per kg)
13. `DayOfWeek` (Discrete integer 0–6)
14. `Month` (Discrete integer 6–8)
15. `Day` (Discrete integer 1–31)
16. `IsWeekend` (Binary integer 0 or 1)

**Total Input Feature Count:** $11 \text{ (One-Hot Encoded)} + 5 \text{ (Numerical/Temporal)} = 16 \text{ features}$.

---

## 5. Data Quality, Outliers & Missing Values

- **Missing Values:** `0` missing values found across all columns.
- **Infinite / Invalid Values:** `0` infinite or non-numeric values found.
- **Duplicates:** `0` duplicate rows.
- **Target Value Distribution (`Waste_Weight_kg`):**
  - Minimum: `0.10` kg
  - Maximum: `5.00` kg
  - Mean: `2.586` kg ($\pm 1.408$ kg)
  - IQR Outlier Count: `0` outliers detected under the 1.5 $\times$ IQR rule.
- **Numerical Feature Distribution (`Unit_Price_per_kg`):**
  - Range: `$1.50` to `$8.00` per kg
  - IQR Outlier Count: `0` outliers.

---

## 6. Target Leakage Verification

An explicit investigation of `Cost_Loss` was performed to detect target leakage:

$$\text{Calculated Cost} = \text{Waste\_Weight\_kg} \times \text{Unit\_Price\_per_kg}$$

### Investigation Findings
- The maximum absolute difference between `Cost_Loss` in the dataset and `Waste_Weight_kg * Unit_Price_per_kg` is less than `$0.005` (attributable strictly to monetary rounding).
- **Leakage Assessment:** Using `Cost_Loss` as a predictor feature would allow an ML model to achieve artificially perfect (trivial) predictions by computing $\text{Waste\_Weight\_kg} = \text{Cost\_Loss} / \text{Unit\_Price\_per\_kg}$.
- **Action Taken:** `Cost_Loss` was **permanently excluded** from the feature set $X$.

---

## 7. Train / Test Split Strategy

An **80% Training / 20% Testing** split was evaluated under two methodologies:

### Method A: Chronological Split (RECOMMENDED & IMPLEMENTED)
- **Training Set (80%):** 2,080 records (Date range: `2025-06-11` to `2025-07-29`)
- **Testing Set (20%):** 520 records (Date range: `2025-07-29` to `2025-08-10`)
- **Rationale:** In a real-world Smart Hostel Food Waste Management System, predictions are made sequentially into the future. A chronological split reflects real operational deployment, preventing future temporal data leakage into the training set.

### Method B: Random Uniform Split (Baseline Alternative)
- **Training Set (80%):** 2,080 records (Random sampling across all dates)
- **Testing Set (20%):** 520 records
- **Rationale:** Useful if evaluating general categorical meal-to-waste mapping across identical calendar days.

---

## 8. Reproducible Scikit-Learn Pipeline Summary

The preprocessing pipeline is codified in [`preprocess.py`](file:///c:/Users/jalas/project/smart-hostel-food-management-system/preprocess.py) using `sklearn.compose.ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Meal', 'Canteen_Section', 'Food_Category']),
        ('num', 'passthrough', ['Unit_Price_per_kg', 'DayOfWeek', 'Month', 'Day', 'IsWeekend'])
    ]
)
```

### Final Preprocessed Output Dimensions
- **Original Dataset Records:** 2,600
- **Final Dataset Records:** 2,600
- **$X_{\text{train}}$ Dimensions:** $(2080, 16)$
- **$X_{\text{test}}$ Dimensions:** $(520, 16)$
- **$y_{\text{train}}$ Dimensions:** $(2080,)$
- **$y_{\text{test}}$ Dimensions:** $(520,)$

---

## 9. Conclusion & Next Steps

Dataset preprocessing for `Dataset Propely.csv` is complete and verified.  
**No models (Linear Regression, Decision Tree, Random Forest) have been trained yet.**  
The pipeline is fully ready for model training in Step 3.
