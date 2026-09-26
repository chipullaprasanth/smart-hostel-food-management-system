# STEP 7: MODEL TRAINING AND EVALUATION REPORT

## 1. Executive Summary

The approved Messy Food Waste Prediction Dataset (`datasets/messy_food_waste/train.csv` and `datasets/messy_food_waste/test.csv`) was used to execute complete model training, hyperparameter tuning, cross-validation evaluation, and test-set inference.

- **Training Records:** 1,000 chronological daily hostel operational records.
- **Test Records:** 200 test rows.
- **Processed Features:** 19 features (11 numerical, 8 one-hot encoded categorical).
- **Target Variable:** `food_waste_kg` (present exclusively in `train.csv`).
- **Validation Protocol:** 5-Fold `TimeSeriesSplit(n_splits=5)` to maintain chronological data integrity without leakage.
- **Test Target Availability:** `test.csv` contains no `food_waste_kg` target column. Therefore, final test-set MAE, RMSE, and R² ground-truth metrics cannot be computed.

## 2. Step 3 Baseline Integrity

The initial baseline metrics established in Step 3 on the synthetic/legacy dataset remain preserved completely untouched:

- **Linear Regression:** MAE = 1.2158 kg, RMSE = 1.3916 kg, R² = 0.0019 (0.19%)
- **Random Forest:** MAE = 1.3190 kg, RMSE = 1.5416 kg, R² = -0.2248 (-22.48%)
- **Decision Tree:** MAE = 1.6775 kg, RMSE = 2.0476 kg, R² = -1.1609 (-116.09%)

These baseline metrics and historical model artifacts (`model/preprocessor.pkl`) were not modified, overwritten, or retrained during Step 7 execution.

## 3. Preprocessing

The preprocessing pipeline recreates the deterministic Step 6 data preparation strictly inside every cross-validation fold to prevent data leakage:

1. **Date Feature Extraction:** Extracted `year`, `month`, `day`, `day_of_week` (0=Monday..6=Sunday), and `is_weekend` (1 if day_of_week ≥ 5 else 0).
2. **Meals Served Numeric Extraction:** Parsed integer meal counts from messy text values (e.g., `"500 meals"` → `500.0`).
3. **Kitchen Staff Numeric Extraction:** Parsed staff counts from messy text values (e.g., `"12 staff"` → `12.0`).
4. **Categorical Normalization:** Whitespace stripping and title-casing for `staff_experience` (`Beginner`, `Expert`, `Intermediate`) and `waste_category` (`Bakery`, `Dairy`, `Meat`, `Rice`, `Vegetables`).
5. **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
6. **Categorical Pipeline:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
7. **Pipeline Fitting:** Preprocessing transformers were fitted **only** on training fold data within each CV loop.

## 4. Validation Method

- **Method:** `TimeSeriesSplit(n_splits=5)`.
- **Chronological Sorting:** `train.csv` was sorted chronologically by `date` prior to cross-validation split generation.
- **Leakage Control:** Preprocessing pipelines were encapsulated inside scikit-learn `Pipeline` objects. `test.csv` was strictly isolated and never used during cross-validation, hyperparameter tuning, preprocessing fitting, or model selection.

## 5. Models Evaluated

Seven regression models were trained, tuned, and evaluated:
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. HistGradientBoostingRegressor
7. Extra Trees Regressor

## 6. Hyperparameter Search

GridSearchCV was executed with `scoring='neg_mean_absolute_error'`, `cv=TimeSeriesSplit(n_splits=5)`, and `refit=True`:

- **Linear Regression:** No hyperparameters (default parameters).
- **Ridge Regression:** `alpha` = [0.01, 0.1, 1, 10, 100]
- **Decision Tree Regressor:** `max_depth` = [None, 3, 5, 10, 15], `min_samples_split` = [2, 5, 10], `min_samples_leaf` = [1, 2, 4]
- **Random Forest Regressor:** `n_estimators` = [200, 400], `max_depth` = [None, 5, 10, 15], `min_samples_split` = [2, 5, 10], `min_samples_leaf` = [1, 2, 4], `max_features` = ['sqrt', 1.0]
- **Gradient Boosting Regressor:** `n_estimators` = [100, 200], `learning_rate` = [0.03, 0.05, 0.1], `max_depth` = [2, 3, 4], `min_samples_leaf` = [2, 5, 10]
- **HistGradientBoostingRegressor:** `max_iter` = [100, 200], `learning_rate` = [0.03, 0.05, 0.1], `max_leaf_nodes` = [15, 31], `l2_regularization` = [0, 0.1, 1.0]
- **Extra Trees Regressor:** `n_estimators` = [200, 400], `max_depth` = [None, 5, 10, 15], `min_samples_split` = [2, 5, 10], `min_samples_leaf` = [1, 2, 4]

## 7. Model Comparison

The following table summarizes the cross-validation performance across all 5 chronological folds:

| Model | Mean CV MAE (kg) | Std CV MAE (kg) | Mean CV RMSE (kg) | Std CV RMSE (kg) | Mean CV R² | Std CV R² | Best Parameters | Training Time |
|---|---|---|---|---|---|---|---|---|
| Ridge Regression | 2.3966 | 0.1575 | 2.9827 | 0.1617 | 0.7930 | 0.0319 | `{'model__alpha': 1}` | 8.25s |
| Linear Regression | 2.3975 | 0.1588 | 2.9842 | 0.1632 | 0.7927 | 0.0324 | `{}` | 0.15s |
| Extra Trees Regressor | 2.4429 | 0.2167 | 3.0382 | 0.2064 | 0.7861 | 0.0279 | `{'model__max_depth': 10, 'model__min_samples_leaf': 4, 'model__min_samples_split': 2, 'model__n_estimators': 400}` | 39.11s |
| Gradient Boosting Regressor | 2.4720 | 0.2016 | 3.0936 | 0.2338 | 0.7778 | 0.0336 | `{'model__learning_rate': 0.03, 'model__max_depth': 2, 'model__min_samples_leaf': 2, 'model__n_estimators': 200}` | 11.70s |
| HistGradientBoostingRegressor | 2.4846 | 0.2133 | 3.1006 | 0.2190 | 0.7764 | 0.0344 | `{'model__l2_regularization': 0, 'model__learning_rate': 0.05, 'model__max_iter': 100, 'model__max_leaf_nodes': 15}` | 12.66s |
| Random Forest Regressor | 2.5239 | 0.1981 | 3.1321 | 0.2189 | 0.7721 | 0.0333 | `{'model__max_depth': 5, 'model__max_features': 1.0, 'model__min_samples_leaf': 4, 'model__min_samples_split': 2, 'model__n_estimators': 400}` | 81.92s |
| Decision Tree Regressor | 2.8246 | 0.2096 | 3.5359 | 0.2160 | 0.7085 | 0.0464 | `{'model__max_depth': 5, 'model__min_samples_leaf': 4, 'model__min_samples_split': 10}` | 1.34s |

## 8. Best Model

- **Selected Best Model:** **Ridge Regression**
- **Selection Criterion:** Lowest 5-Fold TimeSeriesSplit Mean CV MAE.
- **Mean CV MAE:** **2.3966 ± 0.1575 kg**
- **Mean CV RMSE:** **2.9827 ± 0.1617 kg**
- **Mean CV R²:** **0.7930 ± 0.0319**
- **Best Parameters:** `{'model__alpha': 1}`

## 9. Final Model

After selecting **Ridge Regression** based on CV performance, the complete preprocessing + model pipeline was refitted on **all 1,000 training rows**.

- **Saved Pipeline Artifact:** `model/final_food_waste_model.pkl`

## 10. Test Predictions

The final fitted pipeline was used to predict food waste for all 200 test rows in `test.csv`.

- **Prediction Output File:** `outputs/final_test_predictions.csv`
- **Total Test Predictions Generated:** 200
- **Minimum Predicted Waste:** 9.9160 kg
- **Maximum Predicted Waste:** 37.4860 kg
- **Mean Predicted Waste:** 23.6981 kg
- **Median Predicted Waste:** 23.2971 kg

> Test-set ground-truth metrics are unavailable because test.csv does not contain food_waste_kg.

## 11. Feature Importance

Feature importance metrics were extracted from the final fitted pipeline:

| Feature | Absolute Coefficient Magnitude |
|---|---|
| `meals_served_numeric` | 5.061070 |
| `past_waste_kg` | 3.253564 |
| `waste_category_Meat` | 0.450631 |
| `staff_experience_Beginner` | 0.300370 |
| `is_weekend` | 0.286802 |
| `staff_experience_Intermediate` | 0.261460 |
| `month` | 0.207206 |
| `waste_category_Vegetables` | 0.178278 |
| `year` | 0.176328 |
| `waste_category_Bakery` | 0.169164 |

- **Saved Feature Importance Artifact:** `outputs/step7_feature_importance.csv`
- **Feature Importance Plot:** `reports/step7/feature_importance.png`

## 12. Generated Artifacts

The following files and directories were generated and verified:

- `model/final_food_waste_model.pkl` (Final fitted 1,000-row pipeline)
- `outputs/final_test_predictions.csv` (200 test predictions with all test fields)
- `outputs/step7_model_comparison.csv` (Full 7-model CV comparison metrics)
- `outputs/step7_feature_importance.csv` (Transformed feature importances/coefficients)
- `reports/step7/cv_model_comparison.png` (CV MAE comparison chart with error bars)
- `reports/step7/predictions_distribution.png` (Test prediction distribution plot)
- `reports/step7/feature_importance.png` (Feature importance horizontal bar chart)
- `STEP7_MODEL_TRAINING_REPORT.md` (This comprehensive markdown report)
- `step7_training.py` (Fully reproducible training and evaluation script)

## 13. Sanity Checks

| Verification Check | Result |
|---|---|
| train rows = 1000 | **PASS** |
| test rows = 200 | **PASS** |
| processed features = 19 | **PASS** |
| target = food_waste_kg | **PASS** |
| target absent from X | **PASS** |
| test not used in tuning | **PASS** |
| no synthetic data | **PASS** |
| no datasets merged | **PASS** |
| Dataset Propely.csv untouched | **PASS** |
| model/preprocessor.pkl preserved | **PASS** |
| final model exists | **PASS** |
| 200 predictions generated | **PASS** |

## 14. Final Verdict

**STEP 7 COMPLETE — MODEL TRAINED, TUNED, SELECTED, AND 200 TEST PREDICTIONS GENERATED.

The final model was selected using chronological cross-validation on the training dataset. Because the supplied test.csv contains no food_waste_kg ground truth, test-set regression metrics cannot be computed.**
