# STEP 4A: DATASET FEATURE & GAP ANALYSIS REPORT

**Project Title:** Smart Hostel Food Waste Management System  
**Date:** August 10, 2026  
**Status:** Step 4A Completed (Feature Gap Analysis — No Synthetic Data, No Dataset Merging, No Model Retraining)

---

## 1. Context & Motivation

In Step 3, machine learning models were trained on `Dataset Propely.csv` to predict `Waste_Weight_kg` using `Meal`, `Canteen_Section`, `Food_Category`, `Unit_Price_per_kg`, and extracted calendar features (`DayOfWeek`, `Month`, `Day`, `IsWeekend`).

### Benchmark Results from Step 3 (UNTOUCHED):
- **Linear Regression:** MAE = `1.2158 kg` | RMSE = `1.3916 kg` | $R^2 = 0.0019$ (`0.19%`)
- **Random Forest Regressor:** MAE = `1.3190 kg` | RMSE = `1.5416 kg` | $R^2 = -0.2248$ (`-22.48%`)
- **Decision Tree Regressor:** MAE = `1.6775 kg` | RMSE = `2.0476 kg` | $R^2 = -1.1609$ (`-116.09%`)

These empirical results confirm that `Dataset Propely.csv` contains insufficient predictive signal for `Waste_Weight_kg`. The purpose of Step 4A is to re-inspect **all four original Kaggle datasets** across 15 operational variables to diagnose feature deficiencies.

---

## 2. 15-Variable Feature Breakdown Across All 4 Datasets

| Variable # | Feature Category | Dataset 1: `Dataset Propely.csv` | Dataset 2: `food_wastage_data.csv` | Dataset 3: `WorldWide_foodwastage_dataset.csv` | Dataset 4: `global_food_wastage_dataset.csv` |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Attendance** | ❌ NO | ⚠️ PARTIAL (`Number of Guests`) | ❌ NO | ❌ NO |
| **2** | **Number of People / Students** | ❌ NO | ⚠️ PARTIAL (`Number of Guests` 207-491) | ⚠️ MACRO (`Country Population`) | ⚠️ MACRO (`Population`) |
| **3** | **Food Quantity Prepared** | ❌ NO | ⚠️ PARTIAL (`Quantity of Food`) | ❌ NO | ❌ NO |
| **4** | **Food Quantity Consumed** | ❌ NO | ❌ NO | ❌ NO | ❌ NO |
| **5** | **Food Waste Weight (Target)** | ✅ YES (`Waste_Weight_kg`) | ✅ YES (`Wastage Food Amount`) | ⚠️ MACRO (`Total Waste in Tons`) | ⚠️ MACRO (`Total Waste (Tons)`) |
| **6** | **Meal Type** | ✅ YES (`Meal`) | ⚠️ PARTIAL (`Preparation Method`) | ❌ NO | ❌ NO |
| **7** | **Food Category** | ✅ YES (`Food_Category`) | ✅ YES (`Type of Food`) | ✅ YES (`Food Types`) | ✅ YES (`Food Category`) |
| **8** | **Date / Timestamp** | ✅ YES (`Date`) | ❌ NO | ⚠️ ANNUAL ONLY (`Year`) | ⚠️ ANNUAL ONLY (`Year`) |
| **9** | **Temperature** | ❌ NO | ❌ NO | ❌ NO | ❌ NO |
| **10** | **Weather Conditions** | ❌ NO | ❌ NO | ❌ NO | ❌ NO |
| **11** | **Seasonality** | ⚠️ INDIRECT (`Month`) | ⚠️ CATEGORICAL (`Seasonality`) | ❌ NO | ❌ NO |
| **12** | **Holiday / Weekend** | ⚠️ INDIRECT (`IsWeekend`) | ❌ NO | ❌ NO | ❌ NO |
| **13** | **Menu Details** | ❌ NO | ❌ NO | ❌ NO | ❌ NO |
| **14** | **Cost / Pricing** | ✅ YES (`Unit_Price_per_kg`) | ⚠️ CATEGORICAL (`Pricing`) | ⚠️ MACRO (`Economic Loss $M`) | ⚠️ MACRO (`Economic Loss $M`) |
| **15** | **Other Operational Variables** | `Canteen_Section` | `Storage Conditions`, `Event Type` | `Household Waste (%)` | `Household Waste (%)` |

---

## 3. Dataset Comparison & Suitability Matrix

| Dataset File | Total Records | Relevant Predictors Present | Target Variable | Student Attendance | Weather / Temp | Meal Session | Food Prepared | Suitability for Hostel Food Waste ML |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Dataset Propely.csv`** | 2,600 | `Date`, `Meal`, `Canteen_Section`, `Food_Category`, `Unit_Price_per_kg` | `Waste_Weight_kg` (kg) | ❌ Absent | ❌ Absent | ✅ Present | ❌ Absent | **INSUFFICIENT SIGNAL** (Lacks attendance, quantity prepared, and weather). |
| **`food_wastage_data.csv`** | 1,782 | `Number of Guests`, `Quantity of Food`, `Type of Food`, `Event Type`, `Storage` | `Wastage Food Amount` | ⚠️ Event Guests Only | ❌ Absent | ⚠️ Event Style | ⚠️ Event Prep | **UNSUITABLE** (Catering events like weddings/birthdays, no dates, 164 duplicates). |
| **`WorldWide_foodwastage_dataset.csv`** | 5,000 | `Country`, `Year`, `Food Types`, `Population`, `Household Waste (%)` | `Total Waste in Tons` | ❌ Absent | ❌ Absent | ❌ Absent | ❌ Absent | **UNSUITABLE** (Macro annual country totals in millions of tons). |
| **`global_food_wastage_dataset.csv`** | 5,000 | `Country`, `Year`, `Food Category`, `Population`, `Household Waste (%)` | `Total Waste (Tons)` | ❌ Absent | ❌ Absent | ❌ Absent | ❌ Absent | **UNSUITABLE** (Macro annual country totals in millions of tons). |

---

## 4. Key Findings & Explicit Gap Diagnosis

### Primary Finding
**NONE of the four original Kaggle datasets contain sufficient predictors to build a high-accuracy, operational machine learning model for a Smart Hostel Food Waste Management System.**

### Detailed Gap Analysis:
1. **Absence of Student Attendance / Headcount:** In hostel mess operations, daily student headcount (actual number of students eating per meal) is the single strongest predictor of food consumption and food waste. Without attendance data, an ML model cannot distinguish whether high waste is caused by over-preparation or low student turnout (e.g. students eating outside during weekends/events).
2. **Absence of Quantity Prepared:** `Dataset Propely.csv` records the waste weight, but does NOT record how many kilograms of food were cooked/prepared. Without quantity prepared, waste cannot be evaluated as a ratio of production ($\text{Waste Ratio} = \text{Waste} / \text{Prepared}$).
3. **Absence of External Context:** Factors such as weather (extreme heat or heavy rainfall impacts mess attendance), academic calendar (exam periods vs holidays/vacations), and specific menu items are completely missing from all four datasets.

---

## 5. Recommendation for Real-World Kaggle Dataset Enhancements

To transform this into a robust, high-performing, real-world machine learning system, the dataset should incorporate the following **7 core feature domains**:

1. **Student Headcount / Attendance:** `Registered_Students`, `Actual_Attendance_Count`, `Attendance_Percentage`.
2. **Food Production Metrics:** `Food_Prepared_kg` (Total kilograms cooked per meal).
3. **Food Consumption Metrics:** `Food_Consumed_kg` (Total kilograms consumed by students).
4. **Food Waste Target:** `Waste_Weight_kg` (Kilograms wasted per meal/section).
5. **Operational Meal Context:** `Meal` (`Breakfast`, `Lunch`, `Dinner`, `Snacks`), `Canteen_Section` (`A`, `B`, `C`, `D`), `Food_Category` (`Rice`, `Vegetables`, `Meat`, `Soup`).
6. **Academic Calendar Context:** `Day_Type` (`Regular Weekday`, `Weekend`, `Exam Period`, `Holiday / Vacation`).
7. **Weather Context:** `Temperature_C`, `Weather_Condition` (`Clear`, `Rainy`, `Extreme Heat`).

---

## 6. Next Steps Guidelines

- **Do NOT generate synthetic data.**
- **Do NOT fabricate attendance figures artificially.**
- **Do NOT modify existing Step 3 model artifacts or evaluation numbers.**

*Step 4A Feature Gap Analysis Complete.*
