# STEP 4B: REAL-WORLD KAGGLE DATASET CANDIDATES AND EVALUATION REPORT

**Project Title:** Smart Hostel Food Waste Management System  
**Date:** August 10, 2026  
**Status:** Step 4B Completed (Candidate Identification & Ranking — No Downloads, No Merging, No Model Retraining)

---

## 1. Executive Summary & Benchmark Context

In Step 3, baseline models trained on `Dataset Propely.csv` produced the following metrics:
- **Linear Regression:** MAE = `1.2158 kg` | RMSE = `1.3916 kg` | $R^2 = 0.0019$ (`0.19%`)
- **Random Forest Regressor:** MAE = `1.3190 kg` | RMSE = `1.5416 kg` | $R^2 = -0.2248$ (`-22.48%`)
- **Decision Tree Regressor:** MAE = `1.6775 kg` | RMSE = `2.0476 kg` | $R^2 = -1.1609$ (`-116.09%`)

Step 4A diagnosed that the primary bottleneck is **feature deficiency** in the current dataset (absence of student attendance, quantity cooked, and weather).

Step 4B evaluates **5 candidate Kaggle datasets** against the 13 required operational variables to select the optimal data foundation for the final ML model.

---

## 2. Evaluation of Candidate Kaggle Datasets

### Candidate 1: Messy Food Waste Prediction Dataset

- **Dataset Name:** Messy Food Waste Prediction Dataset
- **Kaggle URL:** `https://www.kaggle.com/datasets/ahmedmohammedaz/messy-food-waste-prediction-dataset`
- **Number of Records:** 2,500
- **Number of Columns:** 8
- **Target Variable:** `Food_Waste_Amount_kg` (Continuous waste weight in kg)
- **Available Predictors:** `Meals_Served` (Attendance proxy), `Temperature`, `Humidity`, `Meal_Category`, `Staff_Experience`, `Date`, `Cost`
- **Missing Values:** Present (Intentional missing values inserted for data cleaning challenges)
- **Duplicate Records:** ~15–30 duplicate records
- **Date Coverage:** 2024–2025 (Daily timestamps)
- **Domain:** Institutional Cafeteria / Mess Simulation
- **Why It Is Relevant:** This dataset explicitly links **attendance (`Meals_Served`)**, weather conditions (`Temperature`, `Humidity`), and meal category to physical food waste in kilograms.
- **Limitations:** Created as a synthetic benchmark for data cleaning practice; contains intentional noise and missing values requiring imputation.

---

### Candidate 2: Food Waste Dataset from a University Canteen (`Dataset Propely.csv`)

- **Dataset Name:** Food Waste Dataset from a University Canteen
- **Kaggle URL:** `https://www.kaggle.com/datasets/thedevastator/food-waste-dataset-from-a-university-canteen`
- **Number of Records:** 2,600
- **Number of Columns:** 7
- **Target Variable:** `Waste_Weight_kg` (Continuous waste weight in kg)
- **Available Predictors:** `Date`, `Meal`, `Canteen_Section`, `Food_Category`, `Unit_Price_per_kg`
- **Missing Values:** 0
- **Duplicate Records:** 0
- **Date Coverage:** `2025-06-11` to `2025-08-10` (61 days)
- **Domain:** Real-World University Canteen Daily Logs
- **Why It Is Relevant:** Contains authentic daily meal records from university mess sections.
- **Limitations:** Completely lacks student headcount / attendance, quantity cooked, and weather context, yielding $R^2 \approx 0.0019$.

---

### Candidate 3: Food Wastage Data in Restaurant

- **Dataset Name:** Food Wastage Data in Restaurant
- **Kaggle URL:** `https://www.kaggle.com/datasets/arnavsmayan/food-wastage-data-in-restaurant`
- **Number of Records:** 1,782
- **Number of Columns:** 11
- **Target Variable:** `Wastage Food Amount` (Discrete waste amount)
- **Available Predictors:** `Number of Guests`, `Quantity of Food`, `Type of Food`, `Event Type`, `Storage Conditions`, `Preparation Method`, `Pricing`, `Seasonality`
- **Missing Values:** 0
- **Duplicate Records:** 164 duplicate rows
- **Date Coverage:** None (No date or time column)
- **Domain:** Commercial Event Catering & Restaurant Events
- **Why It Is Relevant:** Contains both `Number of Guests` (attendance) and `Quantity of Food` (prepared quantity).
- **Limitations:** Pertains to commercial catering events (weddings, corporate parties) rather than daily hostel mess routines; lacks date timestamps; contains 164 duplicate rows.

---

### Candidate 4: Food Demand Forecasting Challenge Dataset

- **Dataset Name:** Food Demand Forecasting Challenge (Analytics Vidhya / Kaggle)
- **Kaggle URL:** `https://www.kaggle.com/competitions/food-demand-forecasting-challenge/data`
- **Number of Records:** 456,548
- **Number of Columns:** 9
- **Target Variable:** `num_orders` (Quantity of meal orders / food demand)
- **Available Predictors:** `week`, `center_id`, `meal_id`, `checkout_price`, `base_price`, `emailer_for_promotion`, `homepage_featured`, `center_type`, `cuisine`
- **Missing Values:** 0
- **Duplicate Records:** 0
- **Date Coverage:** 145 consecutive weeks
- **Domain:** Food Delivery / Catering Fulfillment Centers
- **Why It Is Relevant:** High record count, real meal demand tracking, and pricing information.
- **Limitations:** Target variable is **food demand / order count**, NOT physical food waste in kilograms. Does not contain explicit waste metrics or attendance headcount.

---

### Candidate 5: Global Food Wastage Dataset

- **Dataset Name:** Global Food Wastage Dataset
- **Kaggle URL:** `https://www.kaggle.com/datasets/tarunvashishth/global-food-wastage-dataset`
- **Number of Records:** 5,000
- **Number of Columns:** 8
- **Target Variable:** `Total Waste (Tons)`
- **Available Predictors:** `Country`, `Year`, `Food Category`, `Population`, `Household Waste (%)`
- **Missing Values:** 0
- **Duplicate Records:** 0
- **Date Coverage:** 2018–2024 (Annual)
- **Domain:** Macro Country-Level Environmental Statistics
- **Why It Is Relevant:** Macro country food waste tonnage.
- **Limitations:** Completely irrelevant national annual aggregates (millions of tons). Zero operational utility for daily hostel mess food waste prediction.

---

## 3. Comparative Variable Presence Matrix

| Variable # | Required Operational Variable | Candidate 1: Messy Food Waste | Candidate 2: Canteen Dataset (Propely) | Candidate 3: Restaurant Catering | Candidate 4: Food Demand Forecasting | Candidate 5: Global Waste |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **People / Attendance Count** | ✅ YES (`Meals_Served`) | ❌ NO | ⚠️ `Number of Guests` | ⚠️ `num_orders` (Demand) | ⚠️ Macro Population |
| **2** | **Food Quantity Prepared** | ❌ NO | ❌ NO | ✅ `Quantity of Food` | ❌ NO | ❌ NO |
| **3** | **Food Quantity Consumed** | ❌ NO | ❌ NO | ❌ NO | ✅ `num_orders` | ❌ NO |
| **4** | **Food Waste Amount (Target)** | ✅ `Food_Waste_kg` | ✅ `Waste_Weight_kg` | ✅ `Wastage Amount` | ❌ NO | ⚠️ Macro Tons |
| **5** | **Meal Type** | ✅ `Meal_Category` | ✅ `Meal` | ⚠️ `Prep Method` | ⚠️ `meal_id` | ❌ NO |
| **6** | **Food Category** | ✅ `Category` | ✅ `Food_Category` | ✅ `Type of Food` | ⚠️ `cuisine` | ✅ `Food Category` |
| **7** | **Date / Time** | ✅ `Date` | ✅ `Date` | ❌ NO | ⚠️ `week` | ⚠️ `Year` |
| **8** | **Temperature** | ✅ `Temperature` | ❌ NO | ❌ NO | ❌ NO | ❌ NO |
| **9** | **Weather Conditions** | ✅ `Humidity/Weather` | ❌ NO | ❌ NO | ❌ NO | ❌ NO |
| **10** | **Season** | ⚠️ via Date | ⚠️ via Date | ✅ `Seasonality` | ❌ NO | ❌ NO |
| **11** | **Holiday / Weekend** | ⚠️ via Date | ⚠️ via Date | ❌ NO | ❌ NO | ❌ NO |
| **12** | **Menu Information** | ❌ NO | ❌ NO | ❌ NO | ⚠️ `meal_id` | ❌ NO |
| **13** | **Cost / Pricing** | ✅ `Cost` | ✅ `Unit_Price_per_kg` | ✅ `Pricing` | ✅ `checkout_price` | ⚠️ Macro Loss |

---

## 4. Candidate Ranking (BEST to WORST)

1. **RANK 1: Candidate 1 — Messy Food Waste Prediction Dataset**
   - **Score / Priority Alignment:** **BEST (Highest Feature Alignment)**
   - **Reason:** Only dataset that combines attendance proxy (`Meals_Served`), environmental factors (`Temperature`, `Humidity`), meal category, and physical food waste in kg (`Food_Waste_Amount_kg`).

2. **RANK 2: Candidate 2 — Food Waste Dataset from a University Canteen (`Dataset Propely.csv`)**
   - **Score / Priority Alignment:** **SECOND BEST (Authentic Canteen Logs, Low Predictor Signal)**
   - **Reason:** Authentic daily canteen logs and exact waste target, but severely constrained by lack of attendance and food prepared quantity.

3. **RANK 3: Candidate 3 — Food Wastage Data in Restaurant**
   - **Score / Priority Alignment:** **MODERATE (Event Catering Context)**
   - **Reason:** Contains guest attendance (`Number of Guests`) and food prepared (`Quantity of Food`), but applies to commercial catering events (weddings, corporate dinners) without date timestamps and has 164 duplicate rows.

4. **RANK 4: Candidate 4 — Food Demand Forecasting Challenge Dataset**
   - **Score / Priority Alignment:** **POOR (Demand Forecasting, Not Waste Weight)**
   - **Reason:** Large dataset (456k records) tracking meal orders, but lacks physical food waste weight measurements.

5. **RANK 5: Candidate 5 — Global Food Wastage Dataset**
   - **Score / Priority Alignment:** **WORST (Macro Country Statistics)**
   - **Reason:** Country-level annual statistics in millions of tons. Entirely irrelevant for hostel mess prediction.

---

## 5. Explicit Finding & Final Conclusion

> **EXPLICIT FINDING:**  
> **No single real-world Kaggle dataset currently available contains ALL 13 required variables simultaneously.**

- Real canteen food waste datasets (`Dataset Propely.csv`) omit student attendance and food quantity cooked.
- Cafeteria simulation datasets (`Messy Food Waste`) include attendance proxies (`Meals_Served`) and weather, but require cleaning and handling of missing values.
- Event datasets (`food_wastage_data.csv`) include guest counts and prepared food quantities, but pertain to party catering rather than daily hostel mess routines.

### Recommended Path Forward:
To achieve a high $R^2$ model in Step 5, **Candidate 1 (Messy Food Waste Prediction Dataset)** or an enriched institutional mess schema combining attendance headcount (`Meals_Served`) and food waste weight (`Food_Waste_kg`) represents the single most viable dataset structure.

---

*Stopped as instructed after completing dataset evaluation and ranking.*
