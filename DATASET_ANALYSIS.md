# STEP 1: FINAL DATASET VERIFICATION AND ANALYSIS REPORT

**Project Title:** Smart Hostel Food Waste Management System  
**Date of Analysis:** August 10, 2026  
**Status:** Step 1 Completed (Verification & Selection Only - No Model Training / Data Modification Executed)

---

## Executive Summary

Four Kaggle datasets provided for the **Smart Hostel Food Waste Management System** were thoroughly analyzed without modifying original datasets, merging records, generating synthetic data, or training machine learning models.

The primary objective of this step is to evaluate each dataset's data quality, granularity, schema, and genuine alignment with a **real-world prediction problem in hostel/canteen food waste management**.

---

## Detailed Dataset Analysis

### Dataset 1: `Dataset Propely.csv`

1. **Dataset / File Name:** `Dataset Propely.csv`
2. **Number of Records:** 2,600
3. **Number of Columns:** 7
4. **All Column Names:**
   - `Date`
   - `Meal`
   - `Canteen_Section`
   - `Food_Category`
   - `Waste_Weight_kg`
   - `Unit_Price_per_kg`
   - `Cost_Loss`
5. **Data Types:**
   - `Date`: String (ISO Format `YYYY-MM-DD`, convertible to Datetime)
   - `Meal`: String / Categorical (`Breakfast`, `Lunch`, `Dinner`)
   - `Canteen_Section`: String / Categorical (`A`, `B`, `C`, `D`)
   - `Food_Category`: String / Categorical (`Rice`, `Vegetables`, `Meat`, `Soup`)
   - `Waste_Weight_kg`: Float64 (Continuous quantitative measurement in kilograms)
   - `Unit_Price_per_kg`: Float64 (Continuous quantitative pricing)
   - `Cost_Loss`: Float64 (Continuous monetary loss, calculated as `Waste_Weight_kg * Unit_Price_per_kg`)
6. **Missing Values:** 0 missing values across all columns.
7. **Duplicate Records:** 0 duplicate rows.
8. **Numerical Features:** `Waste_Weight_kg`, `Unit_Price_per_kg`, `Cost_Loss`.
9. **Categorical Features:** `Meal`, `Canteen_Section`, `Food_Category`.
10. **Date/Time Features:** `Date` (spanning 2025-06-11 to 2025-08-10).
11. **Possible Target Variables:** `Waste_Weight_kg` (Primary target for regression predicting daily food waste in kg) or `Cost_Loss` (Financial loss target).
12. **Possible ML Input Features:** `Meal`, `Canteen_Section`, `Food_Category`, `Unit_Price_per_kg`, extracted date components (`DayOfWeek`, `Month`, `Day`, `IsWeekend`).
13. **Outliers:** 
    - `Waste_Weight_kg`: 0 outliers (Range: 0.10 kg to 5.00 kg, Mean: 2.59 kg).
    - `Unit_Price_per_kg`: 0 outliers (Range: $1.50 to $8.00).
    - `Cost_Loss`: 291 outliers using standard IQR method ($0.15 to $40.00) due to multiplicative tail effects when high weight coincides with high unit price (e.g. Meat at $8/kg).
14. **Genuine Relevance to "Smart Hostel Food Waste Management System":** **HIGHLY RELEVANT.** This dataset explicitly tracks daily canteen operations, specific meal sessions (Breakfast/Lunch/Dinner), hostel canteen sections, food categories, and physical waste in kilograms.

---

### Dataset 2: `food_wastage_data.csv`

1. **Dataset / File Name:** `food_wastage_data.csv`
2. **Number of Records:** 1,782
3. **Number of Columns:** 11
4. **All Column Names:**
   - `Type of Food`
   - `Number of Guests`
   - `Event Type`
   - `Quantity of Food`
   - `Storage Conditions`
   - `Purchase History`
   - `Seasonality`
   - `Preparation Method`
   - `Geographical Location`
   - `Pricing`
   - `Wastage Food Amount`
5. **Data Types:**
   - `Type of Food`: String / Categorical (`Meat`, `Vegetables`, `Fruits`, `Baked Goods`, `Dairy Products`)
   - `Number of Guests`: Int64 (Discrete integer count, 207 to 491)
   - `Event Type`: String / Categorical (`Corporate`, `Birthday`, `Wedding`, `Social Gathering`)
   - `Quantity of Food`: Int64 (Discrete integer count, 280 to 500)
   - `Storage Conditions`: String / Categorical (`Refrigerated`, `Room Temperature`)
   - `Purchase History`: String / Categorical (`Regular`, `Occasional`)
   - `Seasonality`: String / Categorical (`All Seasons`, `Winter`, `Summer`)
   - `Preparation Method`: String / Categorical (`Buffet`, `Finger Food`, `Sit-down Dinner`)
   - `Geographical Location`: String / Categorical (`Urban`, `Suburban`, `Rural`)
   - `Pricing`: String / Categorical (`Low`, `Moderate`, `High`)
   - `Wastage Food Amount`: Int64 (Discrete integer waste amount, 10 to 63)
6. **Missing Values:** 0 missing values across all columns.
7. **Duplicate Records:** 164 duplicate rows present.
8. **Numerical Features:** `Number of Guests`, `Quantity of Food`, `Wastage Food Amount`.
9. **Categorical Features:** `Type of Food`, `Event Type`, `Storage Conditions`, `Purchase History`, `Seasonality`, `Preparation Method`, `Geographical Location`, `Pricing`.
10. **Date/Time Features:** None (Seasonality is categorical, no explicit dates or timestamps).
11. **Possible Target Variables:** `Wastage Food Amount`.
12. **Possible ML Input Features:** `Type of Food`, `Number of Guests`, `Event Type`, `Quantity of Food`, `Storage Conditions`, `Preparation Method`, `Pricing`.
13. **Outliers:** 
    - `Number of Guests`: 35 outliers (IQR method).
    - `Quantity of Food`: 0 outliers.
    - `Wastage Food Amount`: 10 outliers.
14. **Genuine Relevance to "Smart Hostel Food Waste Management System":** **LOW / IRRELEVANT.** This dataset focuses on event catering (Weddings, Birthdays, Corporate events) rather than institutional daily hostel mess operations. It also contains 164 duplicate records.

---

### Dataset 3: `WorldWide_foodwastage_dataset.csv`

1. **Dataset / File Name:** `WorldWide_foodwastage_dataset.csv`
2. **Number of Records:** 5,000
3. **Number of Columns:** 8
4. **All Column Names:**
   - `Country`
   - `Year`
   - `Food Types`
   - `Total Waste in Tons`
   - `Food Economic Loss (Million $)`
   - `Avg Waste per Capita (Kg)`
   - `Country Population (Million) (Not Accurate)`
   - `Household Waste (%)`
5. **Data Types:**
   - `Country`: String / Categorical (20 countries)
   - `Year`: Int64 (Annual integer year: 2018–2024)
   - `Food Types`: String / Categorical (8 food categories)
   - `Total Waste in Tons`: Float64 (Macro country-level tonnage)
   - `Food Economic Loss (Million $)`: Float64 (Macro economic loss)
   - `Avg Waste per Capita (Kg)`: Float64 (Per capita annual estimate)
   - `Country Population (Million) (Not Accurate)`: Float64
   - `Household Waste (%)`: Float64
6. **Missing Values:** 0 missing values.
7. **Duplicate Records:** 0 duplicate rows.
8. **Numerical Features:** `Year`, `Total Waste in Tons`, `Food Economic Loss (Million $)`, `Avg Waste per Capita (Kg)`, `Country Population (Million) (Not Accurate)`, `Household Waste (%)`.
9. **Categorical Features:** `Country`, `Food Types`.
10. **Date/Time Features:** `Year`.
11. **Possible Target Variables:** `Total Waste in Tons` or `Avg Waste per Capita (Kg)`.
12. **Possible ML Input Features:** `Country`, `Year`, `Food Types`, `Population`, `Household Waste (%)`.
13. **Outliers:** 0 outliers detected via IQR method.
14. **Genuine Relevance to "Smart Hostel Food Waste Management System":** **NOT RELEVANT.** This dataset contains macro country-level annual aggregates (waste in millions of tons per nation). A hostel canteen cannot use nationwide annual statistics to optimize daily hostel food prep or mess waste.

---

### Dataset 4: `global_food_wastage_dataset.csv`

1. **Dataset / File Name:** `global_food_wastage_dataset.csv`
2. **Number of Records:** 5,000
3. **Number of Columns:** 8
4. **All Column Names:**
   - `Country`
   - `Year`
   - `Food Category`
   - `Total Waste (Tons)`
   - `Economic Loss (Million $)`
   - `Avg Waste per Capita (Kg)`
   - `Population (Million)`
   - `Household Waste (%)`
5. **Data Types:**
   - `Country`: String / Categorical (20 countries)
   - `Year`: Int64 (2018–2024)
   - `Food Category`: String / Categorical (8 categories)
   - `Total Waste (Tons)`: Float64
   - `Economic Loss (Million $)`: Float64
   - `Avg Waste per Capita (Kg)`: Float64
   - `Population (Million)`: Float64
   - `Household Waste (%)`: Float64
6. **Missing Values:** 0 missing values.
7. **Duplicate Records:** 0 duplicate rows.
8. **Numerical Features:** `Year`, `Total Waste (Tons)`, `Economic Loss (Million $)`, `Avg Waste per Capita (Kg)`, `Population (Million)`, `Household Waste (%)`.
9. **Categorical Features:** `Country`, `Food Category`.
10. **Date/Time Features:** `Year`.
11. **Possible Target Variables:** `Total Waste (Tons)`.
12. **Possible ML Input Features:** `Country`, `Year`, `Food Category`, `Population`.
13. **Outliers:** 0 outliers detected via IQR method.
14. **Genuine Relevance to "Smart Hostel Food Waste Management System":** **NOT RELEVANT.** Structurally identical to Dataset 3 (`WorldWide_foodwastage_dataset.csv`). Represents macro country-level annual stats, not micro hostel mess operational meal data.

---

## Comparative Matrix

| Feature / Criteria | Dataset 1: `Dataset Propely.csv` | Dataset 2: `food_wastage_data.csv` | Dataset 3: `WorldWide_foodwastage_dataset.csv` | Dataset 4: `global_food_wastage_dataset.csv` |
| :--- | :--- | :--- | :--- | :--- |
| **Total Records** | 2,600 | 1,782 | 5,000 | 5,000 |
| **Total Features** | 7 | 11 | 8 | 8 |
| **Missing Values** | 0 | 0 | 0 | 0 |
| **Duplicate Rows** | 0 | 164 | 0 | 0 |
| **Data Granularity** | Daily Meal & Section level | Per Catering Event | Annual Country Aggregate | Annual Country Aggregate |
| **Domain Alignment** | Institutional Canteen / Hostel Mess | Catering / Social Events | Global National Macro Statistics | Global National Macro Statistics |
| **Target Variable** | `Waste_Weight_kg` (kg) | `Wastage Food Amount` | `Total Waste in Tons` | `Total Waste (Tons)` |
| **Real ML Problem Support** | **YES (Predict meal waste in hostel canteen)** | NO (Event catering, duplicate rows) | NO (Macro national data) | NO (Macro national data) |
| **Selection Verdict** | **SELECTED (PRIMARY DATASET)** | REJECTED | REJECTED | REJECTED |

---

## Final Recommendation & Summary

```text
PRIMARY DATASET: Dataset Propely.csv
REASON FOR SELECTION: It is the only dataset that specifically models institutional mess/canteen operations at the daily meal level (Breakfast, Lunch, Dinner), canteen section level (Sections A, B, C, D), and food category level (Rice, Vegetables, Meat, Soup). Its features natively measure daily physical food waste in kilograms (Waste_Weight_kg), perfectly fitting the real-world operational prediction problem of a Smart Hostel Food Waste Management System.
NUMBER OF RECORDS: 2600
NUMBER OF FEATURES: 7 (Date, Meal, Canteen_Section, Food_Category, Waste_Weight_kg, Unit_Price_per_kg, Cost_Loss)

PROPOSED TARGET: Waste_Weight_kg (Continuous numerical target for regression predicting meal-wise food waste in kilograms)
PROPOSED FEATURES: Meal, Canteen_Section, Food_Category, Unit_Price_per_kg, DayOfWeek, Month, Day, IsWeekend

DATA QUALITY: High (100% complete, clean schema, zero missing values, zero duplicates)
MISSING VALUES: 0
DUPLICATES: 0

LIMITATIONS: 
1. Date range spans 2 months (June 11, 2025 to August 10, 2025), requiring date feature extraction (DayOfWeek, IsWeekend) rather than multi-year seasonal forecasting.
2. Does not explicitly include student head-count per meal, so predictions rely on section, meal type, food item category, price, and calendar patterns.
```
