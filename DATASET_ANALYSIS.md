# Dataset Analysis & Selection Report
**Project:** Smart Hostel Food Waste Management System  
**Document:** DATASET_ANALYSIS.md  

---

## 1. Executive Summary & Dataset Inventory

Four real-world food waste datasets were evaluated to determine the optimal architecture for machine learning forecasting and multi-dimensional dashboard analytics in the **Smart Hostel Food Waste Management System**:

| Dataset Name | Records | Columns | Primary Focus / Domain | Quality & Integrity |
| :--- | :--- | :--- | :--- | :--- |
| **`hostel_food_data.csv`** | 1,200 | 13 | Hostel Student Attendance & Weather-Meal Factors | 0 Duplicates, 0 Nulls, High relevance |
| **`food_wastage_data.csv`** | 1,782 | 11 | Catering / Event Food Waste & Guests | 164 Duplicates removed (1,618 unique), 0 Nulls |
| **`Dataset Propely.csv`** | 2,600 | 7 | Canteen Section Meal Waste & Unit Costs | 0 Duplicates, 0 Nulls, Granular meal logs |
| **`global_food_wastage_dataset.csv`** | 5,000 | 8 | Macro Country-Level Per-Capita Waste & Losses | 0 Duplicates, 0 Nulls, 20 Countries (2018-2024) |

---

## 2. Individual Dataset Analysis

### Dataset 1: `hostel_food_data.csv` (Primary ML Dataset)
* **Schema:** `['Day', 'Month', 'Temperature', 'Rainfall', 'Humidity', 'Wind_Speed', 'Breakfast_Menu', 'Lunch_Menu', 'Dinner_Menu', 'Previous_Attendance', 'Previous_Waste', 'Food_Rating', 'Students_Present']`
* **Target Feature:** `Students_Present` (Forecasting Student Attendance to accurately prepare food portions).
* **Key Features:** Environmental conditions (Temperature, Rainfall, Humidity, Wind Speed), Categorical Factors (Day, Month, Breakfast/Lunch/Dinner Menus), Operational Factors (Previous Attendance, Previous Waste, Food Rating).
* **Missing Values & Duplicates:** 0 missing values, 0 duplicate rows.
* **Hostel Capacity Scaling Strategy:** Scaled from baseline (80 students) to **500–1000 hostel student capacity range** (500 to 950 present) using proportional feature transformation and noise injection, preserving weather/day/menu relationships.

### Dataset 2: `food_wastage_data.csv` (Catering & Guest Waste Analytics)
* **Schema:** `['Type of Food', 'Number of Guests', 'Event Type', 'Quantity of Food', 'Storage Conditions', 'Purchase History', 'Seasonality', 'Preparation Method', 'Geographical Location', 'Pricing', 'Wastage Food Amount']`
* **Purpose:** Provides empirical relationship curves between guest count, food prepared, food type (Meat, Veg, etc.), and wastage amounts across seasons.
* **Preprocessing:** 164 duplicate rows identified and eliminated. Categorical fields normalized.

### Dataset 3: `Dataset Propely.csv` (Canteen Section & Meal Loss Analytics)
* **Schema:** `['Date', 'Meal', 'Canteen_Section', 'Food_Category', 'Waste_Weight_kg', 'Unit_Price_per_kg', 'Cost_Loss']`
* **Purpose:** Meal-by-meal breakdown (Breakfast, Lunch, Dinner), canteen section efficiency (Sections A, B, C, D), and exact monetary cost loss calculations (`Waste_Weight_kg * Unit_Price_per_kg`).
* **Preprocessing:** Date parsing (`YYYY-MM-DD`), cost validation, section grouping.

### Dataset 4: `global_food_wastage_dataset.csv` (Macro Global Benchmark Analytics)
* **Schema:** `['Country', 'Year', 'Food Category', 'Total Waste (Tons)', 'Economic Loss (Million $)', 'Avg Waste per Capita (Kg)', 'Population (Million)', 'Household Waste (%)']`
* **Purpose:** Macro-level comparative analytics and global benchmark metrics on per-capita waste and economic losses.
* **Preprocessing:** Country filtering (highlighting India & regional benchmarks), metric unit standardization.

---

## 3. Primary Dataset Selection for Machine Learning

### **Selected ML Dataset:** `hostel_food_data.csv` (Scaled to 500-1000 Capacity)

### **Rationale & Justification:**
1. **Direct Alignment with Hostel Operations:** The dataset explicitly targets **Hostel Student Attendance (`Students_Present`)**, which is the core driver of daily kitchen preparation for Rice, Dal, Curry, and Chapati.
2. **Comprehensive Feature Set:** It combines meteorological variables (temperature, humidity, rainfall, wind speed), temporal variables (day, month), food quality ratings, past waste feedback, and specific meal menus.
3. **Optimized for Food Waste Minimization:** Predicting exact student attendance before cooking prevents over-preparation, directly cutting down hostel food waste at the source.

---

## 4. Multi-Dataset Utilization Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │     SMART HOSTEL FOOD MANAGEMENT SYSTEM      │
                               └──────────────────────┬───────────────────────┘
                                                      │
         ┌────────────────────────┬───────────────────┴───────────────────┬────────────────────────┐
         │                        │                                       │                        │
  ┌──────▼──────────────┐  ┌──────▼──────────────┐                ┌───────▼──────────────┐  ┌──────▼──────────────┐
  │ hostel_food_data.csv│  │food_wastage_data.csv│                │ Dataset Propely.csv  │  │global_food_wastage.csv│
  └──────┬──────────────┘  └──────┬──────────────┘                └───────┬──────────────┘  └──────┬──────────────┘
         │                        │                                       │                        │
  ┌──────▼──────────────┐  ┌──────▼──────────────┐                ┌───────▼──────────────┐  ┌──────▼──────────────┐
  │   ML PREDICTION     │  │  EVENT & CATERING   │                │ CANTEEN & COST LOSS  │  │ GLOBAL BENCHMARK    │
  │   ENGINE (REGRESSION)│ │  WASTE ANALYTICS    │                │ ANALYTICS DASHBOARD  │  │ COMPARISON DASHBOARD│
  └─────────────────────┘  └─────────────────────┘                └──────────────────────┘  └─────────────────────┘
```

---

## 5. Preprocessing & Feature Engineering Pipeline

### Step 1: Data Cleaning & Deduplication
* Removed 164 duplicate rows from `food_wastage_data.csv`.
* Ensured zero null values across all 4 datasets.

### Step 2: Date Standardization & Dynamic Date Integration
* Integrated dynamic date parsing for calendar inputs.
* Automatically derived `Month`, `Day` of week, `Season` (Summer, Monsoon, Winter, Spring), and `Festival` status directly from selected date pickers.

### Step 3: Feature Encoding & Scaling
* Numerical features (`Temperature`, `Rainfall`, `Humidity`, `Wind_Speed`, `Previous_Attendance`, `Previous_Waste`, `Food_Rating`) scaled using `StandardScaler`.
* Categorical features (`Day`, `Month`, `Breakfast_Menu`, `Lunch_Menu`, `Dinner_Menu`) one-hot encoded using `OneHotEncoder(handle_unknown='ignore')`.

### Step 4: Final ML Dataset Schema (500–1000 Capacity Scale)
* **Numeric Inputs (7):** `Temperature`, `Rainfall`, `Humidity`, `Wind_Speed`, `Previous_Attendance`, `Previous_Waste`, `Food_Rating`
* **Categorical Inputs (5):** `Day`, `Month`, `Breakfast_Menu`, `Lunch_Menu`, `Dinner_Menu`
* **Target Output (1):** `Students_Present` (Range: 500 – 950 students)
