# AI-Based Smart Hostel Food Management & Waste Reduction System

A comprehensive major project solution combining machine learning predictive analytics, rule-based kitchen portion recommendation engines, and a web application interface to reduce food wastage in student hostel mess facilities (capacity 500–1000 students).

---

## 1. System Architecture & Dual ML Model Workflows

The system incorporates two distinct, specialized Machine Learning workflows alongside an operational recommendation engine:

### Workflow A: Food Waste Analysis Model (Step 7–9 ML Audit Pipeline)
- **Target Variable**: `food_waste_kg` (Daily food waste generated in Kg)
- **Model Architecture**: Ridge Regression ($\alpha=1.0$) with standard feature preprocessor pipeline
- **Validation Methodology**: 5-Fold Chronological `TimeSeriesSplit`
- **Verified Performance**: $R^2 = 0.7959$, $\text{MAE} = 2.3966$ Kg, $\text{RMSE} = 2.9871$ Kg
- **Serialized Artifact**: `model/final_food_waste_model.pkl` & `model/preprocessor.pkl`
- **Audit Status**: 64/64 Reproducibility and Validation Integrity Checks **PASS**

### Workflow B: Web Application Attendance Forecasting Model
- **Target Variable**: `Students_Present` (Daily student mess turnout)
- **Model Architecture**: Evaluated Random Forest, Decision Tree, and Linear Regression models on `dataset/hostel_food_data.csv`
- **Selected Best Model**: Random Forest Regressor ($R^2 = 0.9120$, $\text{MAE} = 16.752$ students, $\text{RMSE} = 20.739$ students)
- **Serialized Artifact**: `model/best_model.pkl`
- **Purpose**: Powers real-time student headcount forecasting in web application endpoints (`/predict`, `/dashboard`).

### Recommendation Engine
- **Logic**: Rule-based quantity calculation deriving meal breakdown (Rice, Dal, Curry, Chapati) from predicted attendance, season, temperature, rainfall, humidity, and festival parameters.

---

## 2. Security Architecture & System Hardening

1. **Authentication & Public Registration Security**:
   - Public user registration (`/register`) strictly forces `role='user'`. Role parameters supplied by the browser/form are ignored.
   - Admin privileges can only be assigned through initial database seeding or secure administrator setup.
2. **Secret Management**:
   - `app.secret_key` reads from `os.environ.get("SECRET_KEY")`. A documented fallback is provided for local development.
3. **Production Debug Control**:
   - Flask debug mode is disabled by default in production execution (`debug=False`). Can be toggled for development via `export FLASK_DEBUG=True`.
4. **CSRF Protection**:
   - All state-changing POST forms pass session-backed CSRF tokens (`csrf_token`).
5. **File Upload Security**:
   - Menu uploads (`/upload_menu`) are secured using `werkzeug.utils.secure_filename`, a 5 MB file size limit (`MAX_CONTENT_LENGTH`), and strict extension whitelisting (`CSV`, `PNG`, `JPG`, `PDF`, `DOC`, `DOCX`).

---

## 3. Technology Stack & Dependencies

- **Web Framework**: Flask 3.x
- **Data & ML**: Python 3.10+ / 3.14, `scikit-learn`, `pandas`, `numpy`, `joblib`
- **Database**: SQLite (`food_data.db`) with fallback and full MySQL compatibility (`pymysql`, `schema.sql`, `database/hostel.sql`)
- **PDF & Excel Export**: `reportlab`, `openpyxl`
- **WSGI Server**: `gunicorn`

---

## 4. Environment Setup & Installation

### Step 1: Clone & Navigate
```bash
cd smart-hostel-food-management-system
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables (Optional for Development)
```bash
# Windows PowerShell
$env:SECRET_KEY="your-production-secure-key"
$env:FLASK_DEBUG="False"

# Linux / macOS
export SECRET_KEY="your-production-secure-key"
export FLASK_DEBUG="False"
```

---

## 5. Running the Application

### Local Development Server
```bash
python app.py
```
The server will start at `http://127.0.0.1:5000`.

### Production Execution (WSGI Gunicorn)
```bash
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

---

## 6. Running ML Audit & Reproducibility Verification

To execute the existing Step 9 ML audit verification test suite:
```bash
python step9_audit.py
```
This executes all 64 automated checks validating dataset integrity, chronological cross-validation, target leakage prevention, and model artifact reproducibility.

---

## 7. Project Directory Layout

```
.
├── app.py                      # Core Flask Application & Secured Routes
├── database.py                 # Parameterized DB Access & Table Schema Migrations
├── schema.sql                  # MySQL Schema Definition Script
├── database/hostel.sql         # Alternative MySQL Database Setup DDL
├── model/
│   ├── best_model.pkl          # Serialized Web App Attendance Forecast Model
│   ├── final_food_waste_model.pkl # Serialized Step 7/8/9 Food Waste Model
│   ├── preprocessor.pkl        # Feature Transformer Pipeline for Step 7/8/9
│   ├── prediction.py           # Web Application Inference Pipeline
│   └── train_model.py          # Attendance Model Training Script
├── datasets/
│   └── messy_food_waste/       # TimeSeriesSplit Verified ML Train/Test Sets
├── outputs/                    # Audit CSV Output Files & Predictions
├── reports/                    # Generated Visual Diagnostic Plots
├── static/                     # CSS, JS, Uploaded Menu Artifacts
├── templates/                  # Jinja HTML UI Templates
├── requirements.txt            # Python Runtime Dependencies
├── step7_training.py           # Food Waste Model Training Script
├── step8_analysis.py           # Model Interpretation & Error Analysis Script
├── step9_audit.py              # ML Audit & Reproducibility Test Suite
└── FINAL_APPLICATION_AUDIT.md  # Application Hardening & Final Audit Report
```
