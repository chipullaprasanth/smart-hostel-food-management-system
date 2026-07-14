import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Create dataset directory if not exists
os.makedirs("dataset", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---------------- GENERATE DATASET ----------------
def generate_dataset(n_samples=1200):
    np.random.seed(42)
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    breakfasts = ["Idli Sambar", "Masala Dosa", "Poha Upma", "Aloo Paratha", "Puri Bhaji", "Bread Butter Toast"]
    lunches = ["Veg Biryani", "Chole Bhature", "Rajma Chawal", "Paneer Tadka Roti", "Dal Chawal Sabzi"]
    dinners = ["Roti Green Curry", "Chicken Tikka Rice", "Khichdi Kadhi", "Egg Curry Roti", "Dal Fry Jeera Rice"]
    
    data = []
    for i in range(n_samples):
        day = np.random.choice(days)
        month = np.random.choice(months)
        
        # Weather
        temp = round(np.random.uniform(18, 42), 1)
        rainfall = round(np.random.exponential(scale=5), 1) if np.random.rand() > 0.6 else 0.0
        humidity = int(np.random.uniform(40, 95))
        wind_speed = round(np.random.uniform(2, 28), 1)
        
        # Menu & Rating
        b_menu = np.random.choice(breakfasts)
        l_menu = np.random.choice(lunches)
        d_menu = np.random.choice(dinners)
        food_rating = round(np.random.uniform(2.5, 4.8), 1)
        
        # Baseline attendance computation (hostel specific)
        base_students = 80
        if day in ["Saturday", "Sunday"]:
            base_students -= np.random.randint(10, 20)
            
        students_present = int(np.clip(base_students + np.random.normal(0, 3), 20, 100))
        
        prev_attendance = int(np.clip(students_present + np.random.randint(-5, 5), 20, 100))
        prev_waste = round(np.random.uniform(0.5, 8.0) if students_present < 70 else np.random.uniform(0.1, 3.0), 1)
        
        data.append([
            day, month, temp, rainfall, humidity, wind_speed,
            b_menu, l_menu, d_menu, prev_attendance, prev_waste, food_rating,
            students_present
        ])
        
    df = pd.DataFrame(data, columns=[
        "Day", "Month", "Temperature", "Rainfall", "Humidity", "Wind_Speed",
        "Breakfast_Menu", "Lunch_Menu", "Dinner_Menu", "Previous_Attendance", "Previous_Waste", "Food_Rating",
        "Students_Present"
    ])
    df.to_csv("dataset/hostel_food_data.csv", index=False)
    print(f"Generated dataset with {n_samples} records at dataset/hostel_food_data.csv")
    return df

# ---------------- PREPROCESS AND TRAIN ----------------
def train_and_compare():
    df = generate_dataset()
    
    # Split features and target
    X = df.drop(columns=["Students_Present"])
    y = df["Students_Present"]
    
    categorical_cols = [
        "Day", "Month", "Breakfast_Menu", "Lunch_Menu", "Dinner_Menu"
    ]
    numeric_cols = [
        "Temperature", "Rainfall", "Humidity", "Wind_Speed",
        "Previous_Attendance", "Previous_Waste", "Food_Rating"
    ]
    
    # Build transformation pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_r2 = -float('inf')
    best_model_name = ""
    best_pipeline = None
    comparison_results = []
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        comparison_results.append({
            "Model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        })
        
        print(f"=== {name} ===")
        print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline
            
    # Save best model pipeline
    joblib.dump(best_pipeline, "model/best_model.pkl")
    print(f"Successfully selected best model: {best_model_name} (R2: {best_r2:.4f})")
    print("Saved pipeline package to model/best_model.pkl")
    
    # Save comparison report details as simple metadata
    report_df = pd.DataFrame(comparison_results)
    report_df.to_csv("model/model_comparison.csv", index=False)

if __name__ == "__main__":
    train_and_compare()
