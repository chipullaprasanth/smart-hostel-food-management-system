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

os.makedirs("model", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

DATASET_PATH = "dataset/hostel_food_data.csv"

def train_and_compare():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Real-world training dataset not found at {DATASET_PATH}")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded real-world hostel dataset from {DATASET_PATH}. Shape: {df.shape}")
    
    # Clean duplicates or missing values if any
    df = df.drop_duplicates().dropna()
    
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
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
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
    print("Saved best model pipeline to model/best_model.pkl")
    
    # Save comparison report details
    report_df = pd.DataFrame(comparison_results)
    report_df.to_csv("model/model_comparison.csv", index=False)
    print("Saved comparison table to model/model_comparison.csv")
    return comparison_results, best_model_name

if __name__ == "__main__":
    train_and_compare()
