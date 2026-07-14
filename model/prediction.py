import os
import pandas as pd
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")

def get_prediction_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)

def predict_attendance(input_data):
    """
    input_data: dict containing keys:
        Day, Month, Temperature, Rainfall, Humidity, Wind_Speed,
        Breakfast_Menu, Lunch_Menu, Dinner_Menu, Previous_Attendance, Previous_Waste, Food_Rating
    """
    pipeline = get_prediction_pipeline()
    
    # Cast to DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure correct columns order
    expected_cols = [
        "Day", "Month", "Temperature", "Rainfall", "Humidity", "Wind_Speed",
        "Breakfast_Menu", "Lunch_Menu", "Dinner_Menu", "Previous_Attendance", "Previous_Waste", "Food_Rating"
    ]
    df = df[expected_cols]
    
    # Predict
    predicted_val = pipeline.predict(df)[0]
    return int(max(10, round(predicted_val)))
