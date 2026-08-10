import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_dataframe(df):
    """
    Performs deterministic string cleaning and date feature extraction.
    Does NOT fit imputer or encoders.
    """
    df_clean = df.copy()
    
    # 1. Date feature extraction
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day'] = df_clean['date'].dt.day
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
    df_clean['is_weekend'] = df_clean['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. String to Numeric Conversions
    df_clean['meals_served_numeric'] = df_clean['meals_served'].astype(str).str.extract(r'(\d+)').astype(float)
    df_clean['kitchen_staff_numeric'] = df_clean['kitchen_staff'].astype(str).str.extract(r'(\d+)').astype(float)
    
    # 3. Categorical Normalization (whitespace & casing)
    df_clean['staff_experience'] = df_clean['staff_experience'].astype(str).str.strip().str.title()
    df_clean['waste_category'] = df_clean['waste_category'].astype(str).str.strip().str.title()
    
    return df_clean

def run_step6_preprocessing():
    print("="*70)
    print("STEP 6: PREPROCESSING APPROVED FINAL DATASET")
    print("="*70)
    
    train_path = 'datasets/messy_food_waste/train.csv'
    test_path = 'datasets/messy_food_waste/test.csv'
    
    df_train_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path)
    
    print(f"Original Train Shape: {df_train_raw.shape}")
    print(f"Original Test Shape:  {df_test_raw.shape}")
    
    # Clean string and date fields deterministically
    df_train_clean = clean_dataframe(df_train_raw)
    df_test_clean = clean_dataframe(df_test_raw)
    
    num_cols = [
        'meals_served_numeric', 'kitchen_staff_numeric', 'temperature_C', 
        'humidity_percent', 'past_waste_kg', 'year', 'month', 'day', 
        'day_of_week', 'is_weekend', 'special_event'
    ]
    cat_cols = ['staff_experience', 'waste_category']
    target_col = 'food_waste_kg'
    
    X_train_raw = df_train_clean[num_cols + cat_cols].copy()
    y_train = df_train_clean[target_col].copy()
    X_test_raw = df_test_clean[num_cols + cat_cols].copy()
    
    # Construct reproducible scikit-learn pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    # FIT ONLY ON TRAINING DATA
    preprocessor.fit(X_train_raw)
    
    X_train_proc = preprocessor.transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    cat_feature_names = list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    
    # Save preprocessor artifact to model/final_preprocessor.pkl
    os.makedirs('model', exist_ok=True)
    with open('model/final_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
        
    print("Saved pipeline to 'model/final_preprocessor.pkl' (Preserved Step 3 'model/preprocessor.pkl').")
    print(f"Final Training Feature Matrix Shape: {X_train_proc.shape}")
    print(f"Final Test Feature Matrix Shape:     {X_test_proc.shape}")
    print(f"Total Input Features: {len(all_feature_names)}")
    print("STEP 6 PREPROCESSING COMPLETE — READY FOR STEP 7 MODEL TRAINING.")
    
    return X_train_proc, X_test_proc, y_train, all_feature_names

if __name__ == '__main__':
    run_step6_preprocessing()
