import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

def load_and_preprocess(filepath='datasets/Dataset Propely.csv', split_method='chronological'):
    """
    Loads 'Dataset Propely.csv', parses dates, extracts temporal features,
    verifies target leakage (excluding Cost_Loss), encodes categorical features,
    and returns train/test splits.
    """
    # 1. Load dataset
    df = pd.read_csv(filepath)
    print(f"Original Records: {len(df)}")
    
    # 2. Target Leakage Verification
    # Cost_Loss = Waste_Weight_kg * Unit_Price_per_kg. Must be dropped to prevent target leakage.
    if 'Cost_Loss' in df.columns:
        df = df.drop(columns=['Cost_Loss'])
        print("Dropped 'Cost_Loss' feature to prevent target leakage.")
        
    # 3. Parse Date & Feature Extraction
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 4. Define Feature Groups
    cat_cols = ['Meal', 'Canteen_Section', 'Food_Category']
    num_cols = ['Unit_Price_per_kg', 'DayOfWeek', 'Month', 'Day', 'IsWeekend']
    target_col = 'Waste_Weight_kg'
    
    X = df[cat_cols + num_cols]
    y = df[target_col]
    
    # 5. Build ColumnTransformer Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )
    
    # 6. Fit & Transform Features
    X_processed = preprocessor.fit_transform(X)
    cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    feature_names = cat_feature_names + num_cols
    
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # 7. Train / Test Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    
    if split_method == 'chronological':
        X_train = X_processed_df.iloc[:split_idx]
        X_test = X_processed_df.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        train_date_range = (df.loc[0, 'Date'].strftime('%Y-%m-%d'), df.loc[split_idx-1, 'Date'].strftime('%Y-%m-%d'))
        test_date_range = (df.loc[split_idx, 'Date'].strftime('%Y-%m-%d'), df.loc[len(df)-1, 'Date'].strftime('%Y-%m-%d'))
        print(f"Chronological Split (80/20): Train range {train_date_range}, Test range {test_date_range}")
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.20, random_state=42)
        print("Random 80/20 Split applied.")
        
    print(f"Final Feature Matrix Shape: {X_processed_df.shape}")
    print(f"X_train Shape: {X_train.shape}, X_test Shape: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': feature_names
    }

if __name__ == '__main__':
    data_dict = load_and_preprocess()
    print("Preprocessing completed successfully. Pipeline is ready for ML training in Step 3.")
