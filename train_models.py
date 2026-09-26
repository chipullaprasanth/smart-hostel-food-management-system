import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for high quality plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('model', exist_ok=True)
os.makedirs('static', exist_ok=True)

def train_and_evaluate():
    print("="*70)
    print("STEP 3: TRAINING AND EVALUATING ML MODELS")
    print("="*70)

    # 1. Load Dataset
    df = pd.read_csv('datasets/Dataset Propely.csv')
    print(f"Loaded dataset with {len(df)} records.")

    # 2. Check and remove target leakage feature
    if 'Cost_Loss' in df.columns:
        df = df.drop(columns=['Cost_Loss'])
        print("Explicitly dropped 'Cost_Loss' feature to prevent target leakage.")

    # 3. Parse Date and Extract Features
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Sort chronologically by Date
    df = df.sort_values('Date').reset_index(drop=True)

    cat_cols = ['Meal', 'Canteen_Section', 'Food_Category']
    num_cols = ['Unit_Price_per_kg', 'DayOfWeek', 'Month', 'Day', 'IsWeekend']
    target_col = 'Waste_Weight_kg'

    X = df[cat_cols + num_cols]
    y = df[target_col]

    # 4. Chronological 80/20 Train/Test Split
    split_idx = 2080  # Exactly 2080 train (80%), 520 test (20%)
    X_train_raw = X.iloc[:split_idx].copy()
    X_test_raw = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    train_dates = (df.loc[0, 'Date'].strftime('%Y-%m-%d'), df.loc[split_idx-1, 'Date'].strftime('%Y-%m-%d'))
    test_dates = (df.loc[split_idx, 'Date'].strftime('%Y-%m-%d'), df.loc[len(df)-1, 'Date'].strftime('%Y-%m-%d'))

    print(f"Chronological Split: Training records = {len(X_train_raw)} ({train_dates[0]} to {train_dates[1]})")
    print(f"Chronological Split: Testing records  = {len(X_test_raw)} ({test_dates[0]} to {test_dates[1]})")

    # 5. Fit Preprocessor ONLY on Training Data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    feature_names = cat_feature_names + num_cols

    print(f"Feature matrix shape after encoding: Train {X_train.shape}, Test {X_test.shape}")
    print(f"Total features: {len(feature_names)}")

    # Save preprocessor artifact
    with open('model/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # 6. Train Models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mean_actual = np.mean(y_test)
        mean_pred = np.mean(y_pred)
        abs_errors = np.abs(y_test - y_pred)
        max_error = np.max(abs_errors)
        min_error = np.min(abs_errors)
        neg_count = np.sum(y_pred < 0)

        results[name] = {
            'model_obj': model,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Mean_Actual': mean_actual,
            'Mean_Predicted': mean_pred,
            'Max_Error': max_error,
            'Min_Error': min_error,
            'Negative_Predictions': neg_count
        }

        # Save model pkl
        filename = name.lower().replace(' ', '_') + '.pkl'
        with open(os.path.join('model', filename), 'wb') as f:
            pickle.dump(model, f)

        print(f"[{name}] MAE: {mae:.4f} kg | RMSE: {rmse:.4f} kg | R²: {r2:.4f} ({r2*100:.2f}%)")
        print(f" Mean Actual: {mean_actual:.4f} kg | Mean Pred: {mean_pred:.4f} kg")
        print(f" Max Error: {max_error:.4f} kg | Min Error: {min_error:.4f} kg | Neg Preds: {neg_count}")

    # Determine best model based on R2 / RMSE
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    print(f"\n>>> BEST MODEL SELECTED: {best_model_name} (R² = {results[best_model_name]['R2']:.4f}) <<<")

    # 7. Generate Visualizations

    # Plot 1: Actual vs Predicted Scatter Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.6, color='skyblue' if i!=2 else 'darkblue', edgecolors='k', linewidth=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal 1:1 Line')
        ax.set_title(f"{name}\n(R² = {results[name]['R2']:.4f})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Actual Waste (kg)", fontsize=10)
        ax.set_ylabel("Predicted Waste (kg)", fontsize=10)
        ax.legend()
    plt.tight_layout()
    plt.savefig('static/actual_vs_predicted.png', dpi=300)
    plt.close()
    print("Saved 'static/actual_vs_predicted.png'")

    # Plot 2: Residual / Error Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, color='coral' if i!=2 else 'forestgreen', edgecolors='k', linewidth=0.5)
        ax.axhline(0, color='black', linestyle='--', lw=1.5)
        ax.set_title(f"Residuals: {name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Waste (kg)", fontsize=10)
        ax.set_ylabel("Residual (Actual - Predicted) (kg)", fontsize=10)
    plt.tight_layout()
    plt.savefig('static/residual_plot.png', dpi=300)
    plt.close()
    print("Saved 'static/residual_plot.png'")

    # Plot 3: Model Comparison Chart
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'MAE (kg)': [results[m]['MAE'] for m in results],
        'RMSE (kg)': [results[m]['RMSE'] for m in results],
        'R² Score': [results[m]['R2'] for m in results]
    })

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df['Model']))
    width = 0.25

    rects1 = ax1.bar(x - width, metrics_df['MAE (kg)'], width, label='MAE (kg)', color='#3498db')
    rects2 = ax1.bar(x, metrics_df['RMSE (kg)'], width, label='RMSE (kg)', color='#e74c3c')
    
    ax1.set_ylabel('Error Metric (kg)', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'], fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, metrics_df['R² Score'], width, label='R² Score', color='#2ecc71')
    ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')

    plt.title("Model Performance Comparison (MAE, RMSE, R²)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('static/model_comparison.png', dpi=300)
    plt.close()
    print("Saved 'static/model_comparison.png'")

    # Plot 4: Feature Importance Chart for Random Forest
    rf_model = results['Random Forest Regressor']['model_obj']
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='#8e44ad', edgecolor='black')
    plt.title("Random Forest Regressor - Feature Importance", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Gini Importance / Mean Decrease in Impurity", fontsize=11, fontweight='bold')
    plt.ylabel("Encoded Feature Name", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png', dpi=300)
    plt.close()
    print("Saved 'static/feature_importance.png'")

    return results, fi_df, best_model_name, feature_names

if __name__ == '__main__':
    train_and_evaluate()
