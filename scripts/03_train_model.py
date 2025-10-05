import pandas as pd
import pickle
from prophet import Prophet
from sklearn.linear_model import LinearRegression
import os
import sys

# Add config path for environment variables and get project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Set up data and model directories
MODEL_DIR = os.path.join(project_root, 'models')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
MASTER_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'historical_merged_master.csv')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def train_prophet_model(df):
    """Trains the Prophet model using PM2.5 as target and other features as regressors."""
    print("--- 1. Training Prophet Prediction Model ---")
    
    # 1. Prepare data and features
    # 'y' is the target (PM2.5/AQI proxy)
    # 'ds' is the date column (required by Prophet)
    
    # Exogenous variables (features) for prediction
    regressors = ['AOD', 'NO2', 'Temp_C', 'Humidity_pct', 'Wind_Speed_ms']
    
    # 2. Initialize and configure Prophet
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        # Set a low seasonality prior scale to let the regressors drive more of the prediction
        seasonality_prior_scale=0.1 
    )
    
    # 3. Add Regressors (must be added before calling .fit)
    for reg in regressors:
        model.add_regressor(reg)
        
    # 4. Fit the model
    model.fit(df)
    
    # 5. Save the model object
    model_path = os.path.join(MODEL_DIR, 'prophet_predictor_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"✅ Prophet Model saved to: {model_path}")
    return model


def train_policy_model(df):
    """
    Trains a simple Linear Regression model to find the NO2 -> PM2.5 sensitivity.
    This provides the coefficient for the policy scenario tool.
    """
    print("\n--- 2. Training Policy Advisor Regression Model ---")
    
    # We use a simple linear relationship between our policy proxy (NO2) and the outcome (y/PM2.5)
    X = df[['NO2']].values
    y = df['y'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # The coefficient is the "sensitivity": change in PM2.5 per unit change in NO2
    policy_coefficient = model.coef_[0]
    
    # 4. Save the model and the coefficient
    model_path = os.path.join(MODEL_DIR, 'policy_regression_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'coefficient': policy_coefficient}, f)
        
    print(f"✅ Policy Model saved to: {model_path}")
    print(f"   Sensitivity (PM2.5 per unit NO2): {policy_coefficient:.4f}")
    
    return policy_coefficient


if __name__ == '__main__':
    print("--- Starting Phase 3: Model Training ---")
    
    try:
        # Load the clean, merged dataset
        df_master = pd.read_csv(MASTER_DATA_PATH)
        
        # Train Prediction Model (Prophet)
        prophet_model = train_prophet_model(df_master)
        
        # Train Policy Model (Linear Regression)
        policy_coef = train_policy_model(df_master)
        
        print("\n--- Phase 3 Complete. Ready to build the Streamlit Dashboard. ---")

    except Exception as e:
        print(f"\nFATAL ERROR DURING MODEL TRAINING: {e}")
        print("Ensure 'data/processed/historical_merged_master.csv' is present and valid.")