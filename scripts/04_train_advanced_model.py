import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pickle
from datetime import datetime
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Add config path for environment variables and get project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Set up data and model directories
MODEL_DIR = os.path.join(project_root, 'models')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
os.makedirs(MODEL_DIR, exist_ok=True)

class AdvancedAirQualityModel:
    """
    Advanced ML model for air quality prediction with multiple algorithms
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.poly_features = None
        
    def create_advanced_features(self, df):
        """Create advanced features from the basic data"""
        
        # Convert date to datetime if it's not already
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Time-based features
            df['year'] = df['ds'].dt.year
            df['month'] = df['ds'].dt.month
            df['day_of_year'] = df['ds'].dt.dayofyear
            df['quarter'] = df['ds'].dt.quarter
            df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                          3: 'Spring', 4: 'Spring', 5: 'Spring',
                                          6: 'Summer', 7: 'Summer', 8: 'Summer',
                                          9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
            
            # Encode season
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            
            # Cyclical features for month and day of year
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Meteorological features (if available)
        if 'temp' in df.columns:
            df['temp_squared'] = df['temp'] ** 2
            df['temp_log'] = np.log1p(df['temp'] + 50)  # Add 50 to handle negative temps
        
        if 'humidity' in df.columns:
            df['humidity_squared'] = df['humidity'] ** 2
        
        if 'wind_speed' in df.columns:
            df['wind_speed_log'] = np.log1p(df['wind_speed'])
            df['wind_speed_squared'] = df['wind_speed'] ** 2
        
        # Pollution interaction features
        if 'NO2' in df.columns and 'AOD' in df.columns:
            df['NO2_AOD_interaction'] = df['NO2'] * df['AOD']
        
        if 'NO2' in df.columns:
            df['NO2_squared'] = df['NO2'] ** 2
            df['NO2_log'] = np.log1p(df['NO2'])
            
            # Lagged features (if we have enough data)
            if len(df) > 7:
                df['NO2_lag_1'] = df['NO2'].shift(1)
                df['NO2_lag_7'] = df['NO2'].shift(7)  # Weekly lag
                df['NO2_rolling_7'] = df['NO2'].rolling(window=7).mean()
                df['NO2_rolling_30'] = df['NO2'].rolling(window=30).mean()
        
        # Holiday/Special events (Delhi specific)
        if 'ds' in df.columns:
            # Approximate Diwali dates (varies each year)
            diwali_dates = ['2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12']
            df['is_diwali_week'] = 0
            
            for diwali_date in diwali_dates:
                try:
                    diwali = pd.to_datetime(diwali_date)
                    df.loc[(df['ds'] >= diwali - pd.Timedelta(days=3)) & 
                           (df['ds'] <= diwali + pd.Timedelta(days=3)), 'is_diwali_week'] = 1
                except:
                    continue
            
            # Winter months (high pollution season)
            df['is_winter_peak'] = ((df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)).astype(int)
            
            # Crop burning season
            df['is_crop_burning'] = ((df['month'] == 10) | (df['month'] == 11)).astype(int)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        
        # Create advanced features
        df_enhanced = self.create_advanced_features(df.copy())
        
        # Select features for modeling
        feature_columns = []
        
        # Basic features
        if 'NO2' in df_enhanced.columns:
            feature_columns.extend(['NO2', 'NO2_squared', 'NO2_log'])
            
            # Add lagged features if available
            if 'NO2_lag_1' in df_enhanced.columns:
                feature_columns.extend(['NO2_lag_1', 'NO2_rolling_7'])
        
        # Meteorological features
        for col in ['temp', 'humidity', 'wind_speed', 'pressure', 'AOD']:
            if col in df_enhanced.columns:
                feature_columns.append(col)
                if f'{col}_squared' in df_enhanced.columns:
                    feature_columns.append(f'{col}_squared')
                if f'{col}_log' in df_enhanced.columns:
                    feature_columns.append(f'{col}_log')
        
        # Time features
        time_features = ['month', 'day_of_year', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        for col in time_features:
            if col in df_enhanced.columns:
                feature_columns.append(col)
        
        # Season dummies
        season_cols = [col for col in df_enhanced.columns if col.startswith('season_')]
        feature_columns.extend(season_cols)
        
        # Special event features
        event_features = ['is_diwali_week', 'is_winter_peak', 'is_crop_burning']
        for col in event_features:
            if col in df_enhanced.columns:
                feature_columns.append(col)
        
        # Interaction features
        if 'NO2_AOD_interaction' in df_enhanced.columns:
            feature_columns.append('NO2_AOD_interaction')
        
        # Remove duplicates and ensure all features exist
        feature_columns = list(set(feature_columns))
        feature_columns = [col for col in feature_columns if col in df_enhanced.columns]
        
        return df_enhanced[feature_columns].fillna(0), feature_columns
    
    def train_models(self, df, target_column='y'):
        """Train multiple ML models"""
        
        print("🚀 Training Advanced Air Quality Models...")
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        y = df[target_column]
        
        # Remove any rows with missing target values
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📊 Using {len(feature_names)} features: {feature_names}")
        print(f"📈 Training on {len(X)} samples")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        self.feature_names = feature_names
        
        # Define models to train
        models_to_train = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            ),
            'ridge_regression': Ridge(alpha=1.0),
            'linear_with_poly': Ridge(alpha=1.0)  # Will use polynomial features
        }
        
        # Train models
        for model_name, model in models_to_train.items():
            print(f"🔧 Training {model_name}...")
            
            if model_name == 'linear_with_poly':
                # Create polynomial features for linear model
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_train_poly = poly.fit_transform(X_train_scaled)
                X_test_poly = poly.transform(X_test_scaled)
                
                model.fit(X_train_poly, y_train)
                y_pred = model.predict(X_test_poly)
                
                self.poly_features = poly
            else:
                if model_name in ['ridge_regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            
            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score
            if model_name == 'linear_with_poly':
                cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='neg_mean_absolute_error')
            elif model_name in ['ridge_regression']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            
            cv_mae = -cv_scores.mean()
            
            self.models[model_name] = model
            self.model_performance[model_name] = {
                'test_mae': mae,
                'test_r2': r2,
                'cv_mae': cv_mae,
                'cv_std': cv_scores.std()
            }
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[model_name] = importance_dict
            
            print(f"   ✅ {model_name}: R² = {r2:.3f}, MAE = {mae:.2f}")
        
        # Select best model based on cross-validation MAE
        best_model_name = min(self.model_performance.keys(), 
                             key=lambda x: self.model_performance[x]['cv_mae'])
        
        print(f"🏆 Best model: {best_model_name}")
        print(f"   📊 Cross-validation MAE: {self.model_performance[best_model_name]['cv_mae']:.2f}")
        print(f"   📊 Test R²: {self.model_performance[best_model_name]['test_r2']:.3f}")
        
        return best_model_name
    
    def predict(self, input_data, model_name=None):
        """Make predictions with the trained models"""
        
        if model_name is None:
            # Use the best performing model
            model_name = min(self.model_performance.keys(), 
                           key=lambda x: self.model_performance[x]['cv_mae'])
        
        model = self.models[model_name]
        
        # Prepare input features
        if isinstance(input_data, dict):
            # Single prediction
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        X, _ = self.prepare_features(input_df)
        
        # Ensure we have all required features
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        X = X[self.feature_names]
        
        # Scale features
        if model_name in ['ridge_regression', 'linear_with_poly']:
            X_scaled = self.scalers['main'].transform(X)
            
            if model_name == 'linear_with_poly':
                X_final = self.poly_features.transform(X_scaled)
            else:
                X_final = X_scaled
        else:
            X_final = X
        
        # Make prediction
        prediction = model.predict(X_final)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def save_model(self, filepath):
        """Save the trained models"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'poly_features': self.poly_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Models saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_performance = model_data['model_performance']
        self.poly_features = model_data.get('poly_features')
        
        print(f"📂 Models loaded from {filepath}")

def main():
    """Train and save the advanced models"""
    
    # Load your data
    print("📂 Loading data...")
    data_path = os.path.join(PROCESSED_DATA_DIR, 'historical_merged_master.csv')
    df = pd.read_csv(data_path)
    
    print(f"📊 Data shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Initialize and train the advanced model
    advanced_model = AdvancedAirQualityModel()
    best_model = advanced_model.train_models(df)
    
    # Save the models
    model_path = os.path.join(MODEL_DIR, 'advanced_air_quality_model.pkl')
    advanced_model.save_model(model_path)
    
    # Test prediction
    print("\n🧪 Testing prediction...")
    test_input = {
        'NO2': 45.0,
        'temp': 25.0,
        'humidity': 60.0,
        'wind_speed': 5.0,
        'AOD': 0.5,
        'ds': '2023-12-01'
    }
    
    prediction = advanced_model.predict(test_input)
    print(f"🎯 Test prediction: {prediction:.2f} µg/m³")
    
    # Print model performance summary
    print("\n📊 Model Performance Summary:")
    for model_name, performance in advanced_model.model_performance.items():
        print(f"   {model_name}:")
        print(f"      Cross-val MAE: {performance['cv_mae']:.2f} ± {performance['cv_std']:.2f}")
        print(f"      Test R²: {performance['test_r2']:.3f}")
    
    # Print feature importance for best model
    if best_model in advanced_model.feature_importance:
        print(f"\n🎯 Feature Importance ({best_model}):")
        importance = advanced_model.feature_importance[best_model]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance_val in sorted_features[:10]:
            print(f"   {feature}: {importance_val:.3f}")

if __name__ == "__main__":
    main()
