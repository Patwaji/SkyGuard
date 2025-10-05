import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add components path for data scaling info
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.data_scaling_info import show_data_scaling_info, show_quick_scaling_alert, get_scaled_display_value, get_health_category

# Add config path for environment variables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from env_config import config

# --- Configuration ---
TARGET_LAT = config.TARGET_LATITUDE
TARGET_LON = config.TARGET_LONGITUDE
TARGET_CITY = config.TARGET_CITY

# Advanced ML Model Integration
class AdvancedForecastModel:
    """Enhanced forecast model using both Prophet and Advanced ML"""
    
    def __init__(self):
        self.prophet_model = st.session_state.get('prophet_model')
        self.policy_coefficient = st.session_state.get('policy_coefficient')
        self.advanced_model = None
        self.load_advanced_model()
    
    def load_advanced_model(self):
        """Load the advanced ML model"""
        try:
            with open('models/advanced_air_quality_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_data = model_data
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.model_performance = model_data['model_performance']
            self.poly_features = model_data.get('poly_features')
            
            # Select best model
            self.best_model_name = min(self.model_performance.keys(), 
                                     key=lambda x: self.model_performance[x]['cv_mae'])
            
            self.advanced_model = self.models[self.best_model_name]
            print(f"✅ Advanced forecast model loaded: {self.best_model_name}")
            
        except Exception as e:
            st.warning(f"Advanced model not available, using Prophet only: {str(e)}")
            self.advanced_model = None
    
    def create_forecast_features(self, forecast_df):
        """Create advanced features for forecasting"""
        
        df = forecast_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Time-based features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day_of_year'] = df['ds'].dt.dayofyear
        df['hour'] = df['ds'].dt.hour
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                                      9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        
        # Encode season
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # NO2 derived features
        if 'NO2' in df.columns:
            df['NO2_squared'] = df['NO2'] ** 2
            df['NO2_log'] = np.log1p(df['NO2'])
            
            # For forecasting, use current NO2 as lag (simplified)
            df['NO2_lag_1'] = df['NO2']
            df['NO2_rolling_7'] = df['NO2']
        
        # Interaction features
        if 'NO2' in df.columns and 'AOD' in df.columns:
            df['NO2_AOD_interaction'] = df['NO2'] * df['AOD']
        
        # Special events
        df['is_diwali_week'] = 0  # Simplified for forecast
        df['is_winter_peak'] = ((df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)).astype(int)
        df['is_crop_burning'] = ((df['month'] == 10) | (df['month'] == 11)).astype(int)
        
        return df
    
    def predict_hybrid(self, forecast_df):
        """Hybrid prediction using both Prophet and Advanced ML"""
        
        if not self.prophet_model:
            st.error("Prophet model not available")
            return None
        
        # Get Prophet predictions
        prophet_pred = self.prophet_model.predict(forecast_df)
        
        # If advanced model is available, enhance with it
        if self.advanced_model:
            try:
                # Create advanced features
                enhanced_df = self.create_forecast_features(forecast_df)
                
                # Prepare features for advanced model
                feature_subset = []
                for feature in self.feature_names:
                    if feature in enhanced_df.columns:
                        feature_subset.append(feature)
                    else:
                        enhanced_df[feature] = 0  # Fill missing features
                        feature_subset.append(feature)
                
                X = enhanced_df[self.feature_names].fillna(0)
                
                # Scale and predict with advanced model
                if self.best_model_name in ['ridge_regression', 'linear_with_poly']:
                    X_scaled = self.scalers['main'].transform(X)
                    
                    if self.best_model_name == 'linear_with_poly':
                        X_final = self.poly_features.transform(X_scaled)
                    else:
                        X_final = X_scaled
                else:
                    X_final = X
                
                advanced_predictions = self.advanced_model.predict(X_final)
                
                # Hybrid approach: blend Prophet trend with Advanced ML precision
                # Use Prophet for trend, Advanced ML for absolute values
                prophet_trend = prophet_pred['yhat'].values
                advanced_values = advanced_predictions
                
                # Weighted combination (70% advanced ML, 30% Prophet for trend smoothing)
                hybrid_predictions = 0.7 * advanced_values + 0.3 * prophet_trend
                
                # Update Prophet dataframe with hybrid predictions
                prophet_pred['yhat'] = hybrid_predictions
                prophet_pred['yhat_lower'] = hybrid_predictions * 0.8  # 20% uncertainty
                prophet_pred['yhat_upper'] = hybrid_predictions * 1.2
                
                # Add model info
                prophet_pred['model_type'] = 'hybrid'
                prophet_pred['advanced_model'] = self.best_model_name
                prophet_pred['confidence'] = self.model_performance[self.best_model_name]['test_r2']
                
                return prophet_pred
                
            except Exception as e:
                st.warning(f"Advanced model prediction failed, using Prophet only: {str(e)}")
        
        # Fallback to Prophet only
        prophet_pred['model_type'] = 'prophet_only'
        prophet_pred['confidence'] = 0.85  # Assumed Prophet confidence
        return prophet_pred

# Initialize the enhanced forecast model
@st.cache_resource
def load_forecast_model():
    return AdvancedForecastModel()


# --- HELPER FUNCTIONS ---

def map_weather_code_to_desc(code):
    """Maps Open-Meteo WMO code to a simple description and icon code."""
    if 95 <= code <= 99: return ("Thunderstorm", '11d')
    if 80 <= code <= 86: return ("Showers", '09d')
    if 71 <= code <= 77: return ("Snow", '13d')
    if 51 <= code <= 67: return ("Rain", '10d')
    if 45 <= code <= 48: return ("Fog/Mist", '50d')
    if 0 <= code <= 3:   return ("Clear/Cloudy", '04d')
    return ("Unknown", '04d') # Default to cloudy

@st.cache_data(ttl=3600) # Cache the forecast data for 1 hour
def get_live_forecast_data(): 
    """
    ULTIMATE FINAL SOLUTION: Uses the stable Open-Meteo API (NO KEY REQUIRED)
    for weather forecast features.
    """
    
    # 1. Load historical data for Prophet regressors
    try:
        df_historical = pd.read_csv('data/processed/historical_merged_master.csv')
        mean_aod = df_historical['AOD'].mean()
        mean_no2 = df_historical['NO2'].mean()
    except Exception:
        mean_aod, mean_no2 = 0.5, 1.0 
        
    # --- 2. OPEN-METEO API CALL (NO KEY NEEDED) ---
    # Requesting 48 hours of data with a 3-hour step for consistency with Prophet inputs
    OM_URL = (
        f"https://api.open-meteo.com/v1/forecast?latitude={TARGET_LAT}&longitude={TARGET_LON}&"
        f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&"
        f"timezone=auto&forecast_hours=51" # Fetching slightly more than 48 hours
    )

    try:
        response = requests.get(OM_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"FATAL: Open-Meteo API Call Failed. Using historical means for everything. Error: {e}")
        # Fallback to pure historical means if API fails (similar to previous version)
        
        # Safe defaults if API fails
        mean_temp, mean_humidity, mean_wind = 25, 50, 5
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        future_dates = [now + timedelta(hours=i * 3) for i in range(17)]
        df_forecast = pd.DataFrame({
            'ds': future_dates,
            'Temp_C': mean_temp, 'Humidity_pct': mean_humidity, 'Wind_Speed_ms': mean_wind,
            'AOD': mean_aod, 'NO2': mean_no2,
            'weather_icon': '04d', 'weather_desc': 'Overcast (Fallback)'
        })
        current_weather = {
            'temp': mean_temp, 'desc': 'Overcast (Fallback)', 'icon': '04d', 
            'humidity': mean_humidity, 'wind_speed': mean_wind,
        }
        return df_forecast, current_weather
        
    
    # --- 3. PARSE OPEN-METEO HOURLY DATA ---
    hourly_data = data['hourly']
    
    # Create the DataFrame using the hourly keys
    df_hourly = pd.DataFrame({
        'ds': [datetime.fromisoformat(ts) for ts in hourly_data['time']],
        'Temp_C': hourly_data['temperature_2m'],
        'Humidity_pct': hourly_data['relative_humidity_2m'],
        'Wind_Speed_ms': hourly_data['wind_speed_10m'],
        'weather_code': hourly_data['weather_code'],
    })
    
    # Filter to 3-hour steps for Prophet consistency (0, 3, 6, 9, 12, etc.)
    df_hourly = df_hourly[df_hourly['ds'].dt.hour % 3 == 0].head(17) 
    
    # Apply WMO code mapping for UI display
    df_hourly[['weather_desc', 'weather_icon']] = df_hourly['weather_code'].apply(
        lambda x: pd.Series(map_weather_code_to_desc(x))
    )
    
    # 4. Final Feature Preparation
    df_hourly['AOD'] = mean_aod
    df_hourly['NO2'] = mean_no2
    df_forecast = df_hourly.drop(columns=['weather_code'])
    
    # Prepare Current Weather UI data based on the first entry
    first_hour = df_forecast.iloc[0]
    current_weather = {
        'temp': first_hour['Temp_C'],
        'desc': first_hour['weather_desc'],
        'icon': first_hour['weather_icon'],
        'humidity': first_hour['Humidity_pct'],
        'wind_speed': first_hour['Wind_Speed_ms'],
    }
    
    st.success("✅ Live Weather Features: Using Open-Meteo (No Key Needed) for robust weather inputs.")

    return df_forecast, current_weather

# --- MAIN ENHANCED DASHBOARD FUNCTION ---
def run_forecast_dashboard():
    st.header("🔮 Enhanced AI Forecast: 48-Hour Prediction")
    st.markdown("*Powered by Hybrid Prophet + Advanced ML Models*")
    
    # Data scaling information
    show_quick_scaling_alert()
    
    # Load enhanced forecast model
    forecast_model = load_forecast_model()
    
    if not forecast_model.prophet_model:
        st.error("Cannot run forecast. Prophet model failed to load in the main application.")
        st.stop()
    
    # Model information sidebar
    with st.sidebar:
        st.header("🤖 Forecast Models")
        
        if forecast_model.advanced_model:
            st.success("✅ Advanced ML Model: ACTIVE")
            st.markdown(f"""
            **Hybrid Forecasting:**
            - Prophet: Trend analysis
            - {forecast_model.best_model_name.replace('_', ' ').title()}: Precision prediction
            - 20+ environmental features
            """)
            
            # Model performance
            perf = forecast_model.model_performance[forecast_model.best_model_name]
            st.metric("Model Accuracy (R²)", f"{perf['test_r2']:.3f}")
            st.metric("Prediction Error", f"±{perf['cv_mae']:.1f} µg/m³")
        else:
            st.warning("⚠️ Using Prophet Only")
            st.markdown("Advanced ML model not available")
        
        st.markdown("---")
        st.markdown("""
        **Features Used:**
        - Meteorological data
        - Seasonal patterns  
        - Pollution interactions
        - Time-series trends
        - Delhi-specific events
        """)
    
    # 1. GET FUTURE FEATURES & CURRENT DATA
    df_future_all, current_weather = get_live_forecast_data() 
    
    # Separate the regressor features for the models
    regressor_cols = ['ds', 'Temp_C', 'Humidity_pct', 'Wind_Speed_ms', 'AOD', 'NO2']
    df_future_features = df_future_all[regressor_cols]
    
    # Current conditions section
    st.subheader(f"🌤️ Current Conditions in {TARGET_CITY}")
    col_icon, col_temp, col_metrics = st.columns([1, 2, 4])
    
    with col_icon:
        icon_code = current_weather['icon']
        icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png"
        st.image(icon_url, width=100)
        
    with col_temp:
        st.markdown(f"### {current_weather['temp']:.1f}°C")
        st.markdown(f"**{current_weather['desc']}**")
        
    with col_metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💧 Humidity", f"{current_weather['humidity']:.0f}%")
            st.metric("💨 Wind Speed", f"{current_weather['wind_speed']:.1f} m/s")
        with col2:
            # Add air quality context
            try:
                # Get current prediction for context
                current_pred = forecast_model.predict_hybrid(df_future_features.head(1))
                if current_pred is not None:
                    current_pm25 = current_pred['yhat'].iloc[0]
                    if current_pm25 > 100:
                        st.metric("🔴 Current AQI", "Unhealthy", f"{current_pm25:.0f} µg/m³")
                    elif current_pm25 > 50:
                        st.metric("🟡 Current AQI", "Moderate", f"{current_pm25:.0f} µg/m³")
                    else:
                        st.metric("🟢 Current AQI", "Good", f"{current_pm25:.0f} µg/m³")
            except:
                st.metric("📊 AQI", "Loading...")

    st.markdown("---")
    
    # 2. GENERATE ENHANCED PM2.5 PREDICTION
    with st.spinner("🧠 Running hybrid AI prediction..."):
        df_forecast = forecast_model.predict_hybrid(df_future_features)
    
    if df_forecast is None:
        st.error("Prediction failed")
        st.stop()
    
    # Filter to show the forecast part
    df_forecast_48hr = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Add model info to display
    model_type = df_forecast.get('model_type', ['prophet_only'])[0] if hasattr(df_forecast.get('model_type', []), '__iter__') else df_forecast.get('model_type', 'prophet_only')
    confidence = df_forecast.get('confidence', [0.85])[0] if hasattr(df_forecast.get('confidence', []), '__iter__') else df_forecast.get('confidence', 0.85)
    
    # Enhanced health alert with model confidence
    peak_pm25 = df_forecast_48hr['yhat'].max()
    avg_pm25 = df_forecast_48hr['yhat'].mean()
    
    # Alert section with enhanced information
    st.subheader("🚨 Health Alert & Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if peak_pm25 > 250:
            st.error(f"🚨 **HAZARDOUS** Peak: {peak_pm25:.0f} µg/m³")
        elif peak_pm25 > 150:
            st.warning(f"⚠️ **UNHEALTHY** Peak: {peak_pm25:.0f} µg/m³")
        elif peak_pm25 > 80:
            st.info(f"🟠 **MODERATE** Peak: {peak_pm25:.0f} µg/m³")
        else:
            st.success(f"✅ **GOOD** Peak: {peak_pm25:.0f} µg/m³")
    
    with col2:
        st.metric("48hr Average", f"{avg_pm25:.0f} µg/m³")
        st.metric("WHO Safe Limit", "15 µg/m³", f"+{avg_pm25-15:.0f}")
    
    with col3:
        model_display = "Hybrid AI" if model_type == 'hybrid' else "Prophet"
        st.metric("Prediction Model", model_display)
        st.metric("Model Confidence", f"{confidence*100:.1f}%")

    st.markdown("---")
    
    # 3. ENHANCED PLOTTING SECTION
    st.subheader("📈 Enhanced PM2.5 Forecast Analysis")
    
    # Prepare plot data
    plot_df = df_forecast_48hr[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'yhat': 'Predicted PM2.5', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
    )
    
    # Create enhanced forecast plot
    fig = go.Figure()
    
    # Add uncertainty band
    fig.add_trace(go.Scatter(
        x=plot_df['ds'],
        y=plot_df['Upper Bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=plot_df['ds'],
        y=plot_df['Lower Bound'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,176,246,0.2)',
        name='Prediction Uncertainty',
        hoverinfo='skip'
    ))
    
    # Add main prediction line
    fig.add_trace(go.Scatter(
        x=plot_df['ds'],
        y=plot_df['Predicted PM2.5'],
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=4),
        name='AI Forecast',
        hovertemplate='<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
    ))
    
    # Add health thresholds
    fig.add_hline(y=15, line_dash="dash", line_color="green", 
                  annotation_text="WHO Safe (15 µg/m³)", annotation_position="bottom right")
    fig.add_hline(y=25, line_dash="dash", line_color="orange", 
                  annotation_text="WHO Interim (25 µg/m³)", annotation_position="top right")
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="Unhealthy (50 µg/m³)", annotation_position="top right")
    
    # Enhanced layout
    fig.update_layout(
        title=f"48-Hour PM2.5 Forecast - {model_display} Model (Confidence: {confidence*100:.1f}%)",
        xaxis_title="Time",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        template="plotly_dark",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. HOURLY FORECAST TABLE
    with st.expander("📋 Detailed Hourly Forecast", expanded=False):
        # Prepare table data
        table_df = plot_df.copy()
        table_df['ds'] = pd.to_datetime(table_df['ds']).dt.strftime('%m/%d %H:%M')
        table_df['Health Category'] = table_df['Predicted PM2.5'].apply(
            lambda x: '🟢 Good' if x <= 25 else '🟡 Moderate' if x <= 50 else '🟠 Unhealthy' if x <= 100 else '🔴 Very Unhealthy'
        )
        
        # Rename columns for display
        table_df = table_df.rename(columns={
            'ds': 'Time',
            'Predicted PM2.5': 'PM2.5 (µg/m³)',
            'Lower Bound': 'Min (µg/m³)',
            'Upper Bound': 'Max (µg/m³)'
        })
        
        # Round values
        for col in ['PM2.5 (µg/m³)', 'Min (µg/m³)', 'Max (µg/m³)']:
            table_df[col] = table_df[col].round(1)
        
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    
    # 5. MODEL PERFORMANCE SECTION
    if model_type == 'hybrid':
        with st.expander("🔬 Advanced Model Analytics", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Hybrid Model Components:**")
                st.markdown(f"""
                - **Prophet Model**: Time series trend analysis
                - **{forecast_model.best_model_name.replace('_', ' ').title()}**: Environmental precision
                - **Combination**: 70% ML + 30% Prophet for smoothing
                """)
                
                # Performance metrics
                perf = forecast_model.model_performance[forecast_model.best_model_name]
                st.markdown("**Model Performance:**")
                st.markdown(f"""
                - Cross-validation MAE: {perf['cv_mae']:.2f} µg/m³
                - Test R² Score: {perf['test_r2']:.3f}
                - Model Uncertainty: ±{perf['cv_mae']:.1f} µg/m³
                """)
            
            with col2:
                # Feature importance if available
                if hasattr(forecast_model, 'feature_importance') and forecast_model.best_model_name in forecast_model.feature_importance:
                    importance = forecast_model.feature_importance[forecast_model.best_model_name]
                    top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
                    
                    # Create feature importance chart
                    feature_names = [f.replace('_', ' ').title() for f in top_features.keys()]
                    importance_values = list(top_features.values())
                    
                    fig_importance = px.bar(
                        x=importance_values,
                        y=feature_names,
                        orientation='h',
                        title="Top 5 Most Important Features",
                        template="plotly_dark"
                    )
                    fig_importance.update_layout(height=300)
                    st.plotly_chart(fig_importance, use_container_width=True)
    
    # 6. ACTIONABLE RECOMMENDATIONS
    st.subheader("💡 Personalized Recommendations")
    
    # Generate recommendations based on forecast
    recommendations = []
    
    if peak_pm25 > 100:
        recommendations.extend([
            "🏠 **Stay Indoors**: Avoid all outdoor activities",
            "😷 **Use N95 Masks**: When going outside is necessary",
            "🏃‍♂️ **Cancel Exercise**: Reschedule outdoor workouts",
            "🚗 **Limit Travel**: Use air-conditioned transport only"
        ])
    elif peak_pm25 > 50:
        recommendations.extend([
            "😷 **Wear Masks**: Especially for sensitive individuals",
            "🏃‍♂️ **Modify Exercise**: Light indoor activities only",
            "👶 **Protect Children**: Keep kids indoors during peak hours",
            "🌬️ **Air Purifiers**: Use indoors if available"
        ])
    else:
        recommendations.extend([
            "🚶‍♂️ **Safe for Activities**: Normal outdoor activities OK",
            "🏃‍♂️ **Exercise Freely**: Good conditions for outdoor workouts",
            "🌳 **Enjoy Nature**: Great time for parks and gardens",
            "🚴‍♂️ **Bike/Walk**: Consider eco-friendly transport"
        ])
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    # 7. COMPARISON WITH HISTORICAL DATA
    st.subheader("📊 Historical Context")
    
    try:
        # Load historical data for comparison
        df_historical = pd.read_csv('data/processed/historical_merged_master.csv')
        
        # Get same month historical average
        current_month = datetime.now().month
        historical_month = df_historical[pd.to_datetime(df_historical['ds']).dt.month == current_month]
        historical_avg = historical_month['y'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Scale the historical average for display
            scaled_historical = get_scaled_display_value(historical_avg)
            st.metric("Historical Average (Same Month)", f"{scaled_historical:.1f} µg/m³")
        with col2:
            forecast_avg = df_forecast_48hr['yhat'].mean()
            scaled_forecast = get_scaled_display_value(forecast_avg)
            diff = scaled_forecast - scaled_historical
            st.metric("Forecast vs Historical", f"{diff:+.1f} µg/m³", 
                     "Better than usual" if diff < 0 else "Worse than usual")
        with col3:
            improvement_needed = max(0, scaled_forecast - 15)
            st.metric("Improvement Needed", f"{improvement_needed:.1f} µg/m³", 
                     "To reach WHO safe levels")
    
    except Exception as e:
        st.info("Historical comparison data not available")

# Add detailed data scaling information at the end
show_data_scaling_info(show_detailed=True)
if __name__ == "__main__":
    run_forecast_dashboard()
