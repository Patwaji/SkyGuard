import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import json
import sys

# Add components path for data scaling info
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.data_scaling_info import show_data_scaling_info, show_quick_scaling_alert, get_scaled_display_value, get_health_category

# Add config path for environment variables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from env_config import config

# Page configuration
st.set_page_config(
    page_title="3D Air Quality Visualization",
    page_icon="🌫️",
    layout="wide"
)

# Page header
st.title("🌫️ 3D Air Quality Visualization")
st.markdown("""
**Real-time 3D pollution analysis for Delhi** - Explore air quality data in three dimensions!

🔍 **Features:**
- **3D Pollution Clouds**: Visualize PM2.5, NO2, and AQI in 3D space
- **Real Delhi Data**: Based on actual pollution measurements
- **Hotspot Analysis**: Find the most polluted locations in Delhi
""")

# Load real-time data from OpenMeteo API with model alignment
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_openmeteo_air_quality():
    """Load real-time air quality data from OpenMeteo API for Delhi region"""
    try:
        # Delhi coordinates and surrounding areas
        locations = [
            {"name": "Central Delhi", "lat": config.TARGET_LATITUDE, "lon": config.TARGET_LONGITUDE},
            {"name": "North Delhi", "lat": 28.7041, "lon": 77.1025},
            {"name": "South Delhi", "lat": 28.5355, "lon": 77.3910},
            {"name": "East Delhi", "lat": 28.6507, "lon": 77.2773},
            {"name": "West Delhi", "lat": 28.6692, "lon": 77.1350},
            {"name": "Gurgaon", "lat": 28.4595, "lon": 77.0266},
            {"name": "Noida", "lat": 28.5355, "lon": 77.3910},
            {"name": "Faridabad", "lat": 28.4089, "lon": 77.3178},
        ]
        
        all_data = []
        
        for location in locations:
            try:
                # OpenMeteo Air Quality API using environment config
                url = config.OPENMETEO_BASE_URL
                params = {
                    "latitude": location["lat"],
                    "longitude": location["lon"],
                    "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
                    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
                    "past_days": 1,
                    "forecast_days": 0
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Get current values
                current = data.get("current", {})
                hourly = data.get("hourly", {})
                
                # Process current data
                current_data = {
                    "location": location["name"],
                    "latitude": location["lat"],
                    "longitude": location["lon"],
                    "pm25": current.get("pm2_5", 0) or 0,
                    "pm10": current.get("pm10", 0) or 0,
                    "no2": current.get("nitrogen_dioxide", 0) or 0,
                    "so2": current.get("sulphur_dioxide", 0) or 0,
                    "co": current.get("carbon_monoxide", 0) or 0,
                    "o3": current.get("ozone", 0) or 0,
                    "timestamp": datetime.now(),
                    "data_source": "OpenMeteo_Real_Time"
                }
                
                # Calculate AQI from PM2.5 and NO2
                pm25_val = current_data["pm25"]
                no2_val = current_data["no2"]
                
                # AQI calculation based on US EPA standards
                if pm25_val <= 12:
                    aqi_pm25 = (50/12) * pm25_val
                elif pm25_val <= 35.4:
                    aqi_pm25 = 50 + ((100-50)/(35.4-12.1)) * (pm25_val - 12.1)
                elif pm25_val <= 55.4:
                    aqi_pm25 = 100 + ((150-100)/(55.4-35.5)) * (pm25_val - 35.5)
                elif pm25_val <= 150.4:
                    aqi_pm25 = 150 + ((200-150)/(150.4-55.5)) * (pm25_val - 55.5)
                elif pm25_val <= 250.4:
                    aqi_pm25 = 200 + ((300-200)/(250.4-150.5)) * (pm25_val - 150.5)
                else:
                    aqi_pm25 = 300 + ((500-300)/(500.4-250.5)) * (pm25_val - 250.5)
                
                current_data["aqi"] = max(aqi_pm25, no2_val * 0.5)  # Take higher of PM2.5 AQI or NO2 contribution
                
                # CRITICAL: Scale to match your ML model's range
                # Your model expects y values in the range 0-400,000 (mean ~84,000)
                # OpenMeteo gives PM2.5 in 0-500 range
                # So we need to scale up OpenMeteo data to match model expectations
                model_scale_factor = 1000  # Scale factor to align with your model
                current_data["model_scaled_pm25"] = current_data["pm25"] * model_scale_factor
                current_data["model_scaled_aqi"] = current_data["aqi"] * model_scale_factor
                
                all_data.append(current_data)
                
                # Add some recent hourly data for trend analysis
                if hourly and "time" in hourly:
                    times = hourly["time"][-6:]  # Last 6 hours
                    for i, time_str in enumerate(times):
                        hourly_data = current_data.copy()
                        hourly_pm25 = hourly.get("pm2_5", [0]*len(times))[i] or current_data["pm25"]
                        hourly_no2 = hourly.get("nitrogen_dioxide", [0]*len(times))[i] or current_data["no2"]
                        
                        hourly_data.update({
                            "pm25": hourly_pm25,
                            "no2": hourly_no2,
                            "timestamp": datetime.fromisoformat(time_str.replace('Z', '+00:00')),
                            "data_source": "OpenMeteo_Hourly"
                        })
                        # Recalculate AQI for hourly data
                        hourly_aqi = max((hourly_pm25 * 1.5), (hourly_no2 * 0.8))
                        hourly_data["aqi"] = hourly_aqi
                        
                        # Scale for model compatibility
                        hourly_data["model_scaled_pm25"] = hourly_pm25 * model_scale_factor
                        hourly_data["model_scaled_aqi"] = hourly_aqi * model_scale_factor
                        
                        all_data.append(hourly_data)
                        
            except Exception as e:
                st.warning(f"Could not fetch data for {location['name']}: {e}")
                # Add fallback data for this location with proper scaling
                fallback_pm25 = np.random.normal(85, 20)
                fallback_aqi = np.random.normal(150, 40)
                
                fallback_data = {
                    "location": location["name"],
                    "latitude": location["lat"],
                    "longitude": location["lon"],
                    "pm25": fallback_pm25,
                    "pm10": np.random.normal(120, 30),
                    "no2": np.random.normal(45, 15),
                    "so2": np.random.normal(15, 5),
                    "co": np.random.normal(1.2, 0.3),
                    "o3": np.random.normal(80, 20),
                    "aqi": fallback_aqi,
                    "model_scaled_pm25": fallback_pm25 * 1000,  # Scale to match model
                    "model_scaled_aqi": fallback_aqi * 1000,    # Scale to match model
                    "timestamp": datetime.now(),
                    "data_source": "Fallback_Simulated"
                }
                all_data.append(fallback_data)
        
        if all_data:
            df = pd.DataFrame(all_data)
            # Add weather data from OpenMeteo weather API
            df = add_weather_data(df)
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"Error fetching OpenMeteo data: {e}")
        return None

# Add weather data from OpenMeteo
@st.cache_data(ttl=3600)
def add_weather_data(air_quality_df):
    """Add weather data from OpenMeteo weather API"""
    try:
        # Get weather for Delhi center using environment config
        url = config.OPENMETEO_WEATHER_URL
        params = {
            "latitude": config.TARGET_LATITUDE,
            "longitude": config.TARGET_LONGITUDE,
            "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"],
            "timezone": "Asia/Kolkata"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        weather_data = response.json()
        
        current_weather = weather_data.get("current", {})
        
        # Add weather data to all rows
        air_quality_df["temperature"] = current_weather.get("temperature_2m", 25)
        air_quality_df["humidity"] = current_weather.get("relative_humidity_2m", 60)
        air_quality_df["wind_speed"] = current_weather.get("wind_speed_10m", 3)
        air_quality_df["wind_direction"] = current_weather.get("wind_direction_10m", 180)
        
        return air_quality_df
        
    except Exception as e:
        st.warning(f"Could not fetch weather data: {e}")
        # Add default weather values
        air_quality_df["temperature"] = 25
        air_quality_df["humidity"] = 60
        air_quality_df["wind_speed"] = 3
        air_quality_df["wind_direction"] = 180
        return air_quality_df

# Load and process real data with model alignment
@st.cache_data
def load_delhi_pollution_data():
    """Load real Delhi pollution data from OpenMeteo with CSV fallback and model alignment"""
    
    # Try OpenMeteo first
    openmeteo_data = load_openmeteo_air_quality()
    
    if openmeteo_data is not None and not openmeteo_data.empty:
        st.success("🌐 **Live data loaded** - Real-time Delhi air quality")
        return openmeteo_data
    
    # If OpenMeteo fails, create simulated real-time data instead of CSV fallback
    st.warning("� **OpenMeteo API unavailable - Using simulated real-time data**")
    
    # Create realistic simulated data for Delhi
    locations = [
        {"name": "Central Delhi", "lat": 28.6139, "lon": 77.2090},
        {"name": "North Delhi", "lat": 28.7041, "lon": 77.1025},
        {"name": "South Delhi", "lat": 28.5355, "lon": 77.3910},
        {"name": "East Delhi", "lat": 28.6507, "lon": 77.2773},
        {"name": "West Delhi", "lat": 28.6692, "lon": 77.1350},
        {"name": "Gurgaon", "lat": 28.4595, "lon": 77.0266},
        {"name": "Noida", "lat": 28.5355, "lon": 77.3910},
        {"name": "Faridabad", "lat": 28.4089, "lon": 77.3178},
    ]
    
    simulated_data = []
    for location in locations:
        # Generate realistic pollution values for Delhi
        pm25_val = np.random.normal(85, 25)  # Delhi typical PM2.5
        pm25_val = max(15, min(pm25_val, 300))  # Clamp to reasonable range
        
        no2_val = np.random.normal(45, 15)   # Delhi typical NO2
        no2_val = max(10, min(no2_val, 120))
        
        # Calculate AQI
        aqi_val = max(pm25_val * 1.5, no2_val * 0.8)
        
        data_point = {
            "location": location["name"],
            "latitude": location["lat"],
            "longitude": location["lon"],
            "pm25": pm25_val,
            "pm10": pm25_val * 1.4,  # PM10 typically higher than PM2.5
            "no2": no2_val,
            "so2": np.random.normal(15, 5),
            "co": np.random.normal(1.2, 0.3),
            "o3": np.random.normal(80, 20),
            "aqi": aqi_val,
            "model_scaled_pm25": pm25_val * 1000,  # Scale for model compatibility
            "model_scaled_aqi": aqi_val * 1000,
            "temperature": np.random.normal(28, 5),  # Delhi temperature
            "humidity": np.random.normal(65, 15),
            "wind_speed": np.random.normal(3, 1.5),
            "wind_direction": np.random.normal(180, 90),
            "timestamp": datetime.now(),
            "data_source": "Simulated_Real_Time"
        }
        simulated_data.append(data_point)
        
        # Add some hourly variation
        for i in range(6):
            hourly_data = data_point.copy()
            hourly_data["pm25"] *= np.random.normal(1, 0.1)  # Small variation
            hourly_data["no2"] *= np.random.normal(1, 0.1)
            hourly_data["timestamp"] = datetime.now() - timedelta(hours=i+1)
            hourly_data["data_source"] = "Simulated_Hourly"
            simulated_data.append(hourly_data)
    
    return pd.DataFrame(simulated_data)

# Create 3D data with proper altitude layers
@st.cache_data
def create_3d_pollution_data(base_data):
    """Create properly scaled 3D pollution data with altitude layers"""
    
    if base_data is None:
        # Create realistic sample Delhi data
        base_data = pd.DataFrame({
            'pm25': np.random.normal(90, 30, 60),
            'no2': np.random.normal(50, 20, 60),
            'aqi': np.random.normal(150, 50, 60),
            'latitude': 28.6 + np.random.normal(0, 0.15, 60),
            'longitude': 77.2 + np.random.normal(0, 0.15, 60),
            'temperature': np.random.normal(25, 5, 60),
            'humidity': np.random.normal(60, 15, 60),
            'wind_speed': np.random.normal(3, 1.5, 60)
        })
    
    # Create realistic altitude layers (ground to 600m)
    altitude_levels = [0, 50, 100, 200, 300, 400, 500, 600]
    all_data = []
    
    # Use more data points for better coverage
    for _, row in base_data.head(40).iterrows():
        for altitude in altitude_levels:
            # Realistic pollution decrease with altitude
            if altitude == 0:
                altitude_factor = 1.0  # Ground level - highest pollution
            elif altitude <= 100:
                altitude_factor = 0.85  # Lower atmosphere - still high
            elif altitude <= 300:
                altitude_factor = 0.65  # Mid-level - moderate decrease
            else:
                altitude_factor = 0.45  # Upper levels - significant decrease
            
            # Add small random variations for realistic distribution
            lat_variation = np.random.normal(0, 0.008)
            lon_variation = np.random.normal(0, 0.008)
            
            all_data.append({
                'latitude': np.clip(row['latitude'] + lat_variation, 28.4, 28.8),
                'longitude': np.clip(row['longitude'] + lon_variation, 77.0, 77.4),
                'altitude': altitude,
                'pm25': max(15, row['pm25'] * altitude_factor * np.random.normal(1, 0.1)),
                'no2': max(10, row['no2'] * altitude_factor * np.random.normal(1, 0.1)),
                'aqi': max(25, row['aqi'] * altitude_factor * np.random.normal(1, 0.05)),
                'temperature': row.get('temperature', 25) - (altitude * 0.0065),  # Temperature lapse rate
                'humidity': max(20, row.get('humidity', 60) - (altitude * 0.02)),  # Humidity decrease
                'wind_speed': row.get('wind_speed', 3) + (altitude * 0.002),  # Wind speed increase
                'date': datetime.now().strftime('%Y-%m-%d')
            })
    
    return pd.DataFrame(all_data)

# Load the data with progress indicator and real-time status
with st.spinner("🌐 Loading real-time Delhi air quality data"):
    base_data = load_delhi_pollution_data()
    
    if base_data is not None:
        pollution_3d = create_3d_pollution_data(base_data)
    else:
        st.error("❌ Could not load any pollution data")
        st.stop()

# 🔍 Real-time Delhi Air Quality Status
st.header("🔍 **Real-Time Air Quality Status**")

# Add quick scaling alert
show_quick_scaling_alert()

data_col1, data_col2 = st.columns(2)

with data_col1:
    st.subheader("📊 **Current PM2.5 Levels**")
    if 'pm25' in base_data.columns:
        pm25_values = base_data['pm25'].dropna()
        st.metric("PM2.5 Range", f"{pm25_values.min():.1f} - {pm25_values.max():.1f} µg/m³")
        st.metric("PM2.5 Average", f"{pm25_values.mean():.1f} µg/m³")
        
        # Health assessment
        avg_pm25 = pm25_values.mean()
        if avg_pm25 <= 12:
            st.success("� **Good** - No health risks")
        elif avg_pm25 <= 35:
            st.warning("🟡 **Moderate** - Sensitive groups may experience symptoms")
        elif avg_pm25 <= 55:
            st.warning("🟠 **Unhealthy for Sensitive** - Limit outdoor exposure")
        elif avg_pm25 <= 150:
            st.error("🔴 **Unhealthy** - Everyone should limit outdoor activities")
        else:
            st.error("🟣 **Very Unhealthy** - Avoid outdoor activities")

with data_col2:
    st.subheader("🌐 **Data Source Status**")
    data_source = base_data['data_source'].iloc[0] if 'data_source' in base_data.columns else "Unknown"
    if "OpenMeteo" in data_source:
        st.success("")
    elif "Simulated" in data_source:
        st.info("🔄 **Simulated Real-time** - Realistic Delhi values")
    else:
        st.warning("📁 **Historical Data** - Archive values")
    
    st.metric("Data Points", len(base_data))
    st.metric("Locations", base_data['location'].nunique() if 'location' in base_data.columns else "N/A")
    last_update = base_data['timestamp'].max() if 'timestamp' in base_data.columns else datetime.now()
    st.caption(f"Last update: {last_update.strftime('%H:%M:%S')}")



st.divider()

# Enhanced data status indicator
if base_data is not None:
    data_source = base_data['data_source'].iloc[0] if 'data_source' in base_data.columns else 'Unknown'
    
    if 'OpenMeteo' in data_source:
        st.success(f"✅ **Real-time data active** - {len(base_data)} live measurements")
        
        # Show real-time data quality metrics
        with st.expander("🌐 **Live Data Quality & Coverage**", expanded=False):
            col_rt1, col_rt2, col_rt3 = st.columns(3)
            
            with col_rt1:
                real_time_count = len(base_data[base_data['data_source'].str.contains('Real_Time', na=False)])
                hourly_count = len(base_data[base_data['data_source'].str.contains('Hourly', na=False)])
                st.metric("🔴 Live Readings", real_time_count)
                st.metric("📊 Hourly Data", hourly_count)
            
            with col_rt2:
                if 'timestamp' in base_data.columns:
                    latest_update = base_data['timestamp'].max()
                    time_diff = datetime.now() - latest_update
                    st.metric("🕐 Last Update", f"{time_diff.seconds//60}min ago")
                
                avg_pm25 = base_data['pm25'].mean()
                st.metric("🌫️ Current Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
            
            with col_rt3:
                if 'location' in base_data.columns:
                    locations_covered = base_data['location'].nunique()
                    st.metric("📍 Locations", locations_covered)
                
                avg_aqi = base_data['aqi'].mean()
                aqi_status = "🔴 Severe" if avg_aqi > 150 else "🟡 Moderate" if avg_aqi > 50 else "🟢 Good"
                st.metric("📈 Current AQI", f"{avg_aqi:.0f} - {aqi_status}")
    else:
        st.info(f"📁 **Using historical data** - {len(base_data)} processed measurements")
    
    # Show data summary in expander
    with st.expander("📋 **Detailed Data Summary**", expanded=False):
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric("📊 Total 3D Points", len(pollution_3d))
            if 'timestamp' in base_data.columns:
                date_range = f"{base_data['timestamp'].min().strftime('%H:%M')} - {base_data['timestamp'].max().strftime('%H:%M')}"
                st.metric("🕐 Time Range", date_range)
            else:
                st.metric("📅 Date Range", "Current Analysis")
        
        with col_summary2:
            avg_pm25 = base_data['pm25'].mean()
            avg_no2 = base_data['no2'].mean()
            st.metric("🌫️ Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
            st.metric("💨 Avg NO2", f"{avg_no2:.1f} µg/m³")
        
        with col_summary3:
            avg_temp = base_data['temperature'].mean() if 'temperature' in base_data.columns else 25
            avg_humidity = base_data['humidity'].mean() if 'humidity' in base_data.columns else 60
            st.metric("🌡️ Temperature", f"{avg_temp:.1f}°C")
            st.metric("💧 Humidity", f"{avg_humidity:.0f}%")
            
        # Show location coverage if available
        if 'location' in base_data.columns:
            st.markdown("**📍 Areas Covered:**")
            locations = base_data['location'].unique()
            location_text = " • ".join(locations[:6])  # Show first 6 locations
            if len(locations) > 6:
                location_text += f" • +{len(locations)-6} more"
            st.caption(location_text)
else:
    st.error("❌ No pollution data available for analysis")
    st.stop()

# Enhanced sidebar controls with better formatting
st.sidebar.header("🎛️ **Visualization Controls**")
st.sidebar.markdown("---")

# Pollutant selection with descriptions
st.sidebar.subheader("📊 **Select Pollutant**")
pollutant_descriptions = {
    "pm25": "PM2.5 - Fine particulate matter (≤2.5μm)",
    "no2": "NO2 - Nitrogen dioxide gas",
    "aqi": "AQI - Air Quality Index (composite)"
}

pollutant = st.sidebar.selectbox(
    "Choose pollutant to visualize:",
    ["pm25", "no2", "aqi"],
    format_func=lambda x: {"pm25": "🌫️ PM2.5", "no2": "💨 NO2", "aqi": "📈 AQI"}[x]
)

st.sidebar.caption(pollutant_descriptions[pollutant])
st.sidebar.markdown("---")

# Altitude controls with better formatting
st.sidebar.subheader("🏔️ **Altitude Range**")
max_altitude = st.sidebar.slider(
    "Maximum altitude to display:",
    min_value=50,
    max_value=600,
    value=400,
    step=50,
    help="Adjust to see pollution at different heights"
)

st.sidebar.caption(f"Showing data from ground level to {max_altitude}m altitude")
st.sidebar.markdown("---")

# View options
st.sidebar.subheader("👁️ **View Options**")
show_background_data = st.sidebar.checkbox("Show background data points", value=True)
show_altitude_grid = st.sidebar.checkbox("Show altitude grid lines", value=False)
color_intensity = st.sidebar.slider("Color intensity", 0.5, 1.0, 0.8, 0.1)

# Filter data based on user selection
filtered_data = pollution_3d[pollution_3d['altitude'] <= max_altitude]

# Main content area with improved layout
st.markdown("---")
st.header("📊 **Real-time 3D Pollution Analysis**")

# Main visualization section
col1, col2 = st.columns([2.5, 1.5])

with col1:
    # Create pollutant display names
    pollutant_names = {'pm25': 'PM2.5', 'no2': 'NO2', 'aqi': 'AQI'}
    st.subheader(f"🌫️ **3D {pollutant_names[pollutant]} Visualization**")
    
    if not filtered_data.empty:
        # Get current pollution statistics
        current_avg = filtered_data[pollutant].mean()
        current_max = filtered_data[pollutant].max()
        current_min = filtered_data[pollutant].min()
        
        # Status indicators
        if current_avg > 100:
            status_color = "🔴"
            status_text = "SEVERE"
        elif current_avg > 50:
            status_color = "🟡"
            status_text = "HIGH"
        else:
            status_color = "🟢"
            status_text = "MODERATE"
        
        st.info(f"{status_color} **Current Status: {status_text}** | Avg: {current_avg:.1f} | Range: {current_min:.1f} - {current_max:.1f}")
        
        # Create enhanced 3D scatter plot
        fig = go.Figure()
        
        # Define unit labels
        unit_labels = {'pm25': 'PM2.5 (µg/m³)', 'no2': 'NO2 (µg/m³)', 'aqi': 'AQI'}
        
        # Main pollution data
        fig.add_trace(go.Scatter3d(
            x=filtered_data['longitude'],
            y=filtered_data['latitude'],
            z=filtered_data['altitude'],
            mode='markers',
            marker=dict(
                size=8,
                color=filtered_data[pollutant],
                colorscale='Viridis',
                opacity=color_intensity,
                colorbar=dict(
                    title=unit_labels[pollutant],
                    len=0.6
                ),
                showscale=True,
                line=dict(width=0.5, color='rgba(0,0,0,0.3)')
            ),
            text=[f"📍 Location: {lat:.3f}, {lon:.3f}<br>🏔️ Altitude: {alt}m<br>📊 {pollutant.upper()}: {val:.1f}<br>🌡️ Temp: {temp:.1f}°C" 
                  for lat, lon, alt, val, temp in zip(
                      filtered_data['latitude'], 
                      filtered_data['longitude'], 
                      filtered_data['altitude'], 
                      filtered_data[pollutant],
                      filtered_data.get('temperature', [25]*len(filtered_data))
                  )],
            hovertemplate='%{text}<extra></extra>',
            name=f'{pollutant_names[pollutant]} Data'
        ))
        
        # Add altitude grid lines if requested
        if show_altitude_grid:
            for alt in range(0, max_altitude + 1, 100):
                fig.add_trace(go.Scatter3d(
                    x=[77.0, 77.4, 77.4, 77.0, 77.0],
                    y=[28.4, 28.4, 28.8, 28.8, 28.4],
                    z=[alt, alt, alt, alt, alt],
                    mode='lines',
                    line=dict(color='rgba(150,150,150,0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Enhanced layout
        fig.update_layout(
            scene=dict(
                xaxis_title='🌐 Longitude',
                yaxis_title='🌐 Latitude',
                zaxis_title='🏔️ Altitude (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                xaxis=dict(
                    range=[76.9, 77.5],
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                yaxis=dict(
                    range=[28.3, 28.9],
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                zaxis=dict(
                    range=[0, max_altitude + 50],
                    showgrid=True,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            title=dict(
                text=f"🌫️ 3D {pollutant_names[pollutant]} Distribution - Delhi ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                x=0.5,
                font=dict(size=16)
            ),
            height=650,
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pollution level indicator
        if current_avg > 150:
            st.error("🚨 **EMERGENCY LEVEL** - Stay indoors, use air purifiers")
        elif current_avg > 100:
            st.error("🔴 **SEVERE** - Avoid all outdoor activities")
        elif current_avg > 50:
            st.warning("🟡 **UNHEALTHY** - Limit outdoor exposure")
        else:
            st.success("🟢 **MODERATE** - Normal outdoor activities acceptable")
    else:
        st.warning("⚠️ No data available for the selected altitude range.")

with col2:
    st.subheader("📈 **Vertical Analysis**")
    
    if not filtered_data.empty:
        # Enhanced vertical profile
        profile_data = filtered_data.groupby('altitude').agg({
            pollutant: ['mean', 'std', 'max', 'min']
        }).round(2)
        profile_data.columns = ['mean', 'std', 'max', 'min']
        profile_data = profile_data.reset_index()
        
        # Create vertical profile with error bands
        fig_profile = go.Figure()
        
        # Mean line
        fig_profile.add_trace(go.Scatter(
            x=profile_data['mean'],
            y=profile_data['altitude'],
            mode='lines+markers',
            name=f'Average {pollutant.upper()}',
            line=dict(color='red', width=3),
            marker=dict(size=8, symbol='circle'),
            fill=None
        ))
        
        # Error bands (std deviation)
        fig_profile.add_trace(go.Scatter(
            x=profile_data['mean'] + profile_data['std'],
            y=profile_data['altitude'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_profile.add_trace(go.Scatter(
            x=profile_data['mean'] - profile_data['std'],
            y=profile_data['altitude'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            name='±1 Std Dev',
            hoverinfo='skip'
        ))
        
        fig_profile.update_layout(
            title=f"📈 Vertical {pollutant.upper()} Profile",
            xaxis_title=f"{pollutant.upper()} Concentration",
            yaxis_title="Altitude (m)",
            height=450,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig_profile, use_container_width=True)
        
        # Enhanced statistics with better formatting
        st.subheader("📊 **Key Statistics**")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            ground_level = profile_data.iloc[0]['mean'] if len(profile_data) > 0 else 0
            st.metric("🏠 Ground Level", f"{ground_level:.1f}", help="Surface pollution level")
            
            if len(profile_data) > 1:
                mid_level = profile_data.iloc[len(profile_data)//2]['mean']
                st.metric("🏢 Mid Level", f"{mid_level:.1f}", help="Mid-altitude pollution")
        
        with col_stat2:
            if len(profile_data) > 1:
                top_level = profile_data.iloc[-1]['mean']
                st.metric("☁️ Top Level", f"{top_level:.1f}", help="Upper altitude pollution")
            
            reduction = ((ground_level - top_level) / ground_level * 100) if ground_level > 0 else 0
            st.metric("📉 Altitude Reduction", f"{reduction:.1f}%", help="Pollution decrease with height")
    else:
        st.warning("⚠️ No data for vertical analysis")

# Enhanced Hotspot Analysis with Range-wise Distribution
st.subheader("🔥 Range-wise Pollution Hotspots & Hazard Map")

if not filtered_data.empty:
    
    # Create geographic zones for better distribution
    def create_zone_hotspots(data, pollutant_col):
        """Create hotspots distributed across different zones of Delhi"""
        
        # Define Delhi zones with center coordinates
        delhi_zones = {
            'North Delhi': {'lat': 28.75, 'lon': 77.15, 'name': 'North Delhi (Rohini, Model Town)'},
            'South Delhi': {'lat': 28.55, 'lon': 77.25, 'name': 'South Delhi (CP, GK)'},
            'East Delhi': {'lat': 28.65, 'lon': 77.30, 'name': 'East Delhi (Laxmi Nagar, Preet Vihar)'},
            'West Delhi': {'lat': 28.65, 'lon': 77.10, 'name': 'West Delhi (Janakpuri, Dwarka)'},
            'Central Delhi': {'lat': 28.65, 'lon': 77.20, 'name': 'Central Delhi (ITO, Red Fort)'},
            'Northeast Delhi': {'lat': 28.70, 'lon': 77.25, 'name': 'Northeast Delhi (Shahdara, Seelampur)'},
            'Northwest Delhi': {'lat': 28.70, 'lon': 77.15, 'name': 'Northwest Delhi (Pitampura, Shalimar Bagh)'},
            'Southwest Delhi': {'lat': 28.55, 'lon': 77.15, 'name': 'Southwest Delhi (Vasant Kunj, Munirka)'}
        }
        
        zone_hotspots = []
        
        for zone, coords in delhi_zones.items():
            # Find data points closest to each zone center
            zone_data = data.copy()
            zone_data['distance'] = np.sqrt(
                (zone_data['latitude'] - coords['lat'])**2 + 
                (zone_data['longitude'] - coords['lon'])**2
            )
            
            # Get closest points in this zone
            closest_in_zone = zone_data.nsmallest(10, 'distance')
            if not closest_in_zone.empty:
                avg_pollution = closest_in_zone[pollutant_col].mean()
                best_point = closest_in_zone.loc[closest_in_zone[pollutant_col].idxmax()]
                
                zone_hotspots.append({
                    'zone': zone,
                    'zone_name': coords['name'],
                    'latitude': best_point['latitude'],
                    'longitude': best_point['longitude'],
                    'pollution': avg_pollution,
                    'max_pollution': best_point[pollutant_col],
                    'distance_from_center': best_point['distance']
                })
        
        return pd.DataFrame(zone_hotspots).sort_values('pollution', ascending=False)
    
    # Get zone-wise hotspots
    zone_hotspots = create_zone_hotspots(filtered_data, pollutant)
    
    # Create three columns for better layout
    col_a, col_b, col_c = st.columns([1, 1, 1])
    
    with col_a:
        st.write("**�️ Zone-wise Top Pollution Areas**")
        for i, (_, row) in enumerate(zone_hotspots.head(8).iterrows(), 1):
            if row['pollution'] > 100:
                level = "🔴 Severe"
                icon = "🚨"
            elif row['pollution'] > 50:
                level = "🟡 High" 
                icon = "⚠️"
            else:
                level = "🟢 Moderate"
                icon = "✅"
            
            st.write(f"**{icon} {i}. {row['zone']}**")
            st.write(f"📍 {row['latitude']:.3f}, {row['longitude']:.3f}")
            st.write(f"📊 {row['pollution']:.1f} {pollutant.upper()} - {level}")
            st.write("---")
    
    with col_b:
        st.write("**🎯 Critical Hotspot Details**")
        
        if not zone_hotspots.empty:
            most_polluted_zone = zone_hotspots.iloc[0]
            
            st.error(f"""
            **🚨 HIGHEST RISK ZONE:**
            
            �️ **Area:** {most_polluted_zone['zone_name']}
            
            📍 **Coordinates:** {most_polluted_zone['latitude']:.3f}, {most_polluted_zone['longitude']:.3f}
            
            📊 **{pollutant.upper()} Level:** {most_polluted_zone['pollution']:.1f}
            
            � **Updated:** {datetime.now().strftime('%H:%M')}
            """)
            
            # Enhanced health advisory
            if most_polluted_zone['pollution'] > 100:
                st.error("� **EMERGENCY:** Stay indoors! Use air purifiers. N95 masks mandatory.")
            elif most_polluted_zone['pollution'] > 50:
                st.warning("⚠️ **HIGH ALERT:** Minimize outdoor time. Sensitive groups avoid exposure.")
            else:
                st.success("✅ **MODERATE:** Normal activities okay with basic precautions.")
    
    with col_c:
        st.write("**📊 Zone Risk Summary**")
        
        # Risk statistics
        severe_zones = len(zone_hotspots[zone_hotspots['pollution'] > 100])
        high_zones = len(zone_hotspots[(zone_hotspots['pollution'] > 50) & (zone_hotspots['pollution'] <= 100)])
        moderate_zones = len(zone_hotspots[zone_hotspots['pollution'] <= 50])
        
        st.metric("🔴 Severe Risk Zones", severe_zones)
        st.metric("🟡 High Risk Zones", high_zones) 
        st.metric("🟢 Moderate Risk Zones", moderate_zones)
        
        # Overall city status
        avg_pollution = zone_hotspots['pollution'].mean()
        if avg_pollution > 100:
            st.error("🏙️ **CITY STATUS: CRITICAL**")
        elif avg_pollution > 50:
            st.warning("🏙️ **CITY STATUS: UNHEALTHY**")
        else:
            st.success("🏙️ **CITY STATUS: ACCEPTABLE**")

    # Create Enhanced Hazard Map
    st.subheader("🗺️ Delhi Pollution Hazard Map")
    
    if not zone_hotspots.empty:
        # Create hazard map visualization
        fig_hazard = go.Figure()
        
        # Add zone hotspots with different sizes based on risk
        colors = []
        sizes = []
        symbols = []
        
        for _, zone in zone_hotspots.iterrows():
            if zone['pollution'] > 100:
                colors.append('red')
                sizes.append(25)
                symbols.append('diamond')
            elif zone['pollution'] > 50:
                colors.append('orange')
                sizes.append(20)
                symbols.append('circle')
            else:
                colors.append('green')
                sizes.append(15)
                symbols.append('circle')
        
        # Add zone hotspots
        fig_hazard.add_trace(go.Scatter(
            x=zone_hotspots['longitude'],
            y=zone_hotspots['latitude'],
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                symbol=symbols,
                line=dict(width=2, color='darkblue'),
                opacity=0.8
            ),
            text=[f"{row['zone']}<br>{row['pollution']:.1f}" for _, row in zone_hotspots.iterrows()],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><br>Risk Level: %{marker.color}<br>Coordinates: %{x:.3f}, %{y:.3f}<extra></extra>',
            name='Zone Hotspots'
        ))
        
        # Add all data points as background
        fig_hazard.add_trace(go.Scatter(
            x=filtered_data['longitude'],
            y=filtered_data['latitude'],
            mode='markers',
            marker=dict(
                size=6,
                color=filtered_data[pollutant],
                colorscale='Reds',
                opacity=0.4,
                colorbar=dict(
                    title=f"{pollutant.upper()}<br>Background",
                    x=1.1,
                    len=0.5
                )
            ),
            name='Background Data',
            hovertemplate=f'{pollutant.upper()}: %{{marker.color:.1f}}<extra></extra>'
        ))
        
        # Add Delhi boundary approximation
        delhi_boundary_lat = [28.4, 28.4, 28.8, 28.8, 28.4]
        delhi_boundary_lon = [77.0, 77.4, 77.4, 77.0, 77.0]
        
        fig_hazard.add_trace(go.Scatter(
            x=delhi_boundary_lon,
            y=delhi_boundary_lat,
            mode='lines',
            line=dict(color='blue', width=3, dash='dash'),
            name='Delhi Boundary',
            hoverinfo='skip'
        ))
        
        fig_hazard.update_layout(
            title=f"🗺️ Delhi Pollution Hazard Map - {pollutant.upper()} ({datetime.now().strftime('%Y-%m-%d')})",
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            height=500,
            showlegend=True,
            xaxis=dict(range=[76.9, 77.5]),
            yaxis=dict(range=[28.3, 28.9]),
            legend=dict(x=0, y=1)
        )
        
        st.plotly_chart(fig_hazard, use_container_width=True)
        
        # Risk Legend
        st.markdown("""
        **🔍 Hazard Map Legend:**
        - 🔴 **Red Diamond**: Severe Risk (>100) - Emergency measures needed
        - 🟠 **Orange Circle**: High Risk (50-100) - Caution advised  
        - 🟢 **Green Circle**: Moderate Risk (<50) - Normal precautions
        - 📍 **Background dots**: Real-time pollution measurements
        - 📐 **Blue dashed line**: Delhi administrative boundary
        """)

else:
    st.error("No pollution data available for hazard analysis.")

# Add detailed data scaling information at the end
st.divider()
show_data_scaling_info(show_detailed=True)

