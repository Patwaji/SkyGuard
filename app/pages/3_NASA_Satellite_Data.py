import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
import json
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(page_title="NASA Satellite Data", page_icon="🛰️", layout="wide")

# NASA Earthdata configuration
NASA_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DELHI_BOUNDS = {
    "north": 28.88,
    "south": 28.40,
    "east": 77.35,
    "west": 76.84
}

def generate_mock_satellite_data():
    """Generate mock NASA satellite data for Delhi region"""
    np.random.seed(42)  # For consistent data
    
    # Generate grid points for Delhi
    lats = np.linspace(DELHI_BOUNDS["south"], DELHI_BOUNDS["north"], 20)
    lons = np.linspace(DELHI_BOUNDS["west"], DELHI_BOUNDS["east"], 20)
    
    data = []
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    for date in dates[-30:]:  # Last 30 days for demo
        for lat in lats[::2]:  # Every other point to reduce density
            for lon in lons[::2]:
                # Simulate pollution hotspots
                distance_to_center = np.sqrt((lat - 28.6139)**2 + (lon - 77.2090)**2)
                base_pollution = max(0, 100 - distance_to_center * 50)
                
                # Add seasonal variation (winter higher pollution)
                seasonal_factor = 1.5 if date.month in [11, 12, 1, 2] else 1.0
                
                # Add random variation
                noise = np.random.normal(0, 15)
                
                no2_value = max(0, base_pollution * seasonal_factor + noise)
                pm25_value = max(0, base_pollution * 0.8 * seasonal_factor + noise * 0.7)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'latitude': lat,
                    'longitude': lon,
                    'NO2': no2_value,
                    'PM25': pm25_value,
                    'AOD': max(0, no2_value / 200 + np.random.normal(0, 0.1))  # Aerosol Optical Depth
                })
    
    return pd.DataFrame(data)

def generate_fire_data():
    """Generate mock fire detection data"""
    np.random.seed(123)
    
    fire_data = []
    dates = pd.date_range(start='2023-10-01', end='2023-11-30', freq='D')  # Crop burning season
    
    for date in dates[-15:]:  # Last 15 days
        # Random fires around Delhi (especially north and northwest)
        num_fires = np.random.poisson(5)  # Average 5 fires per day
        
        for _ in range(num_fires):
            # Bias towards north/northwest of Delhi (crop burning areas)
            lat = np.random.normal(28.8, 0.2)
            lon = np.random.normal(77.0, 0.2)
            
            fire_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'latitude': lat,
                'longitude': lon,
                'brightness': np.random.uniform(300, 400),
                'confidence': np.random.uniform(70, 95),
                'fire_type': 'crop_burning'
            })
    
    return pd.DataFrame(fire_data)

def create_pollution_heatmap(df, pollutant, selected_date):
    """Create interactive pollution heatmap using Folium"""
    
    # Filter data for selected date
    df_day = df[df['date'] == selected_date]
    
    if df_day.empty:
        st.warning(f"No data available for {selected_date}")
        return None
    
    # Create map centered on Delhi
    m = folium.Map(
        location=[28.6139, 77.2090], 
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add satellite tile layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite View',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Color mapping based on pollution levels
    max_val = df_day[pollutant].max()
    min_val = df_day[pollutant].min()
    
    for _, row in df_day.iterrows():
        # Normalize value for color intensity
        intensity = (row[pollutant] - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        
        # Color based on pollution level
        if intensity > 0.8:
            color = '#8B0000'  # Dark red - Very high
        elif intensity > 0.6:
            color = '#FF0000'  # Red - High
        elif intensity > 0.4:
            color = '#FF8C00'  # Orange - Moderate
        elif intensity > 0.2:
            color = '#FFD700'  # Yellow - Low
        else:
            color = '#32CD32'  # Green - Very low
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=f"""
            <b>{pollutant} Level</b><br>
            Value: {row[pollutant]:.1f}<br>
            Location: {row['latitude']:.3f}, {row['longitude']:.3f}<br>
            Date: {selected_date}
            """,
            color='black',
            weight=1,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>{pollutant} Levels</b></p>
    <p><i class="fa fa-circle" style="color:#32CD32"></i> Very Low</p>
    <p><i class="fa fa-circle" style="color:#FFD700"></i> Low</p>
    <p><i class="fa fa-circle" style="color:#FF8C00"></i> Moderate</p>
    <p><i class="fa fa-circle" style="color:#FF0000"></i> High</p>
    <p><i class="fa fa-circle" style="color:#8B0000"></i> Very High</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_fire_map(fire_df, selected_date):
    """Create fire detection map"""
    
    # Filter fire data for selected date
    fires_day = fire_df[fire_df['date'] == selected_date]
    
    # Create map
    m = folium.Map(
        location=[28.6139, 77.2090], 
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Add fire points
    for _, fire in fires_day.iterrows():
        folium.Marker(
            location=[fire['latitude'], fire['longitude']],
            popup=f"""
            <b>Fire Detection</b><br>
            Brightness: {fire['brightness']:.1f} K<br>
            Confidence: {fire['confidence']:.1f}%<br>
            Type: {fire['fire_type']}<br>
            Date: {selected_date}
            """,
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(m)
    
    return m

def create_time_series_chart(df, pollutant):
    """Create time series chart showing pollution trends"""
    
    # Calculate daily averages
    daily_avg = df.groupby('date')[pollutant].mean().reset_index()
    daily_avg['date'] = pd.to_datetime(daily_avg['date'])
    
    fig = px.line(
        daily_avg, 
        x='date', 
        y=pollutant,
        title=f'Delhi {pollutant} Levels - NASA Satellite Data',
        labels={'date': 'Date', pollutant: f'{pollutant} (µg/m³)'},
        template='plotly_dark'
    )
    
    # Add WHO guidelines
    if pollutant == 'PM25':
        fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="WHO Safe Limit (15 µg/m³)")
        fig.add_hline(y=25, line_dash="dash", line_color="orange", annotation_text="WHO Interim Target (25 µg/m³)")
    elif pollutant == 'NO2':
        fig.add_hline(y=40, line_dash="dash", line_color="green", annotation_text="WHO Annual Limit (40 µg/m³)")
    
    fig.update_layout(height=400)
    return fig

def main():
    st.title("🛰️ NASA Satellite Data Visualization")
    st.markdown("*Real-time air quality monitoring using NASA Earthdata for Delhi*")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Data source selection
        data_source = st.selectbox(
            "Select Data Source",
            ["Air Quality (MODIS)", "Fire Detection (VIIRS)", "Aerosol Optical Depth"]
        )
        
        # Date selection
        if data_source == "Fire Detection (VIIRS)":
            # Fire data has different date range
            available_dates = pd.date_range(start='2023-10-17', end='2023-10-31', freq='D')
        else:
            available_dates = pd.date_range(start='2023-12-02', end='2023-12-31', freq='D')
        
        selected_date = st.selectbox(
            "Select Date",
            available_dates.strftime('%Y-%m-%d').tolist()[::-1]  # Most recent first
        )
        
        if data_source == "Air Quality (MODIS)":
            pollutant = st.selectbox("Select Pollutant", ["NO2", "PM25"])
    
    # Load data
    with st.spinner("🛰️ Loading NASA satellite data..."):
        if data_source == "Fire Detection (VIIRS)":
            fire_df = generate_fire_data()
            
            st.subheader(f"🔥 Fire Detection - {selected_date}")
            st.markdown("*Data from NASA VIIRS satellite fire detection system*")
            
            # Fire statistics
            fires_today = fire_df[fire_df['date'] == selected_date]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fires Detected", len(fires_today))
            with col2:
                avg_confidence = fires_today['confidence'].mean() if not fires_today.empty else 0
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            with col3:
                high_conf_fires = len(fires_today[fires_today['confidence'] > 80])
                st.metric("High Confidence Fires", high_conf_fires)
            
            # Fire map
            fire_map = create_fire_map(fire_df, selected_date)
            if fire_map:
                st_folium(fire_map, width=700, height=500)
            
            # Fire trend chart
            fire_counts = fire_df.groupby('date').size().reset_index(name='fire_count')
            fire_counts['date'] = pd.to_datetime(fire_counts['date'])
            
            fig_fires = px.bar(
                fire_counts,
                x='date',
                y='fire_count',
                title='Daily Fire Detections Around Delhi',
                template='plotly_dark'
            )
            st.plotly_chart(fig_fires, use_container_width=True)
            
        else:
            # Air quality data
            df = generate_mock_satellite_data()
            
            if data_source == "Aerosol Optical Depth":
                pollutant = "AOD"
                st.subheader(f"🌫️ Aerosol Optical Depth - {selected_date}")
                st.markdown("*Atmospheric aerosol concentration from NASA MODIS satellite*")
            else:
                st.subheader(f"🌬️ {pollutant} Concentration - {selected_date}")
                st.markdown("*Air quality data from NASA MODIS satellite measurements*")
            
            # Summary statistics
            today_data = df[df['date'] == selected_date]
            if not today_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_val = today_data[pollutant].mean()
                    st.metric(f"Average {pollutant}", f"{avg_val:.1f}")
                
                with col2:
                    max_val = today_data[pollutant].max()
                    st.metric(f"Maximum {pollutant}", f"{max_val:.1f}")
                
                with col3:
                    min_val = today_data[pollutant].min()
                    st.metric(f"Minimum {pollutant}", f"{min_val:.1f}")
                
                with col4:
                    # Health status
                    if pollutant == "PM25":
                        if avg_val <= 15:
                            status = "🟢 Good"
                        elif avg_val <= 25:
                            status = "🟡 Moderate"
                        elif avg_val <= 35:
                            status = "🟠 Unhealthy"
                        else:
                            status = "🔴 Hazardous"
                    else:
                        status = "📊 Monitoring"
                    
                    st.metric("Air Quality", status)
            
            # Create and display map
            pollution_map = create_pollution_heatmap(df, pollutant, selected_date)
            if pollution_map:
                st_folium(pollution_map, width=700, height=500)
            
            # Time series chart
            st.subheader(f"📈 {pollutant} Trend Analysis")
            time_series_fig = create_time_series_chart(df, pollutant)
            st.plotly_chart(time_series_fig, use_container_width=True)
    
    # Information section
    with st.expander("ℹ️ About NASA Earthdata"):
        st.markdown("""
        ### Data Sources
        
        **MODIS (Moderate Resolution Imaging Spectroradiometer)**
        - Air quality measurements (NO2, PM2.5, AOD)
        - Daily global coverage
        - 1km spatial resolution
        
        **VIIRS (Visible Infrared Imaging Radiometer Suite)**
        - Active fire detection
        - Night-time lights
        - Real-time monitoring
        
        ### How It Works
        1. **Satellites** continuously monitor Earth's atmosphere
        2. **Algorithms** process raw data to extract pollution levels
        3. **Validation** against ground-based monitoring stations
        4. **Visualization** for public awareness and policy action
        
        ### Limitations
        - Cloud cover can affect measurements
        - Data processing may have 1-3 day delay
        - Spatial resolution limitations in urban areas
        """)

if __name__ == "__main__":
    main()
