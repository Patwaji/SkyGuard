"""
Reusable data scaling information component for all pages that use ML model data
"""
import streamlit as st
import pandas as pd
import numpy as np

def show_data_scaling_info(show_detailed=True):
    """
    Display data scaling information and health context for ML model data
    
    Args:
        show_detailed (bool): Whether to show detailed technical information
    """
    
    # Check if we can access the CSV data for scaling reference
    csv_stats = None
    try:
        df = pd.read_csv('data/processed/historical_merged_master.csv')
        if 'y' in df.columns:
            csv_stats = {
                'min': df['y'].min(),
                'max': df['y'].max(),
                'mean': df['y'].mean(),
                'std': df['y'].std()
            }
    except:
        pass
    
    with st.expander("📊 **Understanding Our Air Quality Data Scale**", expanded=False):
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.subheader("🔍 **Data Scale Information**")
            if csv_stats:
                st.metric("Model Training Range", f"{csv_stats['min']:.0f} - {csv_stats['max']:.0f}")
                st.metric("Model Training Average", f"{csv_stats['mean']:.0f}")
                st.caption("📈 These values represent AOD-derived proxy measurements used for training")
            else:
                st.info("📋 Model training data statistics not available")
            
            st.markdown("""
            **⚠️ Important Note:**
            - Our ML models were trained on **AOD (Aerosol Optical Depth)** proxy values
            - These range from 0 to ~400,000 (much larger than real PM2.5)
            - Real PM2.5 measurements range from 0-500 µg/m³
            """)
        
        with info_col2:
            st.subheader("🌍 **Real-World Context**")
            st.markdown("""
            **WHO PM2.5 Standards (µg/m³):**
            - 🟢 **Good (0-12)**: Safe for all
            - 🟡 **Moderate (12-35)**: Sensitive groups caution
            - 🟠 **Unhealthy for Sensitive (35-55)**: Limit exposure
            - 🔴 **Unhealthy (55-150)**: Everyone limit activities
            - 🟣 **Very Unhealthy (150-250)**: Avoid outdoor activities
            - ⚫ **Hazardous (250+)**: Emergency conditions
            """)
        
        if show_detailed:
            st.divider()
            st.subheader("🔬 **Technical Details**")
            
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.markdown("""
                **🔄 Scale Conversion:**
                - **Model predictions** ÷ 1000 = **Display values**
                - This converts AOD proxy to approximate PM2.5 range
                - Provides realistic visualization context
                """)
            
            
            
            st.info("""
            💡 **Why This Scaling?** 
            AOD measures the total light extinction by aerosols in the atmosphere column, 
            while PM2.5 measures ground-level particulate matter concentration. 
            The relationship is complex and varies by region, season, and atmospheric conditions.
            """)

def show_quick_scaling_alert():
    """Show a quick alert about data scaling without detailed info"""
    st.warning("""
    ⚠️ **Data Scale Note**: This page uses ML model predictions trained on AOD proxy values 
    (range: 0-400k). For display, values are scaled to approximate real PM2.5 levels (0-500 µg/m³). 
    See the data scaling section below for details.
    """)

def get_scaled_display_value(model_value):
    """
    Convert model prediction to display-friendly PM2.5 value
    
    Args:
        model_value: Raw model prediction (AOD proxy scale)
    
    Returns:
        Scaled value for display (approximate PM2.5 µg/m³)
    """
    if model_value is None or pd.isna(model_value):
        return None
    
    # Scale down by 1000 and clamp to reasonable PM2.5 range
    scaled = model_value / 1000
    return max(0, min(scaled, 500))  # Clamp between 0-500 µg/m³

def get_health_category(pm25_value):
    """
    Get health category and color for PM2.5 value
    
    Args:
        pm25_value: PM2.5 concentration in µg/m³
    
    Returns:
        tuple: (category_text, color, emoji)
    """
    if pm25_value is None or pd.isna(pm25_value):
        return ("Unknown", "gray", "❓")
    
    if pm25_value <= 12:
        return ("Good", "green", "🟢")
    elif pm25_value <= 35:
        return ("Moderate", "yellow", "🟡")
    elif pm25_value <= 55:
        return ("Unhealthy for Sensitive", "orange", "🟠")
    elif pm25_value <= 150:
        return ("Unhealthy", "red", "🔴")
    elif pm25_value <= 250:
        return ("Very Unhealthy", "purple", "🟣")
    else:
        return ("Hazardous", "black", "⚫")
