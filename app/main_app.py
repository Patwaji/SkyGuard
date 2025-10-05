import streamlit as st
import pandas as pd
import pickle
import os

# Set Streamlit Page Config
st.set_page_config(
    page_title="SkyGuard: Earthdata to Action",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# --- GLOBAL APP CONFIGURATION ---
@st.cache_resource
def load_models():
    """Load the trained Prophet and Policy Regression models."""
    try:
        # Load Prophet Model
        with open('models/prophet_predictor_model.pkl', 'rb') as f:
            prophet_model = pickle.load(f)
            
        # Load Policy Model Coefficient
        with open('models/policy_regression_model.pkl', 'rb') as f:
            policy_data = pickle.load(f)
            policy_coefficient = policy_data['coefficient']

        return prophet_model, policy_coefficient
    except FileNotFoundError:
        st.error("Model files not found! Ensure 'prophet_predictor_model.pkl' and 'policy_regression_model.pkl' are in the 'models/' directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models once when the app starts
prophet_model, policy_coefficient = load_models()

# CRITICAL FIX: Store models in session state for access by pages/modules
# This resolves the KeyError in the pages (like 1_Forecast.py)
if prophet_model is not None and policy_coefficient is not None:
    st.session_state['prophet_model'] = prophet_model
    st.session_state['policy_coefficient'] = policy_coefficient
    # Optional: If models load correctly, clear the error messages
    st.session_state['models_loaded'] = True
else:
    st.session_state['models_loaded'] = False


# --- MAIN PAGE CONTENT ---
st.title("🚀 SkyGuard: Predicting, Monitoring, and Shaping Cleaner Skies")
st.markdown("---")

st.header("Project Overview")
st.markdown("""
Our solution integrates **NASA Earth Observation (EO) data** and machine learning to provide comprehensive air quality intelligence for citizens and policymakers. Built for a specific region (Delhi), SkyGuard provides three core services:
1.  **Prediction:** Accurate 48-hour air quality forecasts.
2.  **Monitoring:** Real-time pollution maps leveraging satellite and weather data.
3.  **Policy Insight:** Quantifiable analysis of pollution trend sensitivity for strategic planning.
""")

st.subheader("Data Foundation")

coefficient_text = "N/A (Loading Error)" 
# Use the robust check before formatting (Fix 3 was partially applied here)
if st.session_state.get('models_loaded') and policy_coefficient is not None:
    coefficient_text = f"{policy_coefficient:.4f}"

st.markdown(f"""
* **Satellite EO (Features):** MODIS AOD & Sentinel-5P NO₂ (via GEE).
* **Ground Truth (Target 'y'):** AOD-derived PM2.5 proxy.
* **Predictive Model:** **Prophet** (Time Series) + **Linear Regression** (Feature-to-PM2.5 bridge).
* **Policy Coefficient (NO₂ $\\rightarrow$ PM2.5 Sensitivity):** **{coefficient_text}**
""")

st.markdown("---")
st.info("Use the sidebar navigation to explore the Forecast Dashboard and the Policy Advisor.")
