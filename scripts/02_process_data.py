import pandas as pd
import numpy as np
import os
import sys

# Add config path for environment variables and get project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_root, 'app', 'config'))

# Set up data directories
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- CONFIGURATION ---
# NOTE: The NO2 data from TROPOMI is mol/m^2, we must apply the normalization factor (1e9) 
# to scale it up for regression, as intended.
NO2_SCALE_FACTOR = 1e9 

def process_and_merge_data():
    """
    Loads raw data, performs cleaning/resampling, and merges into the final ML training file.
    Handles loading of empty/missing files gracefully.
    """
    
    print("--- Starting Phase 2: Data Processing & Alignment ---")
    
    # 1. LOAD RAW DATA
    try:
        # Check if the OpenAQ file is empty before loading it
        openaq_path = os.path.join(RAW_DATA_DIR, 'openaq_historical_raw.csv')
             df_ground = pd.read_csv(openaq_path)
        else:
             print("🚨 OpenAQ File is EMPTY/MISSING. Initializing empty DataFrame to trigger placeholder.")
             df_ground = pd.DataFrame() # Initialize as empty to trigger placeholder logic
             
        # Load GEE and Weather files (These must be present and valid)
        df_aod = pd.read_csv(os.path.join(RAW_DATA_DIR, 'AOD_RAW.csv'))
        df_no2 = pd.read_csv(os.path.join(RAW_DATA_DIR, 'NO2_RAW.csv'))
        df_weather = pd.read_csv(os.path.join(RAW_DATA_DIR, 'weather_historical_raw.csv'))
        
    except Exception as e:
        print(f"FATAL: Missing raw file. Ensure GEE files are downloaded and placed in '{RAW_DATA_DIR}'. Error: {e}") os
import sys

# Add config path for environment variables and get project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_root, 'app', 'config'))

# Set up data directories
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- CONFIGURATION ---
# NOTE: The NO2 data from TROPOMI is mol/m^2, we must apply the normalization factor (1e9) 
# to scale it up for regression, as intended.
NO2_SCALE_FACTOR = 1e9 

def process_and_merge_data():
    """
    Loads raw data, performs cleaning/resampling, and merges into the final ML training file.
    Handles loading of empty/missing files gracefully.
    """
    
    print("--- Starting Phase 2: Data Processing & Alignment ---")
    
    # 1. LOAD RAW DATA
    try:
        # Check if the OpenAQ file is empty before loading it
        openaq_path = 'data/raw/openaq_historical_raw.csv'
        
        if os.path.exists(openaq_path) and os.path.getsize(openaq_path) > 10: # Check size > 10 bytes
             df_ground = pd.read_csv(openaq_path)
        else:
             print("🚨 OpenAQ File is EMPTY/MISSING. Initializing empty DataFrame to trigger placeholder.")
             df_ground = pd.DataFrame() # Initialize as empty to trigger placeholder logic
             
        # Load GEE and Weather files (These must be present and valid)
        df_aod = pd.read_csv('data/raw/AOD_RAW.csv')
        df_no2 = pd.read_csv('data/raw/NO2_RAW.csv')
        df_weather = pd.read_csv('data/raw/weather_historical_raw.csv')
        
    except FileNotFoundError as e:
        print(f"FATAL: Missing raw file. Ensure GEE files are downloaded and placed in 'data/raw/'. Error: {e}")
        return pd.DataFrame()

    # 2. CLEAN AND NORMALIZE SATELLITE DATA
    df_aod = df_aod.rename(columns={'AOD_RAW': 'AOD'})
    df_aod['date'] = pd.to_datetime(df_aod['date'])

    df_no2 = df_no2.rename(columns={'NO2_RAW': 'NO2'})
    df_no2['date'] = pd.to_datetime(df_no2['date'])
    df_no2['NO2'] = df_no2['NO2'] * NO2_SCALE_FACTOR # Apply normalization
    
    # Merge Satellite Data (AOD + NO2)
    df_sat = pd.merge(df_aod, df_no2, on='date', how='outer')

    # 3. PREPARE WEATHER DATA (FIXED for Robust Column Handling)
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    # Rename/Handle all required columns to ensure consistency before merge
    
    # Check for Wind Speed (wspd) and rename/convert it
    if 'wspd' in df_weather.columns:
        df_weather['Wind_Speed_ms'] = df_weather['wspd'] * 0.277778 
        df_weather = df_weather.drop(columns=['wspd'])
    else:
        df_weather['Wind_Speed_ms'] = 5.0 # Fallback: Create dummy column

    # Check for Relative Humidity (rhum) and rename it
    if 'rhum' in df_weather.columns:
        df_weather = df_weather.rename(columns={'rhum': 'Humidity_pct'})
    else:
        df_weather['Humidity_pct'] = 50.0 # Fallback: Create dummy column

    # Check for Average Temperature (tavg) and rename it
    if 'tavg' in df_weather.columns:
        df_weather = df_weather.rename(columns={'tavg': 'Temp_C'})
    else:
        # Fallback: Create dummy temp (if necessary, but Meteostat is reliable for this)
        df_weather['Temp_C'] = 25.0

    # Select only the cleaned and renamed feature columns
    df_weather = df_weather[['date', 'Temp_C', 'Humidity_pct', 'Wind_Speed_ms']]


    # 4. MERGE SATELLITE AND WEATHER
    df_merged = pd.merge(df_sat, df_weather, on='date', how='inner')
    
    # 5. CREATE TARGET VARIABLE ('y') - CRITICAL HACKATHON SOLUTION
    
    # We must have a 'y' (PM2.5) column for Prophet.
    if df_ground.empty or df_ground.shape[0] < 100:
        print("\n🚨 CRITICAL: Ground data missing. Creating synthetic PM2.5 column (y) from AOD.")
        
        # HACK: Create PM2.5 proxy: PM2.5 is roughly AOD * 100 (a common simple scaling factor)
        df_merged['y'] = df_merged['AOD'] * 100 
        
        # Mark proxy data for transparency in presentation
        df_merged['data_source'] = 'Proxy (AOD-derived)' 
    
    else:
        print("\n✅ OpenAQ data found. Proceeding with ground truth PM2.5.")
        
        # Resample OpenAQ data to daily median
        df_ground['timestamp'] = pd.to_datetime(df_ground['timestamp'])
        df_ground_daily = df_ground.set_index('timestamp').resample('D')['PM25_Ground_ugm3'].median().reset_index()
        df_ground_daily = df_ground_daily.rename(columns={'timestamp': 'date'})
        
        # Final merge with actual ground truth
        df_merged = pd.merge(df_merged, df_ground_daily, on='date', how='inner')
        df_merged['y'] = df_merged['PM25_Ground_ugm3']
        df_merged['data_source'] = 'Ground-Validated'
        df_merged = df_merged.drop(columns=['PM25_Ground_ugm3'])


    # 6. FINAL CLEANUP AND IMPUTATION (Essential for time series models)
    df_merged = df_merged.sort_values('date').set_index('date')
    
    # Imputation: Fill gaps using linear interpolation (essential for Prophet)
    df_merged = df_merged.interpolate(method='linear').reset_index()

    # Rename date column for Prophet's required format ('ds' = datetime stamp)
    df_merged = df_merged.rename(columns={'date': 'ds'}) 
    
    # Save the final file
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, 'historical_merged_master.csv')
    df_merged.to_csv(output_path, index=False)
    
    print(f"\n✅ PHASE 2 COMPLETE: Final merged file created at '{output_path}'")
    print(f"Final Data shape for ML Training: {df_merged.shape}")
    
    return df_merged

if __name__ == '__main__':
    process_and_merge_data()