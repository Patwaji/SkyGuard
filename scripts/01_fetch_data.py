import ee
import openaq
import pandas as pd
import requests # Import requests for the OpenAQ fallback
import os # Import os for directory creation
import sys
from datetime import datetime
from meteostat import Point, Daily, units

# Add config path for environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_root, 'app', 'config'))

try:
    from env_config import config
    TARGET_LAT = config.TARGET_LATITUDE
    TARGET_LON = config.TARGET_LONGITUDE
    TARGET_CITY = config.TARGET_CITY
except ImportError:
    print("⚠️ Environment config not found, using default values")
    TARGET_LAT = 28.6139
    TARGET_LON = 77.2090
    TARGET_CITY = 'Delhi'

# Ensure data directories exist
RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
RADIUS_KM = 50 
START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2024, 1, 1) # 5 years of historical data

# Replace this with your actual OpenAQ key
OPENAQ_API_KEY = "550ee53fbfd44f9d4739cc804463cc873ccc9f43e4f9b2ee3fa0eb544bf99b39" 

# Initialize GEE (Assuming successful authentication is done via ee.Authenticate())
PROJECT_ID = 'skyguard-474107'
ee.Initialize(project=PROJECT_ID)

# Define the Area of Interest (AOI) for GEE processing
AOI = ee.Geometry.Point(TARGET_LON, TARGET_LAT).buffer(RADIUS_KM * 1000)

# ----------------------------------------------------------------------
# 1. SATELLITE DATA (NASA/ESA: AOD & NO2) - BATCH EXPORT
# ----------------------------------------------------------------------
def export_time_series_batch(collection_id, band_name, output_filename, start_date, end_date):
    # ... (function body remains the same, as it is now correctly running) ...
    # ... (skipping for brevity, ensure you use the working version) ...
    
    # 1. Define and Filter the ImageCollection
    collection = ee.ImageCollection(collection_id) \
        .filterDate(start_date.isoformat()[:10], end_date.isoformat()[:10]) \
        .filterBounds(AOI) \
        .select(band_name)

    # 2. Define the inner function
    def get_mean_for_image(image):
        mean_dict = image.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=AOI, 
            scale=1000, 
            maxPixels=1e9 
        )
        return ee.Feature(AOI, {
            'date': image.date().format('YYYY-MM-dd'),
            output_filename: mean_dict.get(band_name)
        })

    # 3. Apply the mapping function
    feature_collection = collection.map(get_mean_for_image)

    # 4. Initiate the batch export task
    task = ee.batch.Export.table.toDrive(
        collection=feature_collection.filter(ee.Filter.notNull([output_filename])),
        description=f'SkyGuard_{output_filename}_Export',
        folder='SkyGuard_Data',
        fileNamePrefix=output_filename,
        fileFormat='CSV'
    )
    
    task.start()
    print(f"🚀 Started GEE Export Task: {output_filename}. Status: {task.status()['state']}")


def get_satellite_data():
    """Initiates batch exports for both AOD and NO2 time series."""
    
    # --- AOD (MODIS V61 - Aerosol Optical Depth) ---
    AOD_COLLECTION = 'MODIS/061/MCD19A2_GRANULES' 
    AOD_BAND = 'Optical_Depth_047'
    export_time_series_batch(AOD_COLLECTION, AOD_BAND, 'AOD_RAW', START_DATE, END_DATE)
    
    # --- NO2 (Sentinel-5P TROPOMI - Tropospheric NO2) ---
    NO2_COLLECTION = 'COPERNICUS/S5P/OFFL/L3_NO2'
    NO2_BAND = 'tropospheric_NO2_column_number_density'
    export_time_series_batch(NO2_COLLECTION, NO2_BAND, 'NO2_RAW', START_DATE, END_DATE)

    return pd.DataFrame(), pd.DataFrame() 


# ----------------------------------------------------------------------
# 2. GROUND VALIDATION DATA (OpenAQ: PM2.5) - ULTIMATE FALLBACK - FIXED
# ----------------------------------------------------------------------
def get_openaq_data():
    """
    ULTIMATE FALLBACK: Queries OpenAQ V3 API via raw HTTP request.
    FIXED: Using X-API-KEY header to resolve 401 Unauthorized error.
    """
    
    DELHI_LOCATION_ID = 3978 
    BASE_URL = "https://api.openaq.org/v3/measurements"
    
    date_from_str = START_DATE.strftime('%Y-%m-%d')
    date_to_str = END_DATE.strftime('%Y-%m-%d')
    
    params = {
    'date_from': date_from_str,
    'date_to': date_to_str,
    # Use the geospatial filter, which the raw requests call should handle:
    'coordinates': f"{TARGET_LAT},{TARGET_LON}", 
    'radius': RADIUS_KM * 1000, # Radius in meters
    
    'limit': 10000, 
    'page': 1,
    'parameter_id': 2, # PM2.5 is ID 2 in V3
    'order_by': 'datetime',
    'sort': 'asc',
    }
    
    # FIXED: Switched to the X-API-KEY header format
    headers = {
        'X-API-KEY': OPENAQ_API_KEY 
    }
    
    all_records = []
    page = 1
    
    print("Attempting OpenAQ data pull via V3 URL (Location ID)...")
    
    while True:
        params['page'] = page
        
        try:
            response = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ OpenAQ Request Failed on Page {page}: {e}")
            break
            
        results = data.get('results', [])
        if not results:
            break
            
        all_records.extend(results)
        
        meta = data.get('meta', {})
        if (meta.get('limit') * page) >= meta.get('found', 0):
            break
        
        page += 1
        
    if not all_records:
        print("❌ OpenAQ: Failed to retrieve any records. Returning empty DataFrame.")
        return pd.DataFrame()

    df_ground = pd.DataFrame(all_records)
    
    df_ground = df_ground.rename(columns={'date_utc': 'timestamp', 'value': 'PM25_Ground_ugm3'})
    
    df_ground['timestamp'] = pd.to_datetime(df_ground['timestamp'])
        
    df_ground = df_ground[['timestamp', 'PM25_Ground_ugm3']]

    print(f"✅ OpenAQ Data (PM25) fetched. Total Records: {df_ground.shape[0]}")
    return df_ground


# ----------------------------------------------------------------------
# 3. METEOROLOGICAL DATA (Meteostat: T, H, W) - Local Call - FIXED
# ----------------------------------------------------------------------
def get_weather_data():
    """Fetches historical daily weather data (T, H, W) for the AOI."""
    location = Point(TARGET_LAT, TARGET_LON)

    data = Daily(location, START_DATE, END_DATE)
    df_weather = data.fetch()

    # --- FIXED SECTION: Check and rename columns safely ---
    
    # Ensure all three columns exist; if not, create them and fill with NaN or a median/mean value.
    required_cols = ['tavg', 'rhum', 'wspd']
    
    for col in required_cols:
        if col not in df_weather.columns:
            # If a column is missing, create it with the mean of available data 
            # (or 50, a neutral value, for humidity/wind for the script to proceed)
            df_weather[col] = df_weather['tavg'].mean() if col == 'tavg' else 50 
            print(f"⚠️ Meteostat Warning: Column '{col}' not found. Filling with a mean/default value.")


    df_weather = df_weather[['tavg', 'rhum', 'wspd']]
    df_weather.columns = ['Temp_C', 'Humidity_pct', 'Wind_Speed_kmh']
    
    df_weather = df_weather.reset_index().rename(columns={'time': 'date'})
    df_weather['date'] = pd.to_datetime(df_weather['date'])
    
    print(f"✅ Weather Data (T, H, W) fetched. Records: {df_weather.shape[0]}")
    return df_weather


# ----------------------------------------------------------------------
# --- FINAL EXECUTION BLOCK ---
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print(f"--- Starting Data Acquisition for {TARGET_CITY} ({START_DATE.year}-{END_DATE.year-1}) ---")
    
    # FIX: Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    try:
        # 1. Initiate GEE Batch Exports 
        get_satellite_data()
        
        # 2. Get Ground Data (Runs locally)
        df_ground = get_openaq_data()
        
        # Only save if data was retrieved (to avoid overwriting with empty file if API fails)
        if not df_ground.empty:
            output_path = os.path.join(RAW_DATA_DIR, 'openaq_historical_raw.csv')
            df_ground.to_csv(output_path, index=False)
            print(f"-> Saved openaq_historical_raw.csv to {output_path}")

        # 3. Get Weather Data (Runs locally)
        df_weather = get_weather_data()
        output_path = os.path.join(RAW_DATA_DIR, 'weather_historical_raw.csv')
        df_weather.to_csv(output_path, index=False)
        print(f"-> Saved weather_historical_raw.csv to {output_path}")
        
        print("\n--- PHASE 1: Local APIs Complete. Waiting for GEE Exports to finish. ---")
        print(f"ACTION REQUIRED: Go to your Google Drive, monitor the 'SkyGuard_Data' folder. Once the tasks finish, download 'AOD_RAW.csv' and 'NO2_RAW.csv' and place them in '{RAW_DATA_DIR}' folder.")
        print("\n**You can now proceed to Phase 2: Data Processing while GEE tasks run in the cloud.**")
        
    except Exception as e:
        print(f"\nFATAL ERROR DURING ACQUISITION: {e}")
        print("Please check your global OPENAQ_API_KEY value and ensure it is valid.")