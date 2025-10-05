import ee
import pandas as pd
import requests
import os
import sys
from datetime import datetime
from meteostat import Point, Daily, units

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(os.path.join(project_root, 'app', 'config'))

try:
    from env_config import config, get_openaq_api_key
    TARGET_LAT = config.TARGET_LATITUDE
    TARGET_LON = config.TARGET_LONGITUDE
    TARGET_CITY = config.TARGET_CITY
    OPENAQ_API_KEY = get_openaq_api_key()
except ImportError:
    print("⚠️ Environment config not found, using default values")
    TARGET_LAT = 28.6139
    TARGET_LON = 77.2090
    TARGET_CITY = 'Delhi'
    OPENAQ_API_KEY = None

RAW_DATA_DIR = os.path.join(project_root, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(project_root, 'data', 'processed')
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
RADIUS_KM = 50 
START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2024, 1, 1) # 5 years of historical data

PROJECT_ID = 'skyguard-474107'
ee.Initialize(project=PROJECT_ID)

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
    Retrieves OpenAQ data for Delhi PM2.5 measurements using the working v3 API pattern.
    Uses the official sensors endpoint: /v3/sensors/{sensor_id}/measurements
    """
    
    # Check if API key is available
    if not OPENAQ_API_KEY:
        print("⚠️ OpenAQ API Key not found. Skipping ground truth data.")
        return pd.DataFrame()
    
    headers = {'X-API-Key': OPENAQ_API_KEY}
    
    print("🔍 Retrieving OpenAQ PM2.5 measurements using official v3 API...")
    
    try:
        # Step 1: Get PM2.5 monitoring locations in Delhi area
        locations_url = "https://api.openaq.org/v3/locations"
        location_params = {
            'bbox': f"{TARGET_LON-0.5},{TARGET_LAT-0.5},{TARGET_LON+0.5},{TARGET_LAT+0.5}",
            'parameter': 'pm25',
            'limit': 10
        }
        
        response = requests.get(locations_url, params=location_params, headers=headers, timeout=30)
        response.raise_for_status()
        locations_data = response.json()
        
        locations = locations_data.get('results', [])
        if not locations:
            print("❌ No PM2.5 monitoring locations found in Delhi area.")
            return pd.DataFrame()
            
        print(f"✅ Found {len(locations)} PM2.5 monitoring locations in Delhi")
        
        # Step 2: Get detailed sensor information for each location
        all_measurements = []
        
        for i, location in enumerate(locations[:3]):  # Process first 3 locations
            location_id = location.get('id')
            location_name = location.get('name', 'Unknown')
            
            print(f"  {i+1}. Processing {location_name} (ID: {location_id})")
            
            # Get detailed location info including sensors
            location_detail_url = f"https://api.openaq.org/v3/locations/{location_id}"
            
            try:
                detail_response = requests.get(location_detail_url, headers=headers, timeout=15)
                if detail_response.status_code != 200:
                    print(f"    ⚠️ Could not get sensor details for {location_name}")
                    continue
                
                detail_data = detail_response.json()
                detail_results = detail_data.get('results', [])
                
                if not detail_results:
                    continue
                
                location_detail = detail_results[0]
                sensors = location_detail.get('sensors', [])
                
                # Find PM2.5 sensor
                pm25_sensor_id = None
                for sensor in sensors:
                    parameter = sensor.get('parameter', {})
                    param_name = parameter.get('name', '')
                    
                    if param_name == 'pm25':
                        pm25_sensor_id = sensor.get('id')
                        print(f"    📡 Found PM2.5 sensor: {pm25_sensor_id}")
                        break
                
                if not pm25_sensor_id:
                    print(f"    ⚠️ No PM2.5 sensor found at {location_name}")
                    continue
                
                # Step 3: Get measurements using the working sensors endpoint
                measurements_url = f"https://api.openaq.org/v3/sensors/{pm25_sensor_id}/measurements"
                measurements_params = {
                    'limit': 100,  # Get recent measurements
                    'sort': 'datetime',
                    'order': 'desc'
                }
                
                measurements_response = requests.get(
                    measurements_url, 
                    params=measurements_params,
                    headers=headers, 
                    timeout=15
                )
                
                if measurements_response.status_code == 200:
                    measurements_data = measurements_response.json()
                    measurements = measurements_data.get('results', [])
                    
                    print(f"    📊 Retrieved {len(measurements)} measurements")
                    
                    # Process measurements
                    for measurement in measurements:
                        value = measurement.get('value')
                        datetime_str = measurement.get('datetime')
                        
                        # Skip invalid measurements
                        if value is None or value < 0:
                            continue
                        
                        # Handle missing datetime by using current date as approximation
                        if not datetime_str:
                            from datetime import datetime
                            datetime_str = datetime.now().isoformat()
                        
                        all_measurements.append({
                            'date': datetime_str,
                            'pm25_ugm3': value,
                            'location': location_name,
                            'location_id': location_id,
                            'sensor_id': pm25_sensor_id
                        })
                else:
                    print(f"    ❌ Failed to get measurements: {measurements_response.status_code}")
                    
            except requests.RequestException as e:
                print(f"    ❌ Error getting data for {location_name}: {e}")
                continue
        
        # Step 4: Process collected measurements
        if all_measurements:
            df_openaq = pd.DataFrame(all_measurements)
            df_openaq['date'] = pd.to_datetime(df_openaq['date'], errors='coerce')
            
            # Remove rows with invalid dates
            df_openaq = df_openaq.dropna(subset=['date'])
            
            if df_openaq.empty:
                print("⚠️ No valid measurements with proper timestamps")
                return pd.DataFrame()
            
            # Group by date (day) and take mean if multiple measurements per day
            df_openaq['date_only'] = df_openaq['date'].dt.date
            df_openaq = df_openaq.groupby('date_only').agg({
                'pm25_ugm3': 'mean',
                'location': 'first',
                'date': 'first'
            }).reset_index(drop=True)
            
            # Sort by date
            df_openaq = df_openaq.sort_values('date')
            
            print(f"✅ Successfully processed {len(df_openaq)} PM2.5 measurements")
            print(f"   Date range: {df_openaq['date'].min()} to {df_openaq['date'].max()}")
            print(f"   PM2.5 range: {df_openaq['pm25_ugm3'].min():.1f} - {df_openaq['pm25_ugm3'].max():.1f} µg/m³")
            print(f"   Average PM2.5: {df_openaq['pm25_ugm3'].mean():.1f} µg/m³")
            
            return df_openaq[['date', 'pm25_ugm3', 'location']]
        else:
            print("⚠️ No measurements could be retrieved from any location")
            print("✅ Continuing with satellite and weather data (sufficient for ML models)")
            return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ OpenAQ API connection failed: {e}")
        print("✅ Continuing without ground truth data")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Unexpected error with OpenAQ: {e}")
        return pd.DataFrame()


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