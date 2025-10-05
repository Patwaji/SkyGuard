# Scripts Execution Guide

This directory contains all data processing and model training scripts. All scripts now use **absolute paths** and will save files to the correct directories regardless of where you run them from.

## 📁 Directory Structure

```
scripts/
├── 01_fetch_data.py           # Fetch satellite and ground data
├── 02_process_data.py          # Process and merge all data
├── 03_train_model.py           # Train Prophet and regression models
├── 04_train_advanced_model.py  # Train advanced ML models
├── test_paths.py               # Test script to verify paths
└── README.md                   # This file
```

## 🚀 Execution Order

Run scripts in this order:

### 1. Data Fetching
```bash
cd scripts
python 01_fetch_data.py
```
**Output:** 
- `../data/raw/openaq_historical_raw.csv`
- `../data/raw/weather_historical_raw.csv`
- Google Drive exports: `AOD_RAW.csv`, `NO2_RAW.csv`

### 2. Data Processing  
```bash
cd scripts
python 02_process_data.py
```
**Output:**
- `../data/processed/historical_merged_master.csv`

### 3. Model Training
```bash
cd scripts
python 03_train_model.py
```
**Output:**
- `../models/prophet_predictor_model.pkl`
- `../models/policy_regression_model.pkl`

### 4. Advanced Model Training
```bash
cd scripts
python 04_train_advanced_model.py
```
**Output:**
- `../models/advanced_air_quality_model.pkl`

## 🔧 Path Configuration

All scripts now use:
- **Environment variables** from `.env` file for API keys
- **Absolute paths** calculated from script location
- **Automatic directory creation** for output folders

### Directory Variables:
- `RAW_DATA_DIR = ../data/raw/`
- `PROCESSED_DATA_DIR = ../data/processed/`  
- `MODEL_DIR = ../models/`

## ✅ Testing

Run the test script to verify all paths work:
```bash
cd scripts
python test_paths.py
```

## 📋 Requirements

Make sure you have installed:
```bash
pip install pandas numpy scikit-learn prophet python-dotenv earthengine-api openaq meteostat
```

## 🔑 Environment Setup

1. Copy `.env.example` to `.env`
2. Add your API keys:
   ```
   GEMINI_API_KEY=your_gemini_key_here
   OPENAQ_API_KEY=your_openaq_key_here
   ```

## 🚨 Common Issues

### "File not found" errors:
- Make sure you run scripts from the `scripts/` directory
- Check that data directories exist (run `test_paths.py`)

### Google Earth Engine errors:
- Authenticate: `ee.Authenticate()` 
- Initialize with project: `ee.Initialize(project='your-project-id')`

### Missing dependencies:
- Install required packages listed above
- Check Python version (3.8+ recommended)

## 📊 Output Verification

After running all scripts, you should have:
```
data/
├── raw/
│   ├── openaq_historical_raw.csv
│   ├── weather_historical_raw.csv
│   ├── AOD_RAW.csv (from Google Drive)
│   └── NO2_RAW.csv (from Google Drive)
├── processed/
│   └── historical_merged_master.csv
models/
├── prophet_predictor_model.pkl
├── policy_regression_model.pkl
└── advanced_air_quality_model.pkl
```

## 🎯 Next Steps

After running all scripts:
1. Start the Streamlit app: `streamlit run app/main_app.py`
2. All models and data will be automatically loaded
3. Use the web interface to explore forecasts and visualizations
