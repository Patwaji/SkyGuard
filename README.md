# 🌫️ SkyGuard - Air Quality Monitoring System

A comprehensive air quality monitoring and prediction system for Delhi, India, featuring real-time data visualization, ML-powered forecasting, and AI-driven policy recommendations.

## # Install dependencies
pip install -r requirements.txt

# Start development server
streamlit run app/main_app.py --server.runOnSave truees

### 📊 **Real-Time Monitoring**
- **3D Air Quality Visualization**: Interactive 3D pollution cloud mapping
- **Live Data Integration**: Real-time data from OpenMeteo API
- **Multi-Pollutant Tracking**: PM2.5, NO2, SO2, CO, Ozone monitoring
- **Health Impact Assessment**: WHO-standard air quality categorization

### 🔮 **AI-Powered Forecasting** 
- **48-Hour Predictions**: Prophet-based time series forecasting
- **Multi-Model Ensemble**: Advanced ML models (Random Forest, Gradient Boosting)
- **Weather Integration**: Temperature, humidity, wind speed correlation
- **Seasonal Analysis**: Historical trend identification and projection

### 🏛️ **Policy Advisory System**
- **AI Policy Assistant**: Gemini-powered policy recommendations
- **Impact Simulation**: Policy intervention effectiveness modeling
- **Multi-Model AI**: Different AI models for various query types
- **Conversational Interface**: Natural language policy discussion

### 🛰️ **Satellite Data Integration**
- **NASA/ESA Data**: AOD and NO2 satellite measurements
- **Google Earth Engine**: Large-scale geospatial analysis
- **Historical Archives**: 5+ years of satellite data processing

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy plotly prophet scikit-learn python-dotenv requests earthengine-api openaq meteostat joblib
```

### 1. Environment Setup
```bash
# Clone/download the project
cd skyguard

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Get Gemini API key from: https://makersuite.google.com/app/apikey
```

### 2. Run the Application
```bash
streamlit run app/main_app.py
```

### 3. Access SkyGuard
Open your browser to: `http://localhost:8501`

## 📁 Project Structure

```
skyguard/
├── 📱 app/                          # Streamlit application
│   ├── main_app.py                  # Main application entry point
│   ├── pages/                       # Individual application pages
│   │   ├── 1_Forecast.py           # 48-hour prediction page
│   │   ├── 2_Policy_Advisor.py     # AI policy recommendations
│   │   ├── 3_NASA_Satellite_Data.py # Satellite data visualization
│   │   └── 4_3D_Air_Quality.py     # 3D pollution mapping
│   ├── components/                  # Reusable UI components
│   │   └── data_scaling_info.py    # Data scaling utilities
│   └── config/                      # Configuration management
│       └── env_config.py            # Environment variables handler
├── 📊 data/                         # Data storage
│   ├── raw/                         # Raw data from APIs/satellites
│   │   ├── openaq_historical_raw.csv
│   │   ├── weather_historical_raw.csv
│   │   ├── AOD_RAW.csv             # From Google Earth Engine
│   │   └── NO2_RAW.csv             # From Google Earth Engine
│   └── processed/                   # Cleaned, merged datasets
│       └── historical_merged_master.csv
├── 🤖 models/                       # Trained ML models
│   ├── prophet_predictor_model.pkl  # Time series forecasting
│   ├── policy_regression_model.pkl  # Policy impact modeling
│   └── advanced_air_quality_model.pkl # Advanced ML ensemble
├── 🔧 scripts/                      # Data processing pipeline
│   ├── 01_fetch_data.py            # Data collection from APIs
│   ├── 02_process_data.py          # Data cleaning and merging
│   ├── 03_train_model.py           # Basic model training
│   └── 04_train_advanced_model.py  # Advanced model training
├── 🔐 .env                         # Environment variables (create from .env.example)
├── 🔐 .env.example                 # Environment template
├── 📋 .gitignore                   # Git ignore rules
└── 📖 README.md                    # This file
```

## 🛠️ Data Pipeline

### Phase 1: Data Collection
```bash
cd scripts
python 01_fetch_data.py
```
- Fetches OpenAQ ground monitoring data
- Retrieves weather data from Meteostat
- Initiates Google Earth Engine satellite data exports

### Phase 2: Data Processing
```bash
python 02_process_data.py
```
- Cleans and normalizes all data sources
- Handles missing values and outliers
- Creates time-aligned merged dataset

### Phase 3: Model Training
```bash
python 03_train_model.py          # Basic models
python 04_train_advanced_model.py # Advanced ensemble
```
- Trains Prophet forecasting models
- Develops policy regression models
- Creates advanced ML ensemble models

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAQ_API_KEY=your_openaq_key_here

# Location Settings
TARGET_CITY=Delhi
TARGET_LATITUDE=28.6139
TARGET_LONGITUDE=77.2090

# API Endpoints
OPENMETEO_BASE_URL=https://air-quality-api.open-meteo.com/v1/air-quality
OPENMETEO_WEATHER_URL=https://api.open-meteo.com/v1/forecast

# System Settings
DEBUG_MODE=False
CACHE_TTL=3600
```

### API Keys Required
1. **Gemini AI API**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **OpenAQ API**: Get from [OpenAQ](https://openaq.org/developers/)
3. **Google Earth Engine**: Authenticate via `ee.Authenticate()`

## 📊 Application Pages

### 🔮 Forecast Page
- **48-hour predictions** using Prophet + Advanced ML
- **Weather correlation** analysis
- **Historical comparison** with seasonal trends
- **Confidence intervals** and model uncertainty

### 🏛️ Policy Advisor
- **AI-powered recommendations** using Gemini AI
- **Policy impact simulation** with regression models
- **Conversational interface** for natural language queries
- **Multi-model AI** for different query types

### 🛰️ NASA Satellite Data
- **Real-time satellite imagery** integration
- **AOD and NO2 visualizations** from space
- **Historical satellite trends** analysis
- **Ground-truth correlation** with satellite data

### 🌫️ 3D Air Quality
- **Interactive 3D pollution clouds** visualization
- **Real-time Delhi air quality** from OpenMeteo
- **Altitude-based pollution modeling**
- **Hotspot identification** and risk mapping

## 🔬 Data Science Details

### Data Sources
- **OpenAQ**: Ground-level PM2.5, NO2, O3 measurements
- **OpenMeteo**: Real-time air quality and weather data
- **NASA Earthdata**: AOD measurements 
- **TROPOMI**: NO2 column density from Sentinel-5P
- **Meteostat**: Historical weather data

### Machine Learning Models
- **Prophet**: Facebook's time series forecasting
- **Random Forest**: Ensemble learning for non-linear patterns
- **Gradient Boosting**: Advanced ensemble with boosting
- **Ridge Regression**: Linear model with regularization
- **Polynomial Features**: Non-linear feature engineering

### Data Scaling Solution
The project handles the complexity of different data scales:
- **Satellite AOD**: Values range 0-400,000 (proxy measurements)
- **Real PM2.5**: Values range 0-500 µg/m³ (WHO standards)
- **Automatic scaling**: 1000x factor applied for visualization
- **Transparent labeling**: Clear indicators of scaled vs. real values

## 🚨 Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt  # Install dependencies
```

**"API Key Error"**
- Check your `.env` file has correct API keys
- Verify API keys are valid and not expired
- Ensure no spaces around `=` in `.env` file

**"File Not Found"**
- Run scripts from their respective directories
- Ensure data directories exist

**"Google Earth Engine Authentication"**
```bash
import ee
ee.Authenticate()  # Follow browser authentication
ee.Initialize(project='your-project-id')
```

### Performance Optimization
- **Caching**: Streamlit caches API calls for 1 hour
- **Data Sampling**: Large datasets are automatically sampled
- **Background Processing**: Long operations run asynchronously

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python scripts/test_paths.py

# Start development server
streamlit run app/main_app.py --server.runOnSave true
```

## 📈 Roadmap

### Short Term (v1.1)
- [ ] Real-time alerts and notifications
- [ ] Mobile-responsive design improvements
- [ ] Additional pollutant tracking (PM10, SO2)
- [ ] Export functionality for reports

### Medium Term (v1.2)
- [ ] Multi-city support beyond Delhi
- [ ] Advanced ML model interpretability
- [ ] Historical data analysis tools
- [ ] API for external integrations

### Long Term (v2.0)
- [ ] Real-time IoT sensor integration
- [ ] Machine learning model marketplace
- [ ] Collaborative policy modeling
- [ ] International city comparisons

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAQ and NASA Earthdata** for open air quality data
- **OpenMeteo** for weather and air quality APIs
- **Google Earth Engine** for satellite data processing
- **NASA/ESA** for satellite imagery and measurements
- **Streamlit** for the amazing web framework
- **Facebook Prophet** for time series forecasting
- **Google Gemini AI** for policy recommendations

## 📞 Contact

For questions, suggestions, or support:
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: [Add your contact email]

---

**🌍 Built with ❤️ for cleaner air and healthier cities**
