"""
Environment Configuration Module
Centralized management of API keys and environment variables
"""
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for environment variables"""
    
    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    NASA_API_KEY = os.getenv('NASA_API_KEY')
    OPENAQ_API_KEY = os.getenv('OPENAQ_API_KEY')
    
    # OpenMeteo Configuration
    OPENMETEO_BASE_URL = os.getenv('OPENMETEO_BASE_URL', 'https://air-quality-api.open-meteo.com/v1/air-quality')
    OPENMETEO_WEATHER_URL = os.getenv('OPENMETEO_WEATHER_URL', 'https://api.open-meteo.com/v1/forecast')
    
    # Location Configuration
    TARGET_CITY = os.getenv('TARGET_CITY', 'Delhi')
    TARGET_LATITUDE = float(os.getenv('TARGET_LATITUDE', '28.6139'))
    TARGET_LONGITUDE = float(os.getenv('TARGET_LONGITUDE', '77.2090'))
    
    # System Configuration
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that required API keys are present"""
        missing_keys = []
        
        if not cls.GEMINI_API_KEY:
            missing_keys.append('GEMINI_API_KEY')
            
        if missing_keys:
            st.error(f"""
            🔑 **Missing API Keys**: {', '.join(missing_keys)}
            
            Please add these to your `.env` file:
            ```
            GEMINI_API_KEY=your_api_key_here
            ```
            
            Get your Gemini API key from: https://makersuite.google.com/app/apikey
            """)
            return False
        return True
    
    @classmethod
    def get_api_key(cls, key_name):
        """Safely get API key with error handling"""
        key = getattr(cls, key_name, None)
        if not key:
            st.warning(f"⚠️ API key '{key_name}' not found in environment variables")
        return key
    
    @classmethod
    def show_config_status(cls):
        """Display configuration status in sidebar"""
        with st.sidebar:
            st.subheader("🔧 Configuration Status")
            
            # API Keys Status
            if cls.GEMINI_API_KEY:
                st.success("✅ Gemini API Key: Loaded")
            else:
                st.error("❌ Gemini API Key: Missing")
                
            if cls.NASA_API_KEY:
                st.success("✅ NASA API Key: Loaded")
            else:
                st.info("ℹ️ NASA API Key: Not configured")
                
            if cls.OPENAQ_API_KEY:
                st.success("✅ OpenAQ API Key: Loaded")
            else:
                st.info("ℹ️ OpenAQ API Key: Not configured")
            
            # Location
            st.info(f"📍 Target: {cls.TARGET_CITY} ({cls.TARGET_LATITUDE}, {cls.TARGET_LONGITUDE})")
            
            # Cache
            st.info(f"⏰ Cache TTL: {cls.CACHE_TTL}s")

# Create global config instance
config = Config()

# Convenience functions for backward compatibility
def get_gemini_api_key():
    """Get Gemini API key"""
    return config.get_api_key('GEMINI_API_KEY')

def get_nasa_api_key():
    """Get NASA API key"""
    return config.get_api_key('NASA_API_KEY')

def get_openaq_api_key():
    """Get OpenAQ API key"""
    return config.get_api_key('OPENAQ_API_KEY')

def validate_environment():
    """Validate environment configuration"""
    return config.validate_api_keys()
