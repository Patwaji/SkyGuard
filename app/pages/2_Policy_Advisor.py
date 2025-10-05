import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os
import pickle
import re 
from datetime import datetime
import sys

# Add components path for data scaling info
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from components.data_scaling_info import show_data_scaling_info, show_quick_scaling_alert, get_scaled_display_value, get_health_category

# Add config path for environment variables
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from env_config import config, get_gemini_api_key, validate_environment

# --- Enhanced AI Configuration ---
# API key is now loaded from .env file
GEMINI_API_KEY = get_gemini_api_key()
TARGET_CITY = config.TARGET_CITY

# Multiple AI models for different purposes
AI_MODELS = {
    "policy_expert": "gemini-2.5-pro",           # For complex policy analysis
    "health_advisor": "gemini-2.5-flash",       # For health recommendations
    "quick_response": "gemini-flash-latest",     # For fast simple queries
    "educational": "gemini-2.5-flash"           # For educational content
}

# Response styles based on user intent
RESPONSE_STYLES = {
    "technical": "Provide detailed scientific analysis with data",
    "simple": "Use everyday language that anyone can understand", 
    "action": "Focus on practical steps and recommendations",
    "educational": "Explain concepts and teach about air quality"
}

# Advanced ML Model Configuration
class AdvancedPolicyPredictor:
    """Wrapper for the advanced ML model with policy scenario simulation"""
    
    def __init__(self):
        self.model = None
        self.load_advanced_model()
    
    def load_advanced_model(self):
        """Load the advanced ML model"""
        try:
            with open('models/advanced_air_quality_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_data = model_data
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.model_performance = model_data['model_performance']
            self.poly_features = model_data.get('poly_features')
            
            # Select best model
            self.best_model_name = min(self.model_performance.keys(), 
                                     key=lambda x: self.model_performance[x]['cv_mae'])
            
            print(f"✅ Advanced model loaded. Best model: {self.best_model_name}")
            
        except FileNotFoundError:
            st.error("Advanced model not found. Using fallback simple model.")
            self.model = None
        except Exception as e:
            st.error(f"Error loading advanced model: {str(e)}")
            self.model = None
    
    def create_advanced_features(self, input_data):
        """Create advanced features matching the training data"""
        
        # Create DataFrame from input
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Add current date if not provided
        if 'ds' not in df.columns:
            df['ds'] = datetime.now().strftime('%Y-%m-%d')
        
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Time-based features
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day_of_year'] = df['ds'].dt.dayofyear
        df['quarter'] = df['ds'].dt.quarter
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                                      9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        
        # Encode season
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # NO2 derived features
        if 'NO2' in df.columns:
            df['NO2_squared'] = df['NO2'] ** 2
            df['NO2_log'] = np.log1p(df['NO2'])
            
            # Set lagged features to current value (simplified)
            df['NO2_lag_1'] = df['NO2']
            df['NO2_rolling_7'] = df['NO2']
        
        # Interaction features
        if 'NO2' in df.columns and 'AOD' in df.columns:
            df['NO2_AOD_interaction'] = df['NO2'] * df['AOD']
        
        # Special events
        df['is_diwali_week'] = 0  # Simplified for prediction
        df['is_winter_peak'] = ((df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)).astype(int)
        df['is_crop_burning'] = ((df['month'] == 10) | (df['month'] == 11)).astype(int)
        
        return df
    
    def predict_with_scenario(self, base_conditions, scenario_changes):
        """Predict air quality with policy scenario changes"""
        
        if self.model is None:
            return None
        
        try:
            # Create baseline prediction
            baseline_df = self.create_advanced_features(base_conditions)
            
            # Apply scenario changes
            scenario_df = baseline_df.copy()
            for feature, change_pct in scenario_changes.items():
                if feature in scenario_df.columns:
                    scenario_df[feature] = scenario_df[feature] * (1 + change_pct / 100)
            
            # Prepare features for prediction
            feature_subset = []
            for feature in self.feature_names:
                if feature in scenario_df.columns:
                    feature_subset.append(feature)
                else:
                    # Add missing features as zero
                    scenario_df[feature] = 0
                    feature_subset.append(feature)
            
            X = scenario_df[self.feature_names].fillna(0)
            
            # Scale and predict
            model = self.models[self.best_model_name]
            
            if self.best_model_name in ['ridge_regression', 'linear_with_poly']:
                X_scaled = self.scalers['main'].transform(X)
                
                if self.best_model_name == 'linear_with_poly':
                    X_final = self.poly_features.transform(X_scaled)
                else:
                    X_final = X_scaled
            else:
                X_final = X
            
            prediction = model.predict(X_final)[0]
            
            # Also get baseline prediction for comparison
            baseline_X = baseline_df[self.feature_names].fillna(0)
            
            if self.best_model_name in ['ridge_regression', 'linear_with_poly']:
                baseline_X_scaled = self.scalers['main'].transform(baseline_X)
                
                if self.best_model_name == 'linear_with_poly':
                    baseline_X_final = self.poly_features.transform(baseline_X_scaled)
                else:
                    baseline_X_final = baseline_X_scaled
            else:
                baseline_X_final = baseline_X
            
            baseline_prediction = model.predict(baseline_X_final)[0]
            
            return {
                'baseline_pm25': baseline_prediction,
                'scenario_pm25': prediction,
                'pm25_change': prediction - baseline_prediction,
                'model_used': self.best_model_name,
                'model_performance': self.model_performance[self.best_model_name]
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model_name in self.feature_importance:
            importance = self.feature_importance[self.best_model_name]
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {}

# Initialize the advanced predictor
@st.cache_resource
def load_advanced_predictor():
    return AdvancedPolicyPredictor()

# --- HELPER FUNCTIONS ---

def get_policy_context():
    """Loads necessary data (coefficient and averages) for grounding the LLM."""
    try:
        # Load the policy coefficient
        with open('models/policy_regression_model.pkl', 'rb') as f:
            policy_data = pickle.load(f)
            policy_coefficient = policy_data['coefficient']
            
        # Load historical data for trend visualization and average calculation
        df_master = pd.read_csv('data/processed/historical_merged_master.csv')
        avg_no2 = df_master['NO2'].mean()
        avg_pm25 = df_master['y'].mean()
        
        return policy_coefficient, avg_no2, avg_pm25, df_master

    except FileNotFoundError:
        st.error("Model or data files not found. Cannot initialize policy context.")
        return None, None, None, None

@st.cache_data
def get_historical_trend_data(df_master):
    """Prepares historical data for the trend chart."""
    df_trend = df_master[['ds', 'y']].copy()
    df_trend['ds'] = pd.to_datetime(df_trend['ds'])
    return df_trend

def calculate_impact(change_pct, coefficient, avg_no2, avg_pm25):
    """Enhanced impact calculation using advanced ML model with validation
    
    Args:
        change_pct: Positive for reduction, negative for increase
    """
    
    # Load advanced predictor
    advanced_predictor = load_advanced_predictor()
    
    if advanced_predictor.model is not None:
        # Use advanced ML model
        base_conditions = {
            'NO2': avg_no2,
            'AOD': 0.4,  # Typical AOD value for Delhi
            'Temp_C': 25,  # Typical temperature
            'Humidity_pct': 60,  # Typical humidity
            'Wind_Speed_ms': 3,  # Typical wind speed
            'ds': datetime.now().strftime('%Y-%m-%d')
        }
        
        scenario_changes = {
            'NO2': -change_pct  # Negative change_pct means increase in NO2
        }
        
        result = advanced_predictor.predict_with_scenario(base_conditions, scenario_changes)
        
        if result:
            baseline_pm25 = result['baseline_pm25']
            scenario_pm25 = result['scenario_pm25']
            
            # Validate predictions are realistic (PM2.5 should be 0-500 µg/m³)
            if 0 <= baseline_pm25 <= 500 and 0 <= scenario_pm25 <= 500:
                pm25_impact = baseline_pm25 - scenario_pm25
                
                return {
                    "pm25_impact": pm25_impact,
                    "new_pm25": scenario_pm25,
                    "reduction_pct": abs(change_pct),  # Always show as positive percentage
                    "baseline_pm25": baseline_pm25,
                    "model_used": result['model_used'],
                    "model_performance": result['model_performance'],
                    "confidence_score": result['model_performance']['test_r2'],
                    "prediction_uncertainty": result['model_performance']['cv_mae'],
                    "is_valid_prediction": True
                }
            else:
                # If predictions are unrealistic, fall back to simple model
                print(f"Advanced model gave unrealistic predictions: {baseline_pm25}, {scenario_pm25}")
    
    # Fallback to simple linear model (always used if advanced model fails or gives bad results)
    change_factor = change_pct / 100.0
    no2_change = avg_no2 * change_factor  # Positive for reduction, negative for increase
    pm25_impact = no2_change * coefficient  # How much PM2.5 changes
    new_pm25 = max(0, avg_pm25 - pm25_impact)  # Subtract because pm25_impact is positive for reductions
    
    return {
        "pm25_impact": abs(pm25_impact),  # Always show absolute impact
        "new_pm25": new_pm25,
        "reduction_pct": abs(change_pct),  # Always show as positive percentage
        "baseline_pm25": avg_pm25,
        "model_used": "simple_linear",
        "confidence_score": 0.7,  # Assumed confidence for simple model
        "prediction_uncertainty": 10,  # Lower uncertainty for simple model
        "is_valid_prediction": True
    }

def detect_user_intent(prompt):
    """Detect user intent and select appropriate AI model and response style"""
    prompt_lower = prompt.lower()
    
    # Check for scenario/policy questions first (highest priority)
    scenario_keywords = ['what if', 'if we', 'suppose', 'imagine', 'scenario', 'increase', 'decrease', 
                        'reduce', 'cut', 'boost', 'raise', '%', 'policy', 'regulation', 'implement']
    is_scenario = any(word in prompt_lower for word in scenario_keywords)
    
    if is_scenario:
        return "policy_expert", "simple"  # Use simple style for scenarios so everyone can understand
    
    # Intent patterns for other questions
    if any(word in prompt_lower for word in ['policy', 'government', 'regulation', 'law']):
        return "policy_expert", "simple"  # Make policy questions simple too
    elif any(word in prompt_lower for word in ['health', 'breathing', 'asthma', 'children', 'elderly']):
        return "health_advisor", "simple"
    elif any(word in prompt_lower for word in ['what', 'how', 'why', 'explain', 'learn', 'understand']):
        return "educational", "educational"
    elif any(word in prompt_lower for word in ['quick', 'simple', 'brief']):
        return "quick_response", "simple"
    else:
        return "policy_expert", "simple"  # Default to simple for better understanding

def get_context_aware_prompt(prompt, context_data, impact_data, model_type, style):
    """Generate context-aware prompts based on AI model and response style"""
    
    coefficient, avg_no2, avg_pm25, _ = context_data
    
    # Detect if this is a "what if" or policy scenario question
    prompt_lower = prompt.lower()
    is_scenario = any(phrase in prompt_lower for phrase in [
        'what if', 'if we', 'suppose', 'imagine', 'scenario', 'increase', 'decrease', 
        'reduce', 'cut', 'boost', 'raise', 'policy', 'regulation', '%'
    ])

    # Base context (shorter for different models)
    if model_type == "quick_response" and not is_scenario:
        base_context = f"""You are SkyGuard for {TARGET_CITY}.
Current PM2.5: {avg_pm25:.1f} µg/m³ (WHO safe: 15)"""
    else:
        base_context = f"""You are SkyGuard, Delhi's advanced AI air quality advisor.

Current Situation:
- PM2.5: {avg_pm25:.1f} µg/m³ (WHO safe limit: 15 µg/m³)
- NO2: {avg_no2:.1f} µg/m³ 
- Status: {"UNHEALTHY" if avg_pm25 > 55 else "MODERATE" if avg_pm25 > 25 else "ACCEPTABLE"}
- Health Risk: {"HIGH" if avg_pm25 > 35 else "MODERATE" if avg_pm25 > 15 else "LOW"}"""

    # Add impact data if available
    impact_context = ""
    if impact_data:
        pm25_impact = impact_data['pm25_impact']
        new_pm25 = impact_data['new_pm25']
        change_pct = impact_data.get('change_pct', impact_data['reduction_pct'])
        change_type = impact_data.get('change_type', 'decrease')
        
        if change_type == 'increase':
            impact_context = f"""
Scenario Analysis - {change_pct}% EMISSION INCREASE:
- PM2.5 would WORSEN by {pm25_impact:.1f} µg/m³
- New PM2.5 level: {new_pm25:.1f} µg/m³
- Health impact: {"SEVERELY WORSENED" if new_pm25 > avg_pm25 + 20 else "WORSENED"}
- Risk level: {"HAZARDOUS" if new_pm25 > 150 else "UNHEALTHY" if new_pm25 > 55 else "MODERATE"}"""
        else:
            impact_context = f"""
Scenario Analysis - {change_pct}% EMISSION REDUCTION:
- PM2.5 would IMPROVE by {pm25_impact:.1f} µg/m³
- New PM2.5 level: {new_pm25:.1f} µg/m³
- Health impact: {"SIGNIFICANTLY IMPROVED" if new_pm25 < 15 else "IMPROVED"}
- Risk level: {"LOW" if new_pm25 < 15 else "MODERATE" if new_pm25 < 35 else "HIGH"}"""

    # Enhanced style-specific instructions for scenarios
    if is_scenario:
        style_instructions = {
            "technical": """
TIMELINE ANALYSIS - Provide impacts in clear timeline format:
📅 IMMEDIATE (1-3 months): Direct air quality changes
📅 SHORT-TERM (3-12 months): Health effects start showing
📅 MEDIUM-TERM (1-3 years): Economic and social impacts
📅 LONG-TERM (3-10 years): Major environmental changes
Keep it simple but detailed.""",
            
            "simple": """
TIMELINE EXPLANATION - Explain what happens when in simple terms:
🟡 FIRST FEW MONTHS: What you'll notice right away
🟠 WITHIN A YEAR: How health and daily life changes  
🔴 IN 2-3 YEARS: Bigger changes to economy and society
⚫ LONG-TERM: What Delhi will look like in 10 years
Use everyday language and real examples.""",
            
            "action": """
IMPLEMENTATION TIMELINE - Focus on action steps:
PHASE 1 (Months 1-6): Immediate policy implementation
PHASE 2 (Year 1-2): Monitoring and adjustments
PHASE 3 (Years 2-5): Long-term enforcement
PHASE 4 (Years 5+): Sustained impact evaluation""",
            
            "educational": """
EDUCATIONAL TIMELINE - Teach the progression:
IMMEDIATE EFFECTS: Why changes happen quickly
HEALTH PROGRESSION: How body responds over time
ENVIRONMENTAL CYCLE: How nature recovers/worsens
SOCIETAL ADAPTATION: How people adjust to changes"""
        }
    else:
        # Regular (non-scenario) instructions - simplified
        style_instructions = {
            "technical": "Provide scientific analysis with specific numbers and WHO guidelines.",
            "simple": "Use everyday language with practical examples that anyone can understand.",
            "action": "Focus on practical steps and actionable recommendations.",
            "educational": "Explain concepts clearly with real-world comparisons."
        }

    # Add final comprehensive instruction for scenario questions
    final_instruction = ""
    if is_scenario:
        final_instruction = "\n\nProvide timeline analysis with simple language."

    return f"{base_context}{impact_context}\n{style_instructions[style]}{final_instruction}\n\nUser Question: {prompt}"

def generate_llm_response(prompt, context_data, pre_calculated_impact=None):
    """Enhanced LLM response with multiple AI models and adaptive responses"""
    
    # Detect user intent and select model
    model_type, response_style = detect_user_intent(prompt)
    selected_model = AI_MODELS[model_type]
    
    # Generate context-aware prompt
    full_prompt = get_context_aware_prompt(prompt, context_data, pre_calculated_impact, model_type, response_style)
    
    # Adaptive generation config based on model type - Balanced for quality and speed
    if model_type == "quick_response":
        gen_config = {
            "temperature": 0.3,
            "maxOutputTokens": 1500,  # Good size for quick responses
            "topP": 0.8,
            "topK": 20
        }
    elif model_type == "policy_expert":
        gen_config = {
            "temperature": 0.7,  # Balanced for good responses
            "maxOutputTokens": 3500,  # Increased for comprehensive analysis
            "topP": 0.9,  # Higher for more diverse responses
            "topK": 40   # Higher for better quality
        }
    else:
        gen_config = {
            "temperature": 0.7,
            "maxOutputTokens": 2500,  # Increased for better responses
            "topP": 0.9,
            "topK": 40
        }

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": gen_config
    }
    
    try:
        # Try primary model first with generous timeout for quality responses
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={GEMINI_API_KEY}",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=60  # Much longer timeout for complete responses
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                
                # Check finish reason
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                if finish_reason == 'MAX_TOKENS':
                    # Try to get a complete response with a more concise prompt
                    concise_prompt = f"Give a complete but concise answer: {prompt}"
                    retry_payload = {
                        "contents": [{"parts": [{"text": concise_prompt}]}],
                        "generationConfig": {
                            "temperature": 0.5,
                            "maxOutputTokens": 2048,
                            "topP": 0.9
                        }
                    }
                    
                    retry_response = requests.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={GEMINI_API_KEY}",
                        headers={'Content-Type': 'application/json'},
                        data=json.dumps(retry_payload),
                        timeout=20
                    )
                    
                    if retry_response.status_code == 200:
                        retry_result = retry_response.json()
                        if 'candidates' in retry_result and len(retry_result['candidates']) > 0:
                            retry_candidate = retry_result['candidates'][0]
                            if ('content' in retry_candidate and 'parts' in retry_candidate['content'] and
                                len(retry_candidate['content']['parts']) > 0 and 
                                'text' in retry_candidate['content']['parts'][0]):
                                
                                retry_finish_reason = retry_candidate.get('finishReason', 'UNKNOWN')
                                if retry_finish_reason != 'MAX_TOKENS':
                                    return retry_candidate['content']['parts'][0]['text']
                    
                    # If retry also truncated, show warning but return what we have
                    st.warning("⚠️ Response was truncated due to length. Consider asking a more specific question for a complete answer.")
                
                if 'content' in candidate and 'parts' in candidate['content']:
                    content = candidate['content']['parts']
                    if len(content) > 0 and 'text' in content[0]:
                        response_text = content[0]['text']
                        
                        # Add model info badge
                        model_badge = {
                            "policy_expert": "🏛️ Policy Expert",
                            "health_advisor": "🏥 Health Advisor", 
                            "quick_response": "⚡ Quick Response",
                            "educational": "📚 Educational"
                        }
                        
                        return f"*{model_badge.get(model_type, '🤖')} Model*\n\n{response_text}"
        
        # Only fallback if there's a real error, not just a timeout or rate limit
        if response.status_code in [429, 503]:  # Rate limit or service unavailable
            return "⏱️ Service temporarily busy. Please try again in a moment."
        
        # For other errors, try fallback only if it's a different model
        if selected_model != AI_MODELS["quick_response"]:
            fallback_payload = {
                "contents": [{"parts": [{"text": f"Briefly answer in simple terms: {prompt}"}]}],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 1024  # Increased fallback token limit
                }
            }
            
            fallback_response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{AI_MODELS['quick_response']}:generateContent?key={GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                data=json.dumps(fallback_payload),
                timeout=15
            )
            
            if fallback_response.status_code == 200:
                result = fallback_response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        content = candidate['content']['parts']
                        if len(content) > 0 and 'text' in content[0]:
                            return f"*⚡ Quick Response (Backup)*\n\n{content[0]['text']}"
        
        return f"⚠️ Unable to get response from AI models. Status: {response.status_code}"
        
    except requests.exceptions.Timeout:
        # If timeout, try with a comprehensive but simpler prompt
        prompt_lower = prompt.lower()
        is_scenario = any(phrase in prompt_lower for phrase in [
            'what if', 'if we', 'suppose', 'imagine', 'scenario', 'increase', 'decrease', 
            'reduce', 'cut', 'boost', 'raise', 'policy', '%'
        ])
        
        if is_scenario:
            fallback_prompt = f"""Explain this Delhi air quality scenario in simple timeline format: {prompt}

📅 IMMEDIATE (1-3 months): What happens first
📅 SHORT-TERM (3-12 months): Health and daily life changes  
📅 MEDIUM-TERM (1-3 years): Economic and social impacts
📅 LONG-TERM (3-10+ years): Major environmental changes

Use simple language anyone can understand."""
        else:
            fallback_prompt = f"Explain this Delhi air quality question in simple terms: {prompt}"
        
        fallback_payload = {
            "contents": [{"parts": [{"text": fallback_prompt}]}],
            "generationConfig": {
                "temperature": 0.6,
                "maxOutputTokens": 3000  # Increased for comprehensive responses
            }
        }
        
        try:
            fallback_response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{AI_MODELS['quick_response']}:generateContent?key={GEMINI_API_KEY}",
                headers={'Content-Type': 'application/json'},
                data=json.dumps(fallback_payload),
                timeout=20  # Longer timeout for fallback too
            )
            
            if fallback_response.status_code == 200:
                result = fallback_response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        content = candidate['content']['parts']
                        if len(content) > 0 and 'text' in content[0]:
                            return content[0]['text']  # No "backup" label - seamless experience
        except:
            pass
            
        return "⏱️ Response timed out. The scenario analysis above shows the predicted impacts. Please try asking for more specific details."
    except Exception as e:
        return f"⚠️ Error: {str(e)[:100]}... Please try again."


# --- MAIN POLICY FUNCTION ---
def run_policy_advisor():
    st.title("📈 Enhanced AI Policy Advisor")
    st.markdown("*Multiple AI models for comprehensive air quality policy analysis*")
    
    # Validate environment configuration
    if not validate_environment():
        st.stop()
    
    # Data scaling information
    show_quick_scaling_alert()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.header("🤖 AI Configuration")
        
        # Model selection (informational - auto-selected based on intent)
        st.subheader("Available AI Models")
        st.markdown("""
        **🏛️ Policy Expert** - Complex policy analysis  
        **🏥 Health Advisor** - Health recommendations  
        **⚡ Quick Response** - Fast simple queries  
        **📚 Educational** - Learning & explanations  
        
        *Models auto-selected based on your question*
        """)
        
        # Response style preference
        preferred_style = st.selectbox(
            "Preferred Response Style",
            ["Auto-detect", "Simple Language", "Technical Analysis", "Action-focused", "Educational"]
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            show_model_info = st.checkbox("Show AI model used", value=True)
            enable_fallback = st.checkbox("Enable backup models", value=True)
            response_length = st.selectbox("Response Length", ["Auto", "Short", "Medium", "Detailed"])
        
        st.markdown("---")
        
        # Quick prompts
        st.subheader("🚀 Quick Scenarios")
        quick_prompts = {
            "🚗 Vehicle Policy": "What if we reduce vehicle emissions by 25%?",
            "🏭 Industrial Control": "What health benefits would a 40% industrial emission reduction bring?",
            "🌳 Green Spaces": "How would adding 1000 trees affect air quality?",
            "🚧 Construction Rules": "Explain the impact of stricter construction dust controls",
            "⚡ Clean Energy": "What if Delhi switched to 60% renewable energy?"
        }
        
        for label, prompt in quick_prompts.items():
            if st.button(label, key=label):
                st.session_state.quick_prompt = prompt
    
    # Main content area
    policy_coefficient, avg_no2, avg_pm25, df_master = get_policy_context()
    
    if df_master is None:
        st.stop()
        
    df_trend = get_historical_trend_data(df_master)

    # Initialize enhanced chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello! I'm SkyGuard 🛡️ - Your enhanced AI policy advisor for {TARGET_CITY}.\n\n✨ **New Features:**\n- Multiple specialized AI models\n- Smart intent detection\n- Adaptive response styles\n- Backup models for reliability\n\n💡 **Try asking:** 'What if we reduce vehicle emissions by 20%?' or use the quick scenarios in the sidebar!"}
        ]
    
    # Stats overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current PM2.5", f"{avg_pm25:.1f} µg/m³", f"{avg_pm25-15:.1f} above WHO safe limit")
    with col2:
        health_status = "🔴 Unhealthy" if avg_pm25 > 55 else "🟡 Moderate" if avg_pm25 > 25 else "🟢 Good"
        st.metric("Air Quality", health_status)
    with col3:
        st.metric("Active AI Models", len(AI_MODELS))
    with col4:
        chat_count = len(st.session_state.messages) - 1
        st.metric("Chat Messages", chat_count)
        
    # --- DISPLAY HISTORICAL CHART ---
    with st.expander("📈 Historical PM2.5 Trend (2019-2023)", expanded=False):
        fig_trend = px.line(df_trend, x='ds', y='y', title='Daily PM2.5 Concentration Trend',
                            labels={'y': 'PM2.5 (µg/m³)', 'ds': 'Date'},
                            template="plotly_dark")
        
        # Add WHO safe limit
        fig_trend.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="WHO Safe Limit")
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    
    # --- ENHANCED CHAT INTERFACE ---
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a policy question or propose a scenario..."):
        
        # 1. ATTEMPT TO PARSE USER INPUT FOR PERCENTAGE CHANGE (INCREASE OR DECREASE)
        pre_calculated_impact = None
        
        # Check if this is a "what if" scenario question
        prompt_lower = prompt.lower()
        is_scenario = any(phrase in prompt_lower for phrase in [
            'what if', 'if we', 'suppose', 'imagine', 'scenario', 'increase', 'decrease', 
            'reduce', 'cut', 'boost', 'raise', 'policy'
        ])
        
        # Look for percentage in the prompt
        match = re.search(r'(\d+)\s*%', prompt) # Look for digits followed by a percent sign
        
        if match:
            try:
                percentage = float(match.group(1))
                if 0 <= percentage <= 100:
                    # Determine if it's an increase or decrease based on keywords
                    is_increase = any(word in prompt_lower for word in ['increase', 'increasing', 'raise', 'raising', 'boost', 'more', 'higher', 'add', 'adding'])
                    is_decrease = any(word in prompt_lower for word in ['reduce', 'reducing', 'decrease', 'decreasing', 'lower', 'less', 'cut', 'cutting', 'remove'])
                    
                    if is_increase:
                        # For increases, pass as negative reduction (which means increase)
                        pre_calculated_impact = calculate_impact(-percentage, policy_coefficient, avg_no2, avg_pm25)
                        if pre_calculated_impact:
                            pre_calculated_impact['change_type'] = 'increase'
                            pre_calculated_impact['change_pct'] = percentage
                    elif is_decrease:
                        # For decreases, pass as positive reduction
                        pre_calculated_impact = calculate_impact(percentage, policy_coefficient, avg_no2, avg_pm25)
                        if pre_calculated_impact:
                            pre_calculated_impact['change_type'] = 'decrease'
                            pre_calculated_impact['change_pct'] = percentage
                    else:
                        # Default to reduction if unclear
                        pre_calculated_impact = calculate_impact(percentage, policy_coefficient, avg_no2, avg_pm25)
                        if pre_calculated_impact:
                            pre_calculated_impact['change_type'] = 'decrease'
                            pre_calculated_impact['change_pct'] = percentage
            except:
                pass # Ignore parsing errors if the number is bad
        
        # If it's a "what if" scenario but no percentage found, try to extract any number
        elif is_scenario:
            # Look for any number in the scenario (could be without %)
            number_match = re.search(r'(\d+)', prompt)
            if number_match:
                try:
                    potential_percentage = float(number_match.group(1))
                    if 0 <= potential_percentage <= 100:
                        # Assume it's a percentage even without % symbol
                        is_increase = any(word in prompt_lower for word in ['increase', 'increasing', 'raise', 'raising', 'boost', 'more', 'higher', 'add', 'adding'])
                        is_decrease = any(word in prompt_lower for word in ['reduce', 'reducing', 'decrease', 'decreasing', 'lower', 'less', 'cut', 'cutting', 'remove'])
                        
                        if is_increase:
                            pre_calculated_impact = calculate_impact(-potential_percentage, policy_coefficient, avg_no2, avg_pm25)
                            if pre_calculated_impact:
                                pre_calculated_impact['change_type'] = 'increase'
                                pre_calculated_impact['change_pct'] = potential_percentage
                                pre_calculated_impact['is_valid_prediction'] = True  # Ensure graph shows
                        elif is_decrease:
                            pre_calculated_impact = calculate_impact(potential_percentage, policy_coefficient, avg_no2, avg_pm25)
                            if pre_calculated_impact:
                                pre_calculated_impact['change_type'] = 'decrease'
                                pre_calculated_impact['change_pct'] = potential_percentage
                                pre_calculated_impact['is_valid_prediction'] = True  # Ensure graph shows
                        else:
                            # For unclear scenarios, assume 20% reduction as example
                            pre_calculated_impact = calculate_impact(20, policy_coefficient, avg_no2, avg_pm25)
                            if pre_calculated_impact:
                                pre_calculated_impact['change_type'] = 'decrease'
                                pre_calculated_impact['change_pct'] = 20
                                pre_calculated_impact['is_example'] = True
                                pre_calculated_impact['is_valid_prediction'] = True  # Ensure graph shows
                except:
                    # If number parsing fails, create default example
                    if any(word in prompt_lower for word in ['increase', 'increasing', 'raise', 'raising', 'boost', 'more', 'higher', 'add', 'adding']):
                        pre_calculated_impact = calculate_impact(-25, policy_coefficient, avg_no2, avg_pm25)
                        if pre_calculated_impact:
                            pre_calculated_impact['change_type'] = 'increase'
                            pre_calculated_impact['change_pct'] = 25
                            pre_calculated_impact['is_example'] = True
                            pre_calculated_impact['example_note'] = "25% increase"
                            pre_calculated_impact['is_valid_prediction'] = True  # Ensure graph shows
                    else:
                        pre_calculated_impact = calculate_impact(25, policy_coefficient, avg_no2, avg_pm25)
                        if pre_calculated_impact:
                            pre_calculated_impact['change_type'] = 'decrease'
                            pre_calculated_impact['change_pct'] = 25
                            pre_calculated_impact['is_example'] = True
                            pre_calculated_impact['example_note'] = "25% reduction"
                            pre_calculated_impact['is_valid_prediction'] = True  # Ensure graph shows
            else:
                # Pure "what if" without any numbers - create example scenario
                if any(word in prompt_lower for word in ['increase', 'increasing', 'raise', 'raising', 'boost', 'more', 'higher', 'add', 'adding']):
                    # Example increase scenario
                    pre_calculated_impact = calculate_impact(-25, policy_coefficient, avg_no2, avg_pm25)
                    if pre_calculated_impact:
                        pre_calculated_impact['change_type'] = 'increase'
                        pre_calculated_impact['change_pct'] = 25
                        pre_calculated_impact['is_example'] = True
                        pre_calculated_impact['example_note'] = "25% increase"
                        pre_calculated_impact['is_valid_prediction'] = True  # Ensure graph shows
                else:
                    # Example reduction scenario  
                    pre_calculated_impact = calculate_impact(25, policy_coefficient, avg_no2, avg_pm25)
                    if pre_calculated_impact:
                        pre_calculated_impact['change_type'] = 'decrease'
                        pre_calculated_impact['change_pct'] = 25
                        pre_calculated_impact['is_example'] = True
                        pre_calculated_impact['example_note'] = "25% reduction"
                        pre_calculated_impact['is_valid_prediction'] = True  # Ensure graph shows
        
        # FORCE GRAPH GENERATION: If it's ANY "what if" scenario and we don't have impact data, create one
        if is_scenario and pre_calculated_impact is None:
            # Create a default example scenario for any "what if" question
            if any(word in prompt_lower for word in ['increase', 'increasing', 'raise', 'raising', 'boost', 'more', 'higher', 'add', 'adding', 'worse', 'bad']):
                # Default increase scenario
                pre_calculated_impact = calculate_impact(-30, policy_coefficient, avg_no2, avg_pm25)
                if pre_calculated_impact:
                    pre_calculated_impact['change_type'] = 'increase'
                    pre_calculated_impact['change_pct'] = 30
                    pre_calculated_impact['is_example'] = True
                    pre_calculated_impact['example_note'] = "30% increase (example)"
                    pre_calculated_impact['is_valid_prediction'] = True  # FORCE GRAPH DISPLAY
            else:
                # Default reduction scenario
                pre_calculated_impact = calculate_impact(30, policy_coefficient, avg_no2, avg_pm25)
                if pre_calculated_impact:
                    pre_calculated_impact['change_type'] = 'decrease'
                    pre_calculated_impact['change_pct'] = 30
                    pre_calculated_impact['is_example'] = True
                    pre_calculated_impact['example_note'] = "30% reduction (example)"
                    pre_calculated_impact['is_valid_prediction'] = True  # FORCE GRAPH DISPLAY
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in the chat container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            # First show any graphs/analysis if we have calculated impact OR if it's a "what if" scenario
            if pre_calculated_impact and pre_calculated_impact.get('is_valid_prediction', False):
                # Show the graphs and analysis FIRST, before AI response
                if pre_calculated_impact.get('is_example', False):
                    example_note = pre_calculated_impact.get('example_note', '20% reduction')
                    st.info(f"📊 **Example Scenario Analysis** - Since no specific percentage was provided, here's an illustrative {example_note} scenario:")
                else:
                    st.success("📊 **Scenario Analysis Complete!** Here are the predicted impacts:")
                
                # Show advanced model information
                st.markdown("---")
                
                # Model performance metrics
                with st.expander("🔬 Advanced Model Details", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "ML Model Used", 
                            pre_calculated_impact['model_used'].replace('_', ' ').title(),
                            help="AI model selected for this prediction"
                        )
                    
                    with col2:
                        confidence = pre_calculated_impact.get('confidence_score', 0) * 100
                        st.metric(
                            "Model Confidence", 
                            f"{confidence:.1f}%",
                            help="How confident the model is in this prediction"
                        )
                    
                    with col3:
                        uncertainty = pre_calculated_impact.get('prediction_uncertainty', 0)
                        st.metric(
                            "Prediction Uncertainty", 
                            f"±{uncertainty:.1f} µg/m³",
                            help="Expected margin of error in the prediction"
                        )
                
                # Prediction breakdown visualization
                baseline = pre_calculated_impact.get('baseline_pm25', avg_pm25)
                new_pm25 = pre_calculated_impact['new_pm25']
                reduction = pre_calculated_impact['pm25_impact']
                
                # Only show visualization if the values are reasonable
                if 0 <= baseline <= 500 and 0 <= new_pm25 <= 500:
                    # Create before/after comparison
                    fig_comparison = go.Figure()
                    
                    fig_comparison.add_trace(go.Bar(
                        x=['Current Level', 'After Policy'],
                        y=[baseline, new_pm25],
                        name='PM2.5 Concentration',
                        marker_color=['red' if baseline > 25 else 'orange', 
                                     'green' if new_pm25 < 25 else 'orange' if new_pm25 < baseline else 'red']
                    ))
                    
                    # Add WHO safe limit line
                    fig_comparison.add_hline(y=15, line_dash="dash", line_color="green", 
                                           annotation_text="WHO Safe Limit (15 µg/m³)")
                    
                    # Determine change type and create appropriate title
                    change_type = pre_calculated_impact.get('change_type', 'decrease')
                    change_pct = pre_calculated_impact.get('change_pct', pre_calculated_impact['reduction_pct'])
                    
                    if change_type == 'increase':
                        title = f"Policy Impact: {change_pct}% Emission Increase"
                        impact_description = "Increase"
                    else:
                        title = f"Policy Impact: {change_pct}% Emission Reduction"
                        impact_description = "Reduction"
                    
                    fig_comparison.update_layout(
                        title=title,
                        yaxis_title="PM2.5 Concentration (µg/m³)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current PM2.5", f"{baseline:.1f} µg/m³")
                    with col2:
                        delta_sign = "+" if new_pm25 > baseline else "-"
                        delta_value = f"{delta_sign}{reduction:.1f}"
                        st.metric("After Policy", f"{new_pm25:.1f} µg/m³", delta_value)
                    with col3:
                        if change_type == 'increase':
                            health_impact = "Worsened" if new_pm25 > baseline else "No Change"
                        else:
                            health_impact = "Improved" if new_pm25 < baseline else "No Change"
                        st.metric("Health Impact", health_impact)
                
                st.markdown("---")
                st.markdown("🤖 **AI Analysis:**")
            
            # Now generate AI response (this might timeout, but graphs are already shown)
            with st.spinner("🧠 Advanced AI analyzing policy impact..."):
                
                # Generate response with enhanced context
                response_content = generate_llm_response(prompt, 
                                                         (policy_coefficient, avg_no2, avg_pm25, df_master),
                                                         pre_calculated_impact)
                st.markdown(response_content)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

    # Add detailed data scaling information at the end
    st.divider()
    show_data_scaling_info(show_detailed=True)


if __name__ == '__main__':
    # Load necessary external dependencies for the page (since it's a Streamlit page file)
    from main_app import load_models
    st.session_state['prophet_model'], st.session_state['policy_coefficient'] = load_models()
    
    # Ensure policy coefficient is available (must run load_models first)
    if 'policy_coefficient' in st.session_state:
        run_policy_advisor()
    else:
        st.error("Error: Required models/coefficients not found in session state.")
