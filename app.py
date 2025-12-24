import streamlit as st
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .risk-level {
        font-size: 1.5rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists('diabetes_model_LightGBM.pkl'):
        return None, "not_found"
    
    try:
        import joblib
        model = joblib.load('diabetes_model_LightGBM.pkl')
        return model, "success"
    except Exception as e:
        return None, f"error: {str(e)}"

def create_features(data):
    """Apply feature engineering"""
    df = data.copy()
    
    # Health risk features
    df['bmi_age'] = df['bmi'] * df['age'] / 100
    df['waist_bmi'] = df['waist_to_hip_ratio'] * df['bmi']
    
    # Blood pressure
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # Cholesterol ratios
    df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1)
    
    # Lifestyle score
    df['health_score'] = (
        df['diet_score'] * 0.3 +
        np.log1p(df['physical_activity_minutes_per_week']) * 0.3 +
        (8 - df['sleep_hours_per_day']).clip(0, 4) * 0.2 +
        (6 - df['screen_time_hours_per_day']).clip(0, 4) * 0.2
    )
    
    # Risk flags
    df['is_senior'] = (df['age'] >= 60).astype(int)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    
    return df

def get_risk_level(probability):
    """Categorize risk level"""
    if probability < 0.3:
        return "Low Risk", "🟢", "Your risk of diabetes is relatively low."
    elif probability < 0.5:
        return "Moderate Risk", "🟡", "You have moderate risk. Consider lifestyle modifications."
    elif probability < 0.7:
        return "High Risk", "🟠", "You have high risk. Consult with a healthcare provider."
    else:
        return "Very High Risk", "🔴", "You have very high risk. Please seek medical attention soon."

def main():
    # Header
    st.title("🏥 Diabetes Risk Prediction System")
    st.markdown("### Predict your diabetes risk based on health metrics")
    
    # Load model
    model, status = load_model()
    
    if status == "not_found":
        st.error("❌ Model file not found! Please add 'diabetes_model_LightGBM.pkl' to the app directory.")
        st.stop()
    elif status != "success":
        st.error(f"❌ Error loading model: {status}")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a machine learning model 
        trained on 700,000+ samples to predict 
        diabetes risk.
        
        **Model:** LightGBM  
        **AUC Score:** ~0.71
        """)
        st.markdown("---")
        st.markdown("⚠️ **Disclaimer:** For informational purposes only. Consult healthcare professionals for medical advice.")
    
    # Main content
    st.header("Enter Your Health Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Asian", "Other"])
        education_level = st.selectbox("Education Level", 
                                      ["Less than high school", "High school", 
                                       "Some college", "College graduate", "Graduate degree"])
        employment_status = st.selectbox("Employment Status", 
                                        ["Employed", "Unemployed", "Retired", "Student", "Other"])
        income_level = st.selectbox("Income Level", ["Low", "Middle", "High"])
        
        st.subheader("Physical Measurements")
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        waist_to_hip_ratio = st.number_input("Waist-to-Hip Ratio", min_value=0.5, max_value=1.5, value=0.85, step=0.01)
    
    with col2:
        st.subheader("Blood Pressure & Vitals")
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
        
        st.subheader("Blood Tests")
        cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
        ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=300, value=120)
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=500, value=150)
    
    with col3:
        st.subheader("Lifestyle")
        diet_score = st.slider("Diet Quality (0-10)", 0, 10, 5)
        physical_activity_minutes_per_week = st.number_input("Physical Activity (min/week)", 0, 1000, 150)
        sleep_hours_per_day = st.number_input("Sleep (hours/day)", 3.0, 12.0, 7.0, 0.5)
        screen_time_hours_per_day = st.number_input("Screen Time (hours/day)", 0.0, 16.0, 4.0, 0.5)
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol_consumption_per_week = st.number_input("Alcohol Drinks per Week", 0, 50, 0)
        
        st.subheader("Medical History")
        family_history_diabetes = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        hypertension_history = st.selectbox("History of Hypertension", ["No", "Yes"])
        cardiovascular_history = st.selectbox("Cardiovascular History", ["No", "Yes"])
    
    # Predict button
    if st.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True):
        try:
            # Convert Yes/No to 1/0 for numerical columns
            family_history_diabetes_num = 1 if family_history_diabetes == "Yes" else 0
            hypertension_history_num = 1 if hypertension_history == "Yes" else 0
            cardiovascular_history_num = 1 if cardiovascular_history == "Yes" else 0
            
            # Create input dataframe with exact column order
            input_data = pd.DataFrame({
                'age': [age],
                'alcohol_consumption_per_week': [alcohol_consumption_per_week],
                'physical_activity_minutes_per_week': [physical_activity_minutes_per_week],
                'diet_score': [diet_score],
                'sleep_hours_per_day': [sleep_hours_per_day],
                'screen_time_hours_per_day': [screen_time_hours_per_day],
                'bmi': [bmi],
                'waist_to_hip_ratio': [waist_to_hip_ratio],
                'systolic_bp': [systolic_bp],
                'diastolic_bp': [diastolic_bp],
                'heart_rate': [heart_rate],
                'cholesterol_total': [cholesterol_total],
                'hdl_cholesterol': [hdl_cholesterol],
                'ldl_cholesterol': [ldl_cholesterol],
                'triglycerides': [triglycerides],
                'family_history_diabetes': [family_history_diabetes_num],  # Numeric!
                'hypertension_history': [hypertension_history_num],  # Numeric!
                'cardiovascular_history': [cardiovascular_history_num],  # Numeric!
                'gender': [gender],  # Categorical
                'ethnicity': [ethnicity],  # Categorical
                'education_level': [education_level],  # Categorical
                'income_level': [income_level],  # Categorical
                'smoking_status': [smoking_status],  # Categorical
                'employment_status': [employment_status]  # Categorical
            })
            
            # Apply feature engineering (adds bmi_age, waist_bmi, pulse_pressure, etc.)
            input_data = create_features(input_data)
            
            # Make prediction
            probability = model.predict_proba(input_data)[0, 1]
            
            # Get risk level
            risk_level, emoji, description = get_risk_level(probability)
            
            # Display result
            st.markdown("---")
            st.markdown(f"""
                <div class="prediction-box">
                    <div class="risk-level">{emoji} {risk_level}</div>
                    <div class="prediction-value">{probability:.1%}</div>
                    <p>Probability of Diabetes</p>
                    <p style="margin-top: 1rem;">{description}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Risk factors analysis
            st.subheader("📊 Risk Factors Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Risk Factors Identified:**")
                risk_factors = []
                if bmi >= 30:
                    risk_factors.append("• BMI indicates obesity")
                if family_history_diabetes == "Yes":
                    risk_factors.append("• Family history of diabetes")
                if age >= 45:
                    risk_factors.append("• Age over 45")
                if physical_activity_minutes_per_week < 150:
                    risk_factors.append("• Insufficient physical activity")
                if hypertension_history == "Yes":
                    risk_factors.append("• History of hypertension")
                if cardiovascular_history == "Yes":
                    risk_factors.append("• Cardiovascular history")
                if smoking_status == "Current":
                    risk_factors.append("• Current smoker")
                if alcohol_consumption_per_week > 14:
                    risk_factors.append("• High alcohol consumption")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.success("✅ No major risk factors identified")
            
            with col2:
                st.markdown("**Recommendations:**")
                recommendations = []
                if bmi >= 25:
                    recommendations.append("• Consider weight management program")
                if physical_activity_minutes_per_week < 150:
                    recommendations.append("• Increase physical activity to 150+ min/week")
                if diet_score < 7:
                    recommendations.append("• Improve diet quality (more vegetables, whole grains)")
                if sleep_hours_per_day < 7:
                    recommendations.append("• Aim for 7-9 hours of sleep per night")
                if smoking_status == "Current":
                    recommendations.append("• Consider smoking cessation program")
                if alcohol_consumption_per_week > 7:
                    recommendations.append("• Reduce alcohol consumption")
                recommendations.append("• Schedule regular health checkups")
                recommendations.append("• Consult with healthcare provider for personalized advice")
                
                for rec in recommendations:
                    st.markdown(rec)
            
            # Additional info
            st.info("💡 **Note:** This prediction is based on a machine learning model and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            with st.expander("See error details"):
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()