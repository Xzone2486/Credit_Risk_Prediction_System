import streamlit as st
import pandas as pd
import numpy as np
import dashboard_utils as utils

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

# Title and Desc
st.title("🏦 Home Credit Default Risk Prediction")
st.markdown("""
This dashboard predicts the probability of a client defaulting on a loan.
Enter the client's information in the sidebar to get a real-time risk assessment.
""")

# Load Resources
model = utils.load_model()
cols, defaults = utils.get_column_defaults()

if model is None or cols is None:
    st.error("Error: Model or Data file not found. Please ensure training is complete.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Client Information")

def user_input_features():
    inputs = {}
    
    st.sidebar.subheader("1. External Sources (Normalized)")
    inputs['EXT_SOURCE_3'] = st.sidebar.slider("External Source 3", 0.0, 1.0, 0.5)
    inputs['EXT_SOURCE_2'] = st.sidebar.slider("External Source 2", 0.0, 1.0, 0.5)
    inputs['EXT_SOURCE_1'] = st.sidebar.slider("External Source 1", 0.0, 1.0, 0.5)

    st.sidebar.subheader("2. Demographics")
    age = st.sidebar.number_input("Age (Years)", 20, 70, 35)
    inputs['DAYS_BIRTH'] = -1 * age * 365 # Feature engineering expects negative days
    inputs['AGE_YEARS'] = age # We added this feature during engineering
    
    # For OHE columns, we need to map selection to specific columns
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    inputs['CODE_GENDER_M'] = 1 if gender == "Male" else 0
    inputs['CODE_GENDER_F'] = 1 if gender == "Female" else 0
    
    education = st.sidebar.selectbox("Education", 
                                     ["Secondary / secondary special", "Higher education", "Imperfect higher", "Lower secondary"])
    # Map education to OHE columns (Simplified mapping, might need adjustment based on exact column names)
    # The processed data has columns like 'NAME_EDUCATION_TYPE_Higher education'
    edu_col = f"NAME_EDUCATION_TYPE_{education}"
    if edu_col in cols:
        inputs[edu_col] = 1

    st.sidebar.subheader("3. Financials")
    inputs['AMT_INCOME_TOTAL'] = st.sidebar.number_input("Annual Income", 25000, 10000000, 150000)
    inputs['AMT_CREDIT'] = st.sidebar.number_input("Credit Amount", 45000, 5000000, 500000)
    inputs['AMT_ANNUITY'] = st.sidebar.number_input("Loan Annuity", 1000, 250000, 25000)
    
    # Recalculate Ratios
    inputs['DEBT_TO_INCOME_RATIO'] = inputs['AMT_ANNUITY'] / inputs['AMT_INCOME_TOTAL']
    inputs['CREDIT_TO_INCOME_RATIO'] = inputs['AMT_CREDIT'] / inputs['AMT_INCOME_TOTAL']
    
    return inputs

user_inputs = user_input_features()

# --- Prediction Logic ---
if st.button("Predict Risk"):
    # Prepare full feature vector
    input_df = utils.prepare_input_data(user_inputs, cols, defaults)
    
    # Predict
    probability = model.predict_proba(input_df)[0][1]
    
    # Display Results
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Default Probability")
        st.metric(label="Risk Score", value=f"{probability:.2%}")
        
    with res_col2:
        st.subheader("Risk Assessment")
        # Using optimal threshold found earlier ~0.17 for high sensitivity, 
        # or stick to 0.5 if using weighted model which is calibrated differently.
        # Since we use 'xgboost_optimized.json' which was class-weighted, 0.5 is standard.
        # But 'xgboost_final_smote.json' used 0.17. Let's use 0.5 for weighted model.
        threshold = 0.5 
        
        if probability > threshold:
            st.error(f"⚠️ HIGH RISK (Probability > {threshold})")
            st.write("This applicant is flagged as likely to default.")
        else:
            st.success("✅ LOW RISK")
            st.write("This applicant is considered safe.")
            
    # Visualize Feature Contribution (Simple bar for input values vs median)
    st.subheader("Key Factors Visualization")
    comparison = pd.DataFrame({
        'Feature': ['Income', 'Credit', 'Source 3'],
        'Applicant': [user_inputs['AMT_INCOME_TOTAL'], user_inputs['AMT_CREDIT'], user_inputs['EXT_SOURCE_3']],
        'Average': [defaults['AMT_INCOME_TOTAL'], defaults['AMT_CREDIT'], defaults['EXT_SOURCE_3']]
    })
    st.bar_chart(comparison.set_index('Feature'))

else:
    st.info("Adjust the details in the sidebar and click 'Predict Risk' to see the result.")

