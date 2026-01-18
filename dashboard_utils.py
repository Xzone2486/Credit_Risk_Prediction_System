import pandas as pd
import numpy as np
import xgboost as xgb
import os
import streamlit as st

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = 'processed_train.csv'
MODEL_FILE = 'xgboost_optimized.json' # Using the optimized (weighted) model for better default detection sensitivity

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

@st.cache_data
def get_column_defaults():
    # Load training data to get feature names and median/mode defaults
    if not os.path.exists(TRAIN_FILE):
        return None, None
    
    # Read only first few rows to get columns if file is huge, but we need medians
    # Reading full file is safer for accurate medians.
    df = pd.read_csv(TRAIN_FILE)
    columns = [c for c in df.columns if c not in ['TARGET', 'SK_ID_CURR']]
    
    # Calculate defaults
    defaults = df[columns].median(numeric_only=True).to_dict()
    # For anything missing (categorical OHE columns might be 0/1, median works)
    
    return columns, defaults

def prepare_input_data(user_inputs, all_columns, defaults):
    # Create DataFrame with all columns initialized to defaults
    input_data = pd.DataFrame([defaults])
    
    # Update with user inputs
    for col, val in user_inputs.items():
        if col in input_data.columns:
            input_data[col] = val
            
    # Ensure correct column order
    input_data = input_data[all_columns]
    
    return input_data
