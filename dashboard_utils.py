import pandas as pd
import numpy as np
import xgboost as xgb
import os
import streamlit as st

import json

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = 'processed_train.csv'
MODEL_FILE = 'xgboost_optimized.json'
ARTIFACTS_FILE = 'model_artifacts.json'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

@st.cache_data
def get_column_defaults():
    # Load pre-calculated artifacts (columns and defaults)
    if not os.path.exists(ARTIFACTS_FILE):
        return None, None
    
    with open(ARTIFACTS_FILE, 'r') as f:
        data = json.load(f)
        
    return data['columns'], data['defaults']

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
