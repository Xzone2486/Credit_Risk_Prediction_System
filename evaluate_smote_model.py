import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = 'processed_train.csv'
MODEL_FILE = 'xgboost_final_smote.json'
THRESHOLD_FILE = 'best_threshold.txt'
OUTPUT_FILE = 'final_smote_metrics.txt'

def evaluate_saved_model():
    print("Loading data...")
    df = pd.read_csv(TRAIN_FILE)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    # Re-create validation split (must use same random_state)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Model
    print("Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    # Load Threshold
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE, 'r') as f:
            threshold = float(f.read().strip())
    else:
        threshold = 0.5
    print(f"Using Threshold: {threshold}")
    
    # Predict
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    results = f"""
    --- Final SMOTE Model Evaluation ---
    Threshold: {threshold:.4f}
    ROC-AUC:   {auc:.4f}
    Accuracy:  {acc:.4f}
    Precision: {prec:.4f}
    Recall:    {rec:.4f}
    F1 Score:  {f1:.4f}
    """
    print(results)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(results)

if __name__ == "__main__":
    evaluate_saved_model()
