import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt

# Options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = 'processed_train.csv'
OPTIMIZED_MODEL_FILE = 'xgboost_optimized.json'
FINAL_MODEL_FILE = 'xgboost_final_smote.json'

def handle_imbalance():
    print("Loading processed data...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    df = pd.read_csv(TRAIN_FILE)
    
    if 'TARGET' not in df.columns:
        print("Error: TARGET column missing.")
        return

    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Original Train Shape: {X_train.shape}, Count 1s: {sum(y_train)}")

    # 1. SMOTE Oversampling
    print("\n--- Applying SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"SMOTE Train Shape: {X_train_smote.shape}, Count 1s: {sum(y_train_smote)}")

    # 2. Train XGBoost on SMOTE data
    # Using optimized params found previously (approximate)
    print("\n--- Training XGBoost on SMOTE Data ---")
    xgb_smote = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=1, # No class weight needed as data is balanced
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )
    xgb_smote.fit(X_train_smote, y_train_smote)
    
    # Evaluate
    y_prob_smote = xgb_smote.predict_proba(X_val)[:, 1]
    roc_smote = roc_auc_score(y_val, y_prob_smote)
    print(f"SMOTE Model ROC-AUC: {roc_smote:.4f}")

    # Compare with Class Weighted (re-loading if needed, or just assuming baseline ~0.764)
    print("Baseline (Class Weighted) ROC-AUC: ~0.764")
    
    # 3. Threshold Tuning
    print("\n--- Threshold Tuning ---")
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob_smote)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Best Threshold (Maximize F1): {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    # Evaluate at Best Threshold
    y_pred_opt = (y_prob_smote >= best_threshold).astype(int)
    
    final_acc = accuracy_score(y_val, y_pred_opt)
    final_prec = precision_score(y_val, y_pred_opt)
    final_rec = recall_score(y_val, y_pred_opt)
    
    print("\n--- Final Metrics (SMOTE + Threshold Tuning) ---")
    print(f"Accuracy:  {final_acc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall:    {final_rec:.4f}")
    print(f"ROC-AUC:   {roc_smote:.4f}")
    
    # Save Model
    xgb_smote.save_model(FINAL_MODEL_FILE)
    print(f"\nSaved final model to '{FINAL_MODEL_FILE}'")
    
    # Save Threshold info
    with open('best_threshold.txt', 'w') as f:
        f.write(str(best_threshold))

if __name__ == "__main__":
    handle_imbalance()
