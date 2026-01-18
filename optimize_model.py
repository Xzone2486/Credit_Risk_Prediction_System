import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib
import os
import time

# Options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = 'processed_train.csv'
MODEL_FILE = 'xgboost_optimized.json'

def optimize_model():
    print("Loading processed data for optimization...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    df = pd.read_csv(TRAIN_FILE)
    
    if 'TARGET' not in df.columns:
        print("Error: TARGET column missing.")
        return

    X = df.drop(columns=['TARGET', 'SK_ID_CURR']) # Drop ID and Target
    y = df['TARGET']
    
    print(f"Data Shape: {X.shape}")

    # XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )

    # Parameter Grid
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [10, 11, 12] # Handling imbalance (approx ratio of neg/pos)
    }

    # Randomized Search
    # Using 3-fold CV and 10 iterations to keep runtime reasonable (~5-10 mins)
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        verbose=3,
        random_state=42,
        n_jobs=-1
    )

    print("\nStarting RandomizedSearchCV...")
    start_time = time.time()
    
    random_search.fit(X, y)
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimization Complete in {elapsed_time:.2f}s")
    
    print("\n--- Best Results ---")
    print(f"Best ROC-AUC Score: {random_search.best_score_:.4f}")
    print("Best Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    # Save Best Model
    best_model = random_search.best_estimator_
    best_model.save_model(MODEL_FILE)
    print(f"\nSaved best model to '{MODEL_FILE}'")

if __name__ == "__main__":
    optimize_model()
