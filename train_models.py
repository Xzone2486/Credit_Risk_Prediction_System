import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve, classification_report
import xgboost as xgb
import os
import time

# Options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = 'processed_train.csv'
RESULTS_FILE = 'model_results.csv'

def train_and_evaluate():
    print("Loading processed data...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    df = pd.read_csv(TRAIN_FILE)
    
    # Separate Features and Target
    if 'TARGET' not in df.columns:
        print("Error: TARGET column missing.")
        return

    X = df.drop(columns=['TARGET', 'SK_ID_CURR']) # Drop ID and Target
    y = df['TARGET']
    
    # Split Data (80% Train, 20% Valid) specifically for model comparison
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train Shape: {X_train.shape}, Validation Shape: {X_val.shape}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', C=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, scale_pos_weight=11, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1) 
        # scale_pos_weight ~ sum(negative) / sum(positive) ~ 282k/24k ~ 11
    }

    results = []
    
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        print(f"Training Time: {train_time:.2f}s")
        
        # Predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_prob)
        
        print(f"ROC-AUC: {roc_auc:.4f}, Recall: {recall:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC_AUC": roc_auc,
            "Training_Time": train_time
        })
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        
        # Feature Importance (for Tree models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10] # Top 10
            print(f"Top 5 Features for {name}:")
            for i in range(5):
                print(f"  {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

    # Plot Settings
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_roc_comparison.png')
    print("\nSaved ROC Comparison to 'model_roc_comparison.png'")

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print("\n--- Final Model Comparison ---")
    print(results_df)

if __name__ == "__main__":
    train_and_evaluate()
