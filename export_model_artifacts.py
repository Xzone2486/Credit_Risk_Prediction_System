import pandas as pd
import json
import os

TRAIN_FILE = 'processed_train.csv'
ARTIFACTS_FILE = 'model_artifacts.json'

def export_artifacts():
    print(f"Loading {TRAIN_FILE}...")
    if not os.path.exists(TRAIN_FILE):
        print("Error: Train file not found!")
        return

    # Load data
    df = pd.read_csv(TRAIN_FILE)
    
    # Exclude non-feature columns
    features = [c for c in df.columns if c not in ['TARGET', 'SK_ID_CURR']]
    
    print(f"Calculating defaults for {len(features)} features...")
    # Calculate median for numeric columns
    defaults = df[features].median(numeric_only=True).to_dict()
    
    # Save to JSON
    artifacts = {
        'columns': features,
        'defaults': defaults
    }
    
    with open(ARTIFACTS_FILE, 'w') as f:
        json.dump(artifacts, f, indent=4)
        
    print(f"Successfully saved artifacts to {ARTIFACTS_FILE}")

if __name__ == "__main__":
    export_artifacts()
