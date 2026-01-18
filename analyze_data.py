import pandas as pd
import os
import sys

# Set options for better display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = os.path.join(DATA_DIR, 'application_train.csv')
OUTPUT_FILE = 'analysis_results.txt'

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def analyze_data():
    f = open(OUTPUT_FILE, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    try:
        print("Loading data...")
        if not os.path.exists(TRAIN_FILE):
            print(f"Error: File not found at {TRAIN_FILE}")
            return

        df = pd.read_csv(TRAIN_FILE)
        
        print("\n--- Data Structure ---")
        print(f"Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\n--- Target Variable Distribution ---")
        if 'TARGET' in df.columns:
            target_counts = df['TARGET'].value_counts(normalize=True)
            print("Normalized Counts:")
            print(target_counts)
            print("\nRaw Counts:")
            print(df['TARGET'].value_counts())
        else:
            print("TARGET column not found!")

        print("\n--- Data Quality Check ---")
        print(f"Duplicates: {df.duplicated().sum()}")
        
        print("\nMissing Values (Top 20 columns):")
        missing_values = df.isnull().sum().sort_values(ascending=False)
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_df = pd.concat([missing_values, missing_percent], axis=1, keys=['Total', 'Percent'])
        print(missing_df.head(20))

        print("\n--- Preprocessing Identifications ---")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical Columns ({len(categorical_cols)}):")
        print(categorical_cols)
        
        print("\nNumerical Columns with Missing Values:")
        num_missing = df.select_dtypes(exclude=['object']).isnull().sum()
        print(num_missing[num_missing > 0].sort_values(ascending=False).head(10))

    finally:
        sys.stdout = original_stdout
        f.close()

if __name__ == "__main__":
    analyze_data()
