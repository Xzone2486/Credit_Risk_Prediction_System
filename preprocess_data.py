import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = os.path.join(DATA_DIR, 'application_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'application_test.csv')
OUTPUT_TRAIN = 'processed_train.csv'
OUTPUT_TEST = 'processed_test.csv'

def preprocess_data():
    print("Loading data...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return
    
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE) if os.path.exists(TEST_FILE) else None
    
    print(f"Train Imbalance: {df_train['TARGET'].value_counts(normalize=True).to_dict()}")

    # combine for consistent preprocessing
    len_train = len(df_train)
    if df_test is not None:
        df_all = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
    else:
        df_all = df_train.copy()

    print("\n--- 1. Missing Value Imputation ---")
    # Numerical: Median
    num_cols = df_all.select_dtypes(include=['number']).columns.tolist()
    if 'TARGET' in num_cols: num_cols.remove('TARGET')
    if 'SK_ID_CURR' in num_cols: num_cols.remove('SK_ID_CURR')
    
    imputer_num = SimpleImputer(strategy='median')
    df_all[num_cols] = imputer_num.fit_transform(df_all[num_cols])
    
    # Categorical: Mode or Unknown
    cat_cols = df_all.select_dtypes(include=['object']).columns.tolist()
    imputer_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
    df_all[cat_cols] = imputer_cat.fit_transform(df_all[cat_cols])
    print("Imputation Complete.")

    print("\n--- 2. Feature Engineering ---")
    # Domain Knowledge Features
    df_all['DEBT_TO_INCOME_RATIO'] = df_all['AMT_ANNUITY'] / df_all['AMT_INCOME_TOTAL']
    df_all['CREDIT_TO_INCOME_RATIO'] = df_all['AMT_CREDIT'] / df_all['AMT_INCOME_TOTAL']
    df_all['ANNUITY_TO_CREDIT_RATIO'] = df_all['AMT_ANNUITY'] / df_all['AMT_CREDIT']
    
    # Age Group (DAYS_BIRTH is negative days)
    df_all['AGE_YEARS'] = df_all['DAYS_BIRTH'] / -365
    df_all['AGE_GROUP'] = pd.cut(df_all['AGE_YEARS'], bins=[20, 30, 40, 50, 60, 70, 100], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70+'])
    
    # Update categorical columns list after new features
    cat_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("Feature Engineering Complete.")

    print("\n--- 3. Categorical Encoding (One-Hot) ---")
    df_all = pd.get_dummies(df_all, columns=cat_cols, dummy_na=False) # dummy_na handled by imputation
    print(f"Shape after encoding: {df_all.shape}")

    print("\n--- 4. Outlier Handling (Clipping) ---")
    # Clip numerical columns to 1st and 99th percentiles to avoid extreme outliers affecting scaling
    # We use 1/99 instead of IQR to be less aggressive given financial data often has valid high values
    for col in num_cols:
        lower = df_all[col].quantile(0.01)
        upper = df_all[col].quantile(0.99)
        df_all[col] = df_all[col].clip(lower, upper)
    print("Outlier Clipping Complete.")

    print("\n--- 5. Scaling ---")
    # Update num_cols to include new numerical features (excluding generated bools from OHE)
    # Actually, scaling binary OHE features is debated, usually we scale continous vars.
    # Let's re-select float/int columns that are not TARGET or ID
    scale_cols = [c for c in df_all.columns if c not in ['TARGET', 'SK_ID_CURR'] and df_all[c].nunique() > 2]
    
    scaler = MinMaxScaler() # MinMax usually better for Neural Nets, Standard for Linear. Financial data often skewed so MinMax or Log is good. Using MinMax for uniformity.
    df_all[scale_cols] = scaler.fit_transform(df_all[scale_cols])
    print("Scaling Complete.")

    print("\n--- 6. Saving Data ---")
    train_processed = df_all.iloc[:len_train]
    test_processed = df_all.iloc[len_train:]

    print(f"Train Shape: {train_processed.shape}")
    print(f"Test Shape: {test_processed.shape}")

    train_processed.to_csv(OUTPUT_TRAIN, index=False)
    if df_test is not None:
        test_processed.drop(columns=['TARGET'], errors='ignore').to_csv(OUTPUT_TEST, index=False)
    
    print(f"Saved to {OUTPUT_TRAIN} and {OUTPUT_TEST}")

if __name__ == "__main__":
    preprocess_data()
