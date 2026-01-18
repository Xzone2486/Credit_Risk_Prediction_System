import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

DATA_DIR = 'home-credit-default-risk'
TRAIN_FILE = os.path.join(DATA_DIR, 'application_train.csv')
OUTPUT_DIR = 'eda_outputs'

def run_eda():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading data for EDA...")
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: {TRAIN_FILE} not found.")
        return

    df = pd.read_csv(TRAIN_FILE)
    
    # 1. Class Imbalance
    print("Generating Class Imbalance Plot...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='TARGET', data=df, palette='viridis')
    plt.title('Class Imbalance: Repayer (0) vs Defaulter (1)')
    plt.xlabel('Target (0: No Default, 1: Default)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_imbalance.png'))
    plt.close()

    # 2. Loan Default Patterns by Age
    print("Generating Age Distribution Plot...")
    df['AGE_YEARS'] = df['DAYS_BIRTH'] / -365
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df.loc[df['TARGET'] == 0, 'AGE_YEARS'], label='Target == 0 (Repayer)', shade=True)
    sns.kdeplot(df.loc[df['TARGET'] == 1, 'AGE_YEARS'], label='Target == 1 (Defaulter)', shade=True)
    plt.title('Distribution of Age by Target')
    plt.xlabel('Age (Years)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'age_distribution_by_target.png'))
    plt.close()

    # 3. Default Rate by Income Type
    print("Generating Income Type Default Rate Plot...")
    plt.figure(figsize=(12, 6))
    # Calculate default rate per category
    income_defaults = df.groupby('NAME_INCOME_TYPE')['TARGET'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(x='TARGET', y='NAME_INCOME_TYPE', data=income_defaults, palette='Reds_d')
    plt.title('Default Rate by Income Type')
    plt.xlabel('Default Rate (Proportion)')
    plt.ylabel('Income Type')
    plt.savefig(os.path.join(OUTPUT_DIR, 'default_rate_by_income_type.png'))
    plt.close()
    
    # 4. Boxplots for Numeric Features (Income & Credit)
    print("Generating Boxplots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Log scale for Income to handle skewness
    sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df, ax=axes[0], showfliers=False) 
    axes[0].set_title('Income Distribution by Target (Outliers Removed for View)')
    axes[0].set_ylabel('Income Total')
    
    sns.boxplot(x='TARGET', y='AMT_CREDIT', data=df, ax=axes[1], showfliers=False)
    axes[1].set_title('Credit Amount Distribution by Target (Outliers Removed for View)')
    axes[1].set_ylabel('Credit Amount')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'numeric_boxplots.png'))
    plt.close()

    # 5. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    # Select correlations with TARGET
    correlations = df.corr(numeric_only=True)['TARGET'].sort_values()
    
    # Top 10 Positive and Top 10 Negative correlations
    top_pos_corr = correlations.tail(11).index.tolist() # Includes TARGET itself
    top_neg_corr = correlations.head(10).index.tolist()
    top_corr_features = top_pos_corr + top_neg_corr
    
    # Remove duplicates if any
    top_corr_features = list(set(top_corr_features))
    
    corr_matrix = df[top_corr_features].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap (Top Correlated Features with Target)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close()

    print(f"EDA Complete. Visualizations saved in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    run_eda()
