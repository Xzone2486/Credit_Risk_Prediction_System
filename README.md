# Credit Risk Prediction System

![Banner](eda_outputs/class_imbalance.png)

## 📌 Overview

The **Credit Risk Prediction System** is a machine learning solution designed to predict the likelihood of loan default. Leveraging the **Home Credit Default Risk** dataset, this project addresses the challenge of severe class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) and utilizes **XGBoost** for robust classification.

The system is deployed as an interactive web application using **Streamlit**, allowing loan officers to input applicant details and receive real-time risk assessments.

## 🚀 Features

- **Machine Learning Pipeline:** End-to-end processing from raw CSVs to a deployed model.
- **Imbalance Handling:** uses SMOTE to improve the detection of defaulters (Recall).
- **Interactive Dashboard:** A user-friendly interface for manual predictions.
- **Performance:** Achieved **ROC-AUC of 0.7458** and optimized Recall.

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

## ⚙️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Xzone2486/Credit_Risk_Prediction_System.git
    cd Credit_Risk_Prediction_System
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```
2.  **Access the App:** Open your browser to `http://localhost:8501`.
3.  **Predict:** Enter applicant details (Income, Age, External Scores) in the sidebar to visualize the default probability.

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `xgboost_optimized.json`: Trained XGBoost model file.
- `PROJECT_SUMMARY.md`: Detailed documentation of methodology, EDA, and results.
- `eda_outputs/`: Directory containing generated visualization images.
- `imbalance_handling.py`: Script for training and SMOTE application.

## 📝 Documentation

For a deep dive into the problem statement, data analysis, and algorithm comparison, please refer to the [Project Summary](PROJECT_SUMMARY.md).

## 📊 Results Summary

| Metric       | Value  |
| :----------- | :----- |
| **ROC-AUC**  | 0.7458 |
| **Accuracy** | 83.6%  |
| **Recall**   | 41.4%  |

---

_Created by [Xzone2486](https://github.com/Xzone2486)_
