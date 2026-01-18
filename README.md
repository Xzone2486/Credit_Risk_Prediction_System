# 🏦 Credit Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge)

> **A sophisticated machine learning solution for predicting loan default risks with high-recall optimization.**

---

## � Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Visual Insights](#-visual-insights)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Project Architecture](#-project-architecture)
- [Model Performance](#-model-performance)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Financial institutions face a critical challenge in distinguishing between reliable borrowers and those likely to default. This project builds a **Credit Risk Prediction System** leveraging the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset.

By addressing severe class imbalance through **SMOTE** (Synthetic Minority Over-sampling Technique) and utilizing **XGBoost Classifiers**, this system prioritizes the identification of high-risk applicants (Recall) to minimize financial losses. The entire pipeline is wrapped in an interactive **Streamlit Dashboard** for real-time inference.

---

## 🌟 Key Features

- **⚡ Real-Time Inference:** Instant risk scoring based on applicant data inputs.
- **⚖️ Imbalance Handling:** Advanced synthetic oversampling (SMOTE) to detecting rare default events effective.
- **📊 Interactive Dashboard:** User-friendly interface with gauges, risk meters, and dynamic visualizations.
- **🧠 Optimized Model:** tuned XGBoost model focusing on maximizing **ROC-AUC** and **Recall**.
- **📈 Insightful EDA:** Comprehensive visualizations identifying key risk drivers like `EXT_SOURCE` scores and `DEBT_TO_INCOME` ratios.

---

## 🎨 Visual Insights

### Data Analysis & Distributions

|                Class Imbalance                |               Correlation Heatmap               |
| :-------------------------------------------: | :---------------------------------------------: |
| ![Imbalance](eda_outputs/class_imbalance.png) | ![Heatmap](eda_outputs/correlation_heatmap.png) |
|     _Severe imbalance handled via SMOTE_      |      _Key feature correlations identified_      |

### Performance Evaluation

|       ROC Curve Comparison       |
| :------------------------------: |
| ![ROC](model_roc_comparison.png) |
| _Model achieves 0.7458 ROC-AUC_  |

---

## 🛠 Tech Stack

| Category               | Technologies             |
| :--------------------- | :----------------------- |
| **Core**               | Python 3.8+              |
| **Data Manipulation**  | Pandas, NumPy            |
| **Machine Learning**   | XGBoost, Scikit-Learn    |
| **Imbalance Handling** | Imbalanced-Learn (SMOTE) |
| **Visualization**      | Matplotlib, Seaborn      |
| **Deployment**         | Streamlit                |

---

## 🚀 Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Xzone2486/Credit_Risk_Prediction_System.git
    cd Credit_Risk_Prediction_System
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 🏗 Project Architecture

```plaintext
Credit_Risk_Prediction_System/
│
├── 📂 eda_outputs/          # Generated plots & visualizations
├── 📂 home-credit-default-risk/ # Raw Dataset (Ignored in Git)
│
├── 📜 app.py                # 🚀 Main Streamlit Dashboard Application
├── 📜 dashboard_utils.py    # 🛠 Helper functions for the dashboard
├── 📜 imbalance_handling.py # 🧠 Script for SMOTE & Model Training
├── 📜 evaluate_smote_model.py # � Model Evaluation Script
├── 📜 preprocess_data.py    # 🧹 Data Cleaning & Feature Engineering
│
├── 📄 xgboost_optimized.json# 🤖 Trained Model File
├── 📄 PROJECT_SUMMARY.md    # 📑 Detailed Project Documentation
├── 📄 requirements.txt      # 📦 Project Dependencies
└── 📄 README.md             # 📖 This file
```

---

## 📊 Model Performance

Our model was specifically optimized to improve **Recall** (capturing more defaulters) rather than just accuracy.

| Metric        | Score    | Note                                           |
| :------------ | :------- | :--------------------------------------------- |
| **ROC-AUC**   | `0.7458` | Strong ability to distinguish classes          |
| **Accuracy**  | `83.59%` | Overall correctness                            |
| **Recall**    | `41.43%` | % of actual defaulters correctly identified    |
| **Precision** | `22.25%` | Trade-off for higher recall in imbalanced data |

> **Note:** We utilized a custom decision threshold (`~0.17`) instead of the default `0.5` to maximize the F1-Score and sensitive risk detection.

---

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

### 📬 Contact

Created by [Xzone2486](https://github.com/Xzone2486) - Feel free to reach out!
