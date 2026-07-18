
# Intelligent Customer Behavior & Marketing Strategy Dashboard

An end-to-end data pipeline, modeling suite, and interactive Streamlit application designed around the Online Retail dataset. This system covers data preprocessing, dimensionality reduction, predictive modeling (classification and regression), and an operational reinforcement learning policy engine.

## 📊 Features

* **Interactive Profile Diagnostics:** Real-time KPI analysis and predictive modeling insights for target customer segments.
* **Dimensionality Reduction Mapping:** Interactive visualization of customer distributions using PCA and LDA coordinate mapping.
* **Risk & Feature Audit Focus:** Confidence-based risk alert logic paired with micro-level feature weight attribution plots.
* **RL Strategy Operations Simulator:** Evaluates single-profile marketing strategies and runs multi-quarter decision journey tracking simulations using a Trained Deep Q-Network (DQN).

---

## 📂 Project Structure

```text
project-root/
│
├── data/                  # Raw + processed CSV datasets
├── models/                # Saved serialization artifacts (.pkl / .pt)
├── notebooks/             # Step-by-step experiment pipelines
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_lda.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_regression.ipynb
│   └── 05_qlearning_dqn.ipynb
├── src/                   # Reusable python helper modules
├── app.py                 # Core Streamlit GUI web interface
├── category_map.json      # Keyword-to-category mapping data
├── requirements.txt       # Project package dependencies
└── README.md              # Setup and execution instructions
```
