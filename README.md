# Bank Loan Risk Engine

A machine learning pipeline for credit risk assessment that predicts loan default probability using SQL data management, Random Forest classification, and a real-time Streamlit web interface.
🔗 **[Live Demo]([https://your-streamlit-url.streamlit.app](https://bank-loan-risk-engine.streamlit.app/))** | 📂 **[GitHub](https://github.com/abdulahadishaq512/Bank-Loan-Risk-Engine)**

## Features

SQL Integration — Normalized data management using SQLite with multi-table joins
Feature Engineering — Custom domain-specific metric Loan_to_Income_Ratio to improve predictive power
Predictive Modeling — Random Forest classifier achieving ~85% accuracy on held-out test data
Interactive Dashboard — Real-time risk inference via a deployed Streamlit web app


## Tech Stack
CategoryToolsLanguagePython 3.12ML & Datascikit-learn, pandas, numpyVisualizationseaborn, matplotlibDeploymentStreamlit CloudData ManagementSQLite3, Kaggle APIModel Persistencejoblib

## Project Structure
Bank-Loan-Risk-Engine/
├── app.py                  # Streamlit web application
├── credit_risk_model.pkl   # Trained Random Forest model
├── requirements.txt        # Project dependencies
└── README.md

## How to Run Locally
1. Clone the repository:
bashgit clone https://github.com/abdulahadishaq512/Bank-Loan-Risk-Engine.git
cd Bank-Loan-Risk-Engine
2. Install dependencies:
bashpip install -r requirements.txt
3. Run the application:
bashstreamlit run app.py

## How It Works

Data Ingestion — Dataset downloaded via Kaggle API and stored in a normalized SQLite database across 3 tables (Users, Financials, Loan_Status)
Feature Engineering — SQL JOIN query extracts relevant features; Loan_to_Income_Ratio computed as a custom risk metric
Model Training — Random Forest trained on encoded features with 80/20 train-test split
Deployment — Model serialized with joblib and served via Streamlit Cloud


## Model Performance
MetricScoreAccuracy~85%Precision~84%Recall~85%

## Author
Abdul Ahad Ishaq
**[GitHub](https://github.com/abdulahadishaq512/)** · [LinkedIn](https://linkedin.com/in/abdulahadishaq/)

Just replace:

your-streamlit-url with your actual Streamlit app URL
The model performance numbers with your actual results from the classification report
The LinkedIn URL with your actual profile
 Sonnet 4.6
