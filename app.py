import streamlit as st
import joblib
import pandas as pd
import os

# Load model from same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'credit_risk_model.pkl')

st.title("Bank Loan Risk Engine")

if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.stop()

model = joblib.load(model_path)

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
dti = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.2)
education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])

if st.button("Predict Risk"):
    loan_to_income = loan_amount / income if income > 0 else 0

    input_df = pd.DataFrame([[age, income, loan_amount, dti, loan_to_income]],
                              columns=['Age', 'Income', 'LoanAmount', 'DTIRatio', 'Loan_to_Income_Ratio'])

    for col in model.feature_names_in_:
        if col.startswith('Education_') and col not in input_df.columns:
            input_df[col] = 0

    edu_col = f"Education_{education}"
    if edu_col in input_df.columns:
        input_df[edu_col] = 1

    input_df = input_df[model.feature_names_in_]
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Status: High Risk / Denied")
    else:
        st.success("Status: Approved ✅")
