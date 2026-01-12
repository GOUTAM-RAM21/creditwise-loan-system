import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
model = pickle.load(open("logistic_regression_model.pkl", "rb"))

# -------------------------------
# App Header
# -------------------------------
st.title("🏦 Smart Loan Approval System")
st.subheader("AI-powered Loan Eligibility Prediction using Machine Learning")
st.caption("Developed using Logistic Regression | Streamlit Deployment")

st.markdown("---")

# -------------------------------
# User Input Section
# -------------------------------
st.header("📋 Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100, value=28)
income = st.number_input("Monthly Income (₹)", min_value=0.0, value=45000.0, step=1000.0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0, value=120000.0, step=5000.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict Loan Status"):
    
    st.warning(
        "⚠️ Note: This prediction is a **demo placeholder**. "
        "The trained model uses **27 features with scaling**. "
        "For accurate results, all features and the original scaler must be implemented."
    )

    # ----------------------------------
    # Placeholder input (27 features)
    # ----------------------------------
    dummy_input = np.zeros((1, 27))

    # Mapping few real inputs (ONLY FOR DEMO)
    dummy_input[0, 0] = income        # Proxy feature
    dummy_input[0, 2] = age           # Proxy feature
    dummy_input[0, 8] = loan_amount   # Proxy feature

    # No real scaling applied (demo only)
    scaled_input = dummy_input

    # Prediction
    prediction = model.predict(scaled_input)

    # -------------------------------
    # Result
    # -------------------------------
    st.markdown("---")
    st.header("📊 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ **Loan Approved**")
    else:
        st.error("❌ **Loan Rejected**")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("⚙️ This project demonstrates end-to-end ML model deployment using Streamlit.")
