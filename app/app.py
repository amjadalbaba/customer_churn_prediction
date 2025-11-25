import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---- Load Model + Scaler ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---- Streamlit Page Config ----
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# ---- Custom Purple Style ----
st.markdown("""
    <style>
        .main {
            background-color: #f8f4fc;
        }
        h1, h2, h3, h4 {
            color: #5c2d91;
        }
        .stButton>button {
            background-color: #5c2d91;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #7b42c3;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("Customer Churn Prediction App")

st.write("Use the horizontal form below to enter customer details and predict churn.")


# ---- Two Column Layout ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Information")
    credit_score = st.number_input("Credit Score", 300, 900, 600)
    age = st.slider("Age", 18, 92, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
    products_number = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    st.subheader("")
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    has_cr_card = 1 if has_cr_card == "Yes" else 0

    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    is_active_member = 1 if is_active_member == "Yes" else 0

    country = st.selectbox("Country", ["France", "Germany", "Spain"])


# ---- Prepare Input ----
input_data = {
    "credit_score": credit_score,
    "gender": gender,
    "age": age,
    "tenure": tenure,
    "balance": balance,
    "products_number": products_number,
    "credit_card": has_cr_card,
    "active_member": is_active_member,
    "estimated_salary": estimated_salary,
    "country_Germany": 1 if country == "Germany" else 0,
    "country_Spain": 1 if country == "Spain" else 0
}

df = pd.DataFrame([input_data])

# Apply scaler only to numeric fields
num_features = [
    "credit_score", "age", "tenure", "balance",
    "products_number", "estimated_salary"
]

df[num_features] = scaler.transform(df[num_features])

# ---- Prediction ----
st.write("")

if st.button("Predict Churn"):
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.markdown('<div class="result-box" style="color:#b30000;">⚠️ The model predicts this customer is <b>likely</b> to churn**.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box" style="color:#008000;">✅ The customer is <b>not likely</b> to churn</div>', unsafe_allow_html=True)
