import streamlit as st
import pickle
import numpy as np

# Load model
import os

model_path = os.path.join(os.getcwd(), 'credit_model.pkl')
model = pickle.load(open(model_path, 'rb'))

# Page config
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳")

# Title
st.title("💳 Credit Risk Predictor")
st.markdown("### Check if a customer is risky or safe")

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.info("This app uses a Machine Learning model (KNN) to predict credit risk.")

# Inputs (better UI)
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 25)
    income = st.number_input("Income (₹)", min_value=0, value=300000)

with col2:
    loan = st.number_input("Loan Amount (₹)", min_value=0, value=100000)
    credit = st.slider("Credit Score", 300, 900, 650)

# Warning logic
if loan > income * 5:
    st.warning("⚠️ Loan amount is too high compared to income!")

# Prediction
if st.button("Predict Risk"):
    input_data = np.array([[age, income, loan, credit]])
    
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    risk_score = prob[0][1] * 100

    st.subheader("📊 Result")

    # Progress bar
    st.progress(int(risk_score))

    st.write(f"Risk Probability: *{risk_score:.2f}%*")

    if prediction[0] == 1:
        st.error("⚠️ High Risk Customer")
        st.write("This applicant has a higher chance of defaulting.")
    else:
        st.success("✅ Low Risk Customer")
        st.write("This applicant is likely safe for loan approval.")
    
