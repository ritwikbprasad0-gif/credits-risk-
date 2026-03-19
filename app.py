import streamlit as st
import pickle 
import numpy as np 

model = pickle.load(open('model.pkl', 'rb'))

st.title("Credit Risk Predictor")

age = st.number_input("Age")
income = st.number_input("Income")
loan = st.number_input("Loan Amount")
credit= st.number_input("Credit Score")

if st.button("Predict"):
  input_data = np.array([[age, income, loan, credit]])
  prediction = model.predict(input_data)  

 if prediction[0] == 1:
   st.eroor("High Risk")
else:
  st.success("Low Risk")
