import streamlit as st
import joblib
import numpy as np

model = joblib.load('src/model.joblib')

st.title("Diabetes Prediction App")

# User input
fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
values = []

for field in fields:
    values.append(st.number_input(field, step=1.0))

if st.button("Predict"):
    pred = model.predict([np.array(values)])
    st.success("Diabetic" if pred[0] == 1 else "Not Diabetic")
