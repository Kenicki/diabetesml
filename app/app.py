import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('../src/model.joblib')

st.title("Diabetes Prediction App")

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
user_input = []

for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0)
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"Prediction: {result}")
