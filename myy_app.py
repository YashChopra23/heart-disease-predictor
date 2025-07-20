import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="💓")
st.title("💓 Heart Disease Predictor")
st.write("Enter patient information below to assess the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=130)
chol = st.number_input("Serum Cholesterol in mg/dl (chol)", value=250)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", value=1.0, format="%.1f")
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", options=[1, 2, 3])

# Convert categorical
sex = 1 if sex == "Male" else 0

# Form the input array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale the input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk: The patient is likely to have heart disease. (Confidence: {prediction_proba:.2%})")
    else:
        st.success(f"✅ Low Risk: The patient is unlikely to have heart disease. (Confidence: {1 - prediction_proba:.2%})")

