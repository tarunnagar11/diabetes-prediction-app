#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras

import streamlit as st


# load modeland scaler 
model = tf.keras.models.load_model('Diabetic_model.h5')
scaler = joblib.load('scaler.pkl')

# age gape
st.set_page_config(page_title='Diabetes Prediction')
st.title('Diabetes Prediction')
st.markdown("Entre the following details to predict the duabetes")

# input fields
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=10, value=1)
glucose =st.number_input('Glucose Level', min_value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin Level', min_value=0)
bmi = st.number_input('BMI', min_value=1)
diabetespedigreefunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
age = st.number_input('Age', min_value=0)

# make predictions
if st.button('Predict Diabetes'):
    # scale the input data
    input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    result = "Not Diabetic " if prediction < 0.5 else "Diabetic"
    st.subheader("The result of the prediction are : ",result)
    