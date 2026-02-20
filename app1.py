import pandas as pd 
import numpy as np
import joblib
import streamlit as st
#loading model
model=joblib.load("model1.pkl")
scaler=joblib.load("scaler1.pkl")
#start of streamlit code
st.title("Student Performance Classifier")
st.write("<span style='color:#4CAF50'>**Input the details based on Study Hours & Sleep Hours of a single student to classify performance**</span>",unsafe_allow_html=True)
Study_Hours=st.number_input("**Study time(hrs)**",min_value=0.500000,max_value=24.000000,step=0.100000,format="%.9f")
Sleep_Hours=st.number_input('**Sleep time(hrs)**',min_value=0.500000,max_value=24.000000,step=0.100000,format="%.9f")
# action button
if st.button("Classify",type="primary"):
    input_features=np.array([[Study_Hours,Sleep_Hours]])
    #scale feature
    input_features_scaled=scaler.transform(input_features)
    #make prediction now
    prediction=model.predict(input_features)
    #map pass values
    pas= { 0:'Student Fails',1:'Student Passes'}
    student_performance=pas[prediction[0]]
    #display now
    st.success(f"The predicted performance is: {student_performance}")