'''
Script that runs UI to feed inputs to backend ML models toward Healthcare Premium Predictions
'''

import streamlit as st
# run from project root - Healthcare_Premium. Python then treats it as root and App within it as a package.
from Prediction_helper import predict

# setup title
st.title(':red[Healthcare] Premium Predictions')

# setup 4 rows each with 3 columns
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(1)

# display all categorical attributes and all possible values they take
categorical_attributes = {
    'Gender': ['Female', 'Male'],
    'Region': ['Southwest', 'Northwest', 'Southeast', 'Northeast'],
    'Marital_status': ['Unmarried', 'Married'],
    'BMI_Category': ['Normal', 'Underweight', 'Obesity', 'Overweight'],
    'Smoking_Status': ['No Smoking', 'Regular', 'Occasional'],
    'Employment_Status': ['Self-Employed', 'Freelancer', 'Salaried'],
    'Income_Level': ['<10L', '> 40L', '10L - 25L', '25L - 40L'],
    'Medical History': ['No Disease', 'Diabetes', 'Thyroid','Diabetes & High blood pressure', 'Diabetes & Heart disease','High blood pressure', 'Heart disease', 'Diabetes & Thyroid','High blood pressure & Heart disease'],
    'Insurance_Plan': ['Silver','Gold','Bronze']
    }

# setup placeholders for different attributes
with row1[0]:
    Age = st.number_input("Age", min_value=18, max_value=72)
with row1[1]:
    Gender = st.selectbox("Gender", categorical_attributes['Gender'])
with row1[2]:
    Region = st.selectbox("Region", categorical_attributes['Region'])

with row2[0]:
    Marital_status = st.selectbox("Marital_status", categorical_attributes['Marital_status'])
with row2[1]:
    Number_of_dependents = st.number_input("Number_of_dependents", min_value=1, max_value=5)
with row2[2]:
    BMI_Category = st.selectbox("BMI_Category", categorical_attributes['BMI_Category'])

with row3[0]:
    Smoking_Status = st.selectbox("Smoking_Status", categorical_attributes['Smoking_Status'])
with row3[1]:
    Employment_Status = st.selectbox("Employment_Status", categorical_attributes['Employment_Status'])
with row3[2]:
    Income_Level = st.selectbox("Income_Level", categorical_attributes['Income_Level'])

with row4[0]:
    Income_Lakhs = st.number_input("Income_Lakhs", min_value=1, max_value=930)
with row4[1]:
    Medical_History = st.selectbox("Medical History", categorical_attributes['Medical History'])
with row4[2]:
    Insurance_Plan =  st.selectbox("Insurance_Plan", categorical_attributes['Insurance_Plan'])

with row5[0]:
    Genetical_Risk = st.number_input("Genetical_Risk", min_value=0, max_value=5)

# setup input that is fed to the model
model_input = {
    'Age':Age,
    'Gender':Gender,
    'Region':Region,
    'Marital_status':Marital_status,
    'Number Of Dependants':Number_of_dependents,
    'BMI_Category':BMI_Category,
    'Smoking_Status':Smoking_Status,
    'Employment_Status':Employment_Status,
    'Income_Level':Income_Level,
    'Income_Lakhs':Income_Lakhs,
    'Medical History':Medical_History,
    'Insurance_Plan':Insurance_Plan,
    'Genetical_Risk':Genetical_Risk
}

if st.button("Predict"):
    # fetch result from optimal model
    result = predict(model_input)

    # display result
    st.success(f'Healthcare Premium Amount: INR {result:.2f}')