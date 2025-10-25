"""
Predictions.py

Purpose:
- Load the trained model from the UCI College Dropout Prediction project.
- Allow predictions for a single student input or batch predictions with hardcoded test data.
- Output predictions (Dropout, Enrolled, Graduate) and class probabilities.

Dependencies:
- Saved model (e.g., models/RandomForest_model.joblib)
- Cleaned dataset (cleaned_data.csv) for feature reference
"""

# Imports
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load the trained model
MODEL_PATH = "models/RandomForest_pipeline_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run 03_Modelling.ipynb first.")

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.\n")

# Load cleaned data to get feature names and types
data = "cleaned_data.csv"
if not os.path.exists(data):
    raise FileNotFoundError(f"Cleaned data not found at {data}. Run 01_Data_Loading_Cleaning.py first.")

data = pd.read_csv(data)
feature_cols = [col for col in data.columns if col != 'target']
categorical_cols = [
    'gender', 'marital_status', 'daytime/evening_attendance', 'application_mode', 'course',
    'mothers_qualification', 'fathers_qualification', 'mothers_occupation', 'fathers_occupation',
    'nationality', 'displaced', 'educational_special_needs', 'debtor', 'tuition_fees_up_to_date',
    'scholarship_holder', 'international'
]
numerical_cols = [col for col in feature_cols if col not in categorical_cols]

# Target mapping (from 01_Data_Loading_Cleaning.py)
target_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
print("Target mapping:", target_mapping, "\n")

# Function to preprocess single student input
def preprocess_single_input(student_data, feature_cols, categorical_cols, numerical_cols):
    # Create DataFrame with feature columns
    input_df = pd.DataFrame([student_data], columns=feature_cols)
    
    # Ensure categorical columns are strings
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    
    # Ensure numerical columns are floats
    for col in numerical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(float)
    
    return input_df

# Function to make predictions
def predict_student(model, input_data, feature_cols, is_single=True):
    """
    Predict class and probabilities for single or batch input.
    Returns predictions and probabilities with class labels.
    """
    # Ensure input is a DataFrame
    if is_single:
        input_data = preprocess_single_input(input_data, feature_cols, categorical_cols, numerical_cols)
    
    # Make predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
    
    # Map predictions to class names
    pred_labels = [target_mapping[pred] for pred in predictions]
    
    # Format probabilities
    if probabilities is not None:
        prob_df = pd.DataFrame(probabilities, columns=[target_mapping[i] for i in range(len(target_mapping))])
    else:
        prob_df = None
    
    return pred_labels, prob_df

# 1. Single Student Prediction
print("="*50)
print("Single Student Prediction")
print("="*50)

# Example single student input
single_student = {
    'marital_status': 'Single',
    'application_mode': '1',
    'course': '171',
    'daytime/evening_attendance': 'Daytime',
    'mothers_qualification': '1',
    'fathers_qualification': '1',
    'mothers_occupation': '9',
    'fathers_occupation': '9',
    'nationality': '1',
    'displaced': 'Yes',
    'educational_special_needs': 'No',
    'debtor': 'No',
    'tuition_fees_up_to_date': 'Yes',
    'gender': 'Male',
    'scholarship_holder': 'No',
    'international': 'No',
    'age_at_enrollment': 20.0,
    'admission_grade': 130.0,
    'gdp': 1.79,
    'unemployment_rate': 9.4,
    'inflation_rate': 0.3,
    'curricular_units_1st_sem_(grade)': 14.5,
    'curricular_units_2nd_sem_(grade)': 14.0
}

# Mean Median Mode imputation for missing features
for col in feature_cols:
    if col not in single_student:
        if col in numerical_cols:
            single_student[col] = data[col].median()
        else:
            single_student[col] = data[col].mode()[0]

# Predict for single student
pred_label, pred_proba = predict_student(model, single_student, feature_cols, is_single=True)

print("\nSingle Student Input:")
for key, value in single_student.items():
    print(f"{key}: {value}")
print("\nPrediction:", pred_label[0])
if pred_proba is not None:
    print("\nProbabilities:")
    print(pred_proba.round(3))

# 2. Batch Predictions with Hardcoded Test Data
print("\n" + "="*50)
print("Batch Predictions")
print("="*50)

# Hardcoded test data (3 students, realistic values based on UCI dataset)
batch_data = pd.DataFrame([
    {
        'marital_status': 'Single',
        'application_mode': '1',
        'course': '171',
        'daytime/evening_attendance': 'Daytime',
        'mothers_qualification': '1',
        'fathers_qualification': '1',
        'mothers_occupation': '9',
        'fathers_occupation': '9',
        'nationality': '1',
        'displaced': 'Yes',
        'educational_special_needs': 'No',
        'debtor': 'Yes',
        'tuition_fees_up_to_date': 'No',
        'gender': 'Female',
        'scholarship_holder': 'No',
        'international': 'No',
        'age_at_enrollment': 19.0,
        'admission_grade': 120.0,
        'gdp': 0.5,
        'unemployment_rate': 10.8,
        'inflation_rate': 1.4,
        'curricular_units_1st_sem_(grade)': 10.0,
        'curricular_units_2nd_sem_(grade)': 9.5
    },
    {
        'marital_status': 'Married',
        'application_mode': '17',
        'course': '9254',
        'daytime/evening_attendance': 'Evening',
        'mothers_qualification': '3',
        'fathers_qualification': '3',
        'mothers_occupation': '5',
        'fathers_occupation': '5',
        'nationality': '1',
        'displaced': 'No',
        'educational_special_needs': 'No',
        'debtor': 'No',
        'tuition_fees_up_to_date': 'Yes',
        'gender': 'Male',
        'scholarship_holder': 'Yes',
        'international': 'No',
        'age_at_enrollment': 25.0,
        'admission_grade': 140.0,
        'gdp': 2.0,
        'unemployment_rate': 8.9,
        'inflation_rate': 0.8,
        'curricular_units_1st_sem_(grade)': 15.0,
        'curricular_units_2nd_sem_(grade)': 15.5
    },
    {
        'marital_status': 'Single',
        'application_mode': '39',
        'course': '9085',
        'daytime/evening_attendance': 'Daytime',
        'mothers_qualification': '2',
        'fathers_qualification': '2',
        'mothers_occupation': '7',
        'fathers_occupation': '7',
        'nationality': '1',
        'displaced': 'Yes',
        'educational_special_needs': 'No',
        'debtor': 'No',
        'tuition_fees_up_to_date': 'Yes',
        'gender': 'Female',
        'scholarship_holder': 'Yes',
        'international': 'No',
        'age_at_enrollment': 18.0,
        'admission_grade': 135.0,
        'gdp': 1.0,
        'unemployment_rate': 9.0,
        'inflation_rate': 0.5,
        'curricular_units_1st_sem_(grade)': 13.0,
        'curricular_units_2nd_sem_(grade)': 13.5
    }
], columns=feature_cols)

# Fill missing features in batch data
for col in feature_cols:
    if col not in batch_data.columns:
        if col in numerical_cols:
            batch_data[col] = data[col].median()
        else:
            batch_data[col] = data[col].mode()[0]

# Predict for batch
batch_pred_labels, batch_pred_proba = predict_student(model, batch_data, feature_cols, is_single=False)

print("\nBatch Predictions:")
for idx, (pred, row) in enumerate(zip(batch_pred_labels, batch_data.to_dict('records'))):
    print("\n"+f"Student {idx+1}:"+"\n")
    for key, value in row.items():
        print(f"{key}: {value}")
    print("\nPrediction:", pred)
    if batch_pred_proba is not None:
        print("\nProbabilities:")
        print(batch_pred_proba.iloc[idx].round(3))

print("\nPredictions completed successfully.")
print("*"*50)