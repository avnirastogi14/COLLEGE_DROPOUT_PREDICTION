"""
01_Data_Loading_and_Preprocessing.py

Purpose:
- Load the UCI College Dropout dataset
- Inspect data (info, missing values, target distribution)
- Clean missing values
- Encode categorical variables
- Save cleaned dataset for later steps
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
path = "data.csv"
data = pd.read_csv(path, sep=';')
print("Dataset loaded successfully.\n")

# Basic Information
print("\nData Description:")
print(data.describe(include='all'))
print("\nDataset Info:")
print(data.info())
print("\nTotal Rows:", data.shape[0])
print("Total Columns:", data.shape[1])
print("\nData Head:")
print(data.head())

# Check for missing values
print("\nMissing Values per Column:")
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])
print("\nTotal Missing Values in Dataset:", missing_values.sum())

# Cleaning Column Names
data.columns = (
    data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("’", "") .str.replace("'", "") 
)
print("\nCleaned Column Names:")
print(data.columns.tolist())

data = data.rename(columns={"nacionality": "nationality"}) # fixing typo

'''
Dataset includes entries with 1,2,...n for numeric columns, and strings for categorical columns.
To ensure data is meaningful, encoding numeric categorical variables as strings, as given in the website:
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
'''

# Converting and Mapping numerical codes to meaningful categories, for using ML Model(s) later

# mapping only the columns which add semantic clarity and ease during EDA i.e. Exploratory Data Analysis, this avoids noise.

print("\nMapping Numeric Columns to Meaningful labels: ")
mapping = {
    "marital_status": {1: "Single", 2: "Married", 3: "Widow", 4: "Divorced", 5: "Facto union", 6: "Legally separated"},
    "gender": {1: "Male", 0: "Female"},
    "daytime/evening_attendance": {1: "Daytime", 0: "Evening"},
    "displaced": {1: "Yes", 0: "No"},
    "educational_special_needs": {1: "Yes", 0: "No"},
    "debtor": {1: "Yes", 0: "No"},
    "tuition_fees_up_to_date": {1: "Yes", 0: "No"},
    "scholarship_holder": {1: "Yes", 0: "No"},
    "international": {1: "Yes", 0: "No"}
}


print("Attributes to be mapped:")
for it in mapping.keys():
    print(f"- {it}",": ", mapping[it])

for att, map_dict in mapping.items():
    if att in data.columns:
        data[att] = data[att].map(map_dict)

print("\nMapping completed successfully.\n")

print("\nTEST: Sample after mapping:")
print(data[["marital_status", "gender", "daytime/evening_attendance", "debtor"]].head())

# Label Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder

enc_col = [
    "application_mode",
    "course",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
    "nationality"
]

encoders = {}

for col in enc_col:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le  # store the encoder

print("\nLabel Encoding completed successfully.\n")
print("\nTEST: Sample after Label Encoding:")
print(data[enc_col].head())

# Encoding Target Variable

print("\nEncoding Target Variable:")
target_le = LabelEncoder()
data['target'] = target_le.fit_transform(data['target'])

target_mapping = dict(zip(target_le.classes_, target_le.transform(target_le.classes_))) # storing the mapping
print("Target mapping:", target_mapping)


# Save cleaned dataset
cleaned_path = "cleaned_data.csv"
data.to_csv(cleaned_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_path}\n")

# Data Cleaning
'''
Dataset includes no missing values.
'''

print("Data loading and preprocessing completed successfully.")
print("*"*50)

