"""
02_Raw_Data_Visualization.py

Purpose:
- Visualize raw dataset before encoding & preprocessing
- Understand class balance, distributions, categorical frequencies, outliers, and correlations
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw data
raw_data = pd.read_csv("data.csv", sep=';')
print("Raw Dataset Loaded.\n")

# Column Name Cleaning.
# All columns names were initially messy, a string with leading spaces, punctuation, inconsistent casing, etc.
raw_data.columns = (
    raw_data.columns.str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace("’", "")
    .str.replace("'", "")
)
raw_data = raw_data.rename(columns={"nacionality": "nationality"})
print("Column Names Cleaned.\n")

# 1. Target Variable Distribution 
plt.figure(figsize=(6,4))
sns.countplot(data=raw_data, x='target')
plt.title("Target Variable Distribution (Raw Data)")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.show()
print("\nTarget Value Counts:\n", raw_data['target'].value_counts())


# 2. Identify Categorical & Numerical Columns 
CatC = raw_data.select_dtypes(include=['object']).columns.tolist()
NumC = raw_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
NumC = [col for col in NumC if col != 'target']

# 3. Grid of Categorical Count Plots
print("\n"," "*50,"Categorical Feature Distributions")
n_cat = len(CatC)
cols = 3
rows = (n_cat // cols) + 1
print("\nCategorical Columns:")
for col in CatC:
    print(f"- {col}")

plt.figure(figsize=(18, 5 * rows))
for i, col in enumerate(CatC, 1):
    plt.subplot(rows, cols, i)
    sns.countplot(data=raw_data, x=col)
    plt.title(col.replace('_',' ').title())
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 4. Grid of Numerical Histograms
print("\n"," "*50,"Numerical Feature Distributions")
n_num = len(NumC)
cols = 3
rows = (n_num // cols) + 1
print("\nNumerical Columns:")
for col in NumC:
    print(f"- {col}")

plt.figure(figsize=(18, 5 * rows))
for i, col in enumerate(NumC, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(raw_data[col], kde=True)
    plt.title(col.replace('_',' ').title())
plt.tight_layout()
plt.show()


# 5. Outlier Detection (Boxplots)
print("\n"," "*50,"Outlier Detection in Numerical Features")
plt.figure(figsize=(15, 3 * rows))
for i, col in enumerate(NumC, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x=raw_data[col])
    plt.title(f"Outliers in {col.replace('_',' ').title()}")
plt.tight_layout()
plt.show()


# 6. Correlation Heatmap (Numerical Only)
print("\n"," "*50,"Correlation Heatmap (Numerical Features)")
plt.figure(figsize=(8,6))
sns.heatmap(raw_data[NumC].corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap (Raw Numerical Data)")
plt.show()

# End
print("\nRaw Data Visualization Completed.")
print("*"*60)
