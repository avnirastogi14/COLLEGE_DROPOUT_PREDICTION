"""
02_Data_Visualization_EDA.py

Purpose:
- Load the cleaned UCI College Dropout dataset
- Perform Exploratory Data Analysis (EDA)
- Visualize distributions and relationships in the data
- Understand features for ML modeling
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# Load cleaned dataset
path = "cleaned_data.csv"
data = pd.read_csv(path)
print("Cleaned dataset loaded successfully.\n")

# Basic Information
print("\nData Description:")
print(data.describe(include='all'))
print("\nDataset Info:")
print(data.info())
print("\nTotal Rows:", data.shape[0])
print("Total Columns:", data.shape[1])
print("\nData Head:")
print(data.head())

# VISUALIZATIONS

#) Setting the style for seaborn
sns.set(style="whitegrid")

## 1. Target Variable Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=data)
plt.title("Target Variable Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.show()

print("\nTarget Value Counts:")
print(data['target'].value_counts())

##2. Categorical Features Distribution
print("\n"," "*50,"Categorical Feature Distributions")

categorical_cols = [
    'gender', 'marital_status', 'daytime/evening_attendance',
    'application_mode', 'course', 
    'mothers_qualification', 'fathers_qualification',
    'mothers_occupation', 'fathers_occupation', 'nationality'
]

# Keeping only columns that exist and have data
validC = [col for col in categorical_cols if col in data.columns and data[col].dropna().shape[0] > 0]
'''
col for col in categorical_cols if col in data.columns and data[col].dropna().shape[0] > 0 
checks for the existence of each column in the dataset and ensures it has non-missing values.
'''

# Grid layout
n_cols = 2
n_rows = (len(validC) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(validC):
    ## enumerate allows us to loop with an index and the column name simultaneously.
    sns.countplot(x=col, data=data, order=data[col].value_counts().index, ax=axes[i])
    axes[i].set_title(f"{col}", fontsize=11)
    axes[i].tick_params(axis='x', rotation=45)

# Hide unused subplots
# - unused subplots are the extra axes in the grid layout.
for j in range(len(validC), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout() # tight_layout adjusts subplots to fit into figure area.
plt.show()

## 3. Categorical Features vs Target
print("\n"+" "*45+"Categorical Features vs Target")
# - This helps in understanding how the target variable is distributed across different categories of each categorical feature.

print("\n" + " "*20 + " Categorical vs Target Distributions " + " "*20)
target = 'target' # target variable name

# For Grid Layout
fig, axes = plt.subplots(len(validC)//2 + 1, 2, figsize=(18, 4*len(validC)//2))
axes = axes.flatten()

for i, col in enumerate(validC):
    sns.countplot(x=col, hue=target, data=data, ax=axes[i])
    axes[i].set_title(f"{col} vs {target}")
    axes[i].tick_params(axis='x', rotation=45)
# For each categorical feature, a countplot is created to visualize the distribution of the target variable across its categories.

# Hide unused subplots
for j in range(len(validC), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

## 4. Numerical Features Distribution
### numeric vs numeric

print("\n" + " "*20 + " Numeric Feature Distributions " + " "*20)

numeric_cols = [
    'admission_grade', 'age_at_enrollment', 'gdp', 
    'unemployment_rate', 'inflation_rate',
    'curricular_units_1st_sem_(grade)', 'curricular_units_2nd_sem_(grade)'
]

# Table Summary
print("\n"+" "*25+"Numeric Feature Summary"+" "*25)
print(data[numeric_cols].describe())

# Skewness
# - Skewness measures the asymmetry of the data distribution.

print("\n\nSkewness:")
for col in numeric_cols:
    if col in data.columns:
        print(f"{col}: skew={data[col].skew():.2f}")

# For Histograms
# - Keep only columns that exist and have non-null data
valid_numeric_cols = [col for col in numeric_cols if col in data.columns and data[col].dropna().shape[0] > 0]

# Grid Layout
n_cols = 2
n_rows = (len(valid_numeric_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(valid_numeric_cols):
    sns.histplot(data[col].dropna(), kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f"{col} Distribution", fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

# Hide unused plots
for j in range(len(valid_numeric_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

## 5. Correlation Heatmap (for numerical features)

plt.figure(figsize=(8, 6))
corr = data[numeric_cols + ['target']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.show()
'''
Allows us to see how numerical variables relate to each other and to the target.
Also narrows down features for modeling.

High Correlation amongst features gives multicollinearity issues in models like Linear Regression.
High Correlation with target indicates important predictors.
'''

## 6. Pairplot (numeric vs target hue)
sns.pairplot(data[numeric_cols + ['target']], hue='target', diag_kind='kde')
plt.suptitle("Pairwise Relationships by Target", y=1.02)
plt.show()

'''
Helps visualize relationships between numerical features colored by target class.
Reveals patterns and potential clusters in the data, also confirms for linear/non-linear relationships.
'''

## 7. Boxplots for Outlier Detection

print("\n" + " "*20 + " Boxplots for Outlier Detection " + " "*20)
plt.figure(figsize=(14, 5 * n_rows))
axes = plt.subplots(n_rows, n_cols)[1].flatten()
for i, col in enumerate(valid_numeric_cols):
    sns.boxplot(x=data[col], ax=axes[i])
    axes[i].set_title(f"{col} Boxplot", fontsize=11)

# Hide unused plots
for j in range(len(valid_numeric_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

## 8. Mutual Information between features and target
# Feature Importance using Mutual Information
print("\n" + " "*40 + " Feature Importance using Mutual Information " + " "*40)

target = 'target'

X = data.drop(columns=[target]) # pred var
y = data[target] # target var

# One-hot encode categorical features for MI calculation
# - MI requires numeric inputs.
X_enc = pd.get_dummies(X, drop_first=True)

# MI Scores.
# - MI = 0 means the feature gives no information about the target; higher MI means stronger dependence.
mi_scores = mutual_info_classif(X_enc, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X_enc.columns).sort_values(ascending=False)

# Top N features, threshold (used in graph and listing)
top_n = 10
mi_top = mi_scores.head(top_n)

# BarPlot of top 10 features, by MI Score.
plt.figure(figsize=(10, 6))
sns.barplot(x=mi_top.values, y=mi_top.index, color="mediumseagreen")  # avoids palette warning
plt.title(f"Top {top_n} Features by Mutual Information")
plt.xlabel("Mutual Information Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Listing top 10 features.
print("\nTop features by MI score:")
print(mi_top)

print("Data Visualization / EDA completed successfully.")
print("*"*50)