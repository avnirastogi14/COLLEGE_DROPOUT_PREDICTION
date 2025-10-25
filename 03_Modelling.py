# 03_Modelling.ipynb

'''
Aim:
- Load cleaned data
- Prepare data for modelling
- Train initial models
- Evaluate model performance
- Save trained models for future use
'''
# Imports.
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
#### Functions.
## divider function

def divider():
        print("\n" + "*"*70 + "\n") # Divider for sections

# Divider function is used to separate sections in the notebook for better readability.
# Values taken in the divider function are arbitrary and can be modified as per requirement.
def evaluate_model(trModel, X_test, Y_test, name="Model"):
    # Predictions
    Y_cap = trModel.predict(X_test)

    # Base metrics for multiclass
    criteria = {
        "Accuracy": accuracy_score(Y_test, Y_cap),
        "Precision": precision_score(Y_test, Y_cap, average='weighted', zero_division=0),
        "Recall": recall_score(Y_test, Y_cap, average='weighted', zero_division=0),
        "F1-Score": f1_score(Y_test, Y_cap, average='weighted', zero_division=0)
    }

    # ROC-AUC (only if model can predict probabilities)
    if hasattr(trModel, "predict_proba"):
        try:
            Y_proba = trModel.predict_proba(X_test)
            criteria["ROC-AUC"] = roc_auc_score(Y_test, Y_proba, multi_class='ovr', average='weighted')
        except:
            criteria["ROC-AUC"] = None
    else:
        criteria["ROC-AUC"] = None

    print(f"--- {name} ---")
    for k, v in criteria.items():
        print(f"{k:10s}: {v:.4f}" if v is not None else f"{k:10s}: None")
    
    print("\nClassification Report:\n", classification_report(Y_test, Y_cap, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_cap))
    return criteria
# Plotting top n features function

def plot_feature_importance(featName,imps,N=20,save_path=None):
    feat_imp = pd.Series(imps, index=featName) # Making a series from feature importance array
    feat_imp = feat_imp.sort_values(ascending=False)[:N] # Sorting and taking top N(20) features
    
    # visualizing feature importance using bar plot
    plt.figure(figsize=(10,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f'Top {N} Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature importance plot saved to {save_path}") # Save the plot if save_path is provided
    plt.show()
# To get feature names after one-hot.

def get_Ft_Names(preprocess, catcol, numcol):
    numft = numcol # numerical feature names are same as original column names.
    catft = [] # to store categorical feature names after one-hot encoding

    if catcol:
        # get feature names for one-hot encoded columns
        ohe = preprocess.named_transformers_['cat'].named_steps['onehot']
        categories = ohe.categories_
        for cols,cats in zip(catcol, categories): # zip to iterate over column names and their categories
            catft.extend([f"{cols}_{cat}" for cat in cats])
    return list(numft) + catft
##### End of Functions
# Data Loading.
data = pd.read_csv('cleaned_data.csv')
print("Cleaned Data Loaded.\n")

print("Basic Data Info:")
display(data.info())
print("\nData Sample:")
display(data.head())
print("\nData Shape:", data.shape)
divider()
# Identifying Target Column

target_column = 'target'
print(f"Target Column Identified: {target_column}")
print(f"Target Column Count:\n{data[target_column].value_counts()}")
divider()
# Features and Target.

X = data.drop(columns=[target_column])
Y = data[target_column]

if Y.dtype == 'object' or Y.dtype=='0':
    ## introducing mapping for categorical target variable
    map1 = {'Dropout': 1, 'dropout': 1, 'Enrolled': 0, 'Graduate': 0, 'enrolled':0, 'graduate':0}
    Y = Y.map(map1).fillna(Y)

    # if we still have object type, we need to encode it
    if Y.dtype == 'object':
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        print(f"Target variable encoded with classes: {le.classes_}")

## printing quick information:
print("\nFeatures Types:")
print(X.dtypes.value_counts())
divider()
# categorical and numerical columns detection

catC = X.select_dtypes(include=['object', 'category']).columns.tolist()
NumC = X.select_dtypes(include=['number']).columns.tolist()

# for Categorical Columns which have numerical values stored as object type
Cardinality_Threshold = 20
# cardinality threshold is an arbitrary value to identify high cardinality categorical features
# You can adjust this value based on your dataset and requirements.
# We basically consider categorical features with unique values less than this threshold as low cardinality features ie categorical features suitable for visualization.

NumCat = []
for col in catC:
    unique_vals = X[col].nunique()
    if unique_vals <= Cardinality_Threshold and col not in catC:
        NumCat.append(col)
        if col in NumC:
            NumC.remove(col)
        if col not in catC:
            catC.append(col)

if len(NumCat) == 0:
    print(f"\nCategorical Columns with numerical values detected.")
    print("These columns will be treated as categorical for visualization purposes.")
    for col in NumCat:
        print(f"- {col}")

# Final Column Counts
print(f"\nFinal Categorical Columns Count: {len(catC)}")
print(f"Final Numerical Columns Count: {len(NumC)}")
divider()
# preprocessing steps before modelling

numTransformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# numerical transformer pipeline: imputing missing values with median and scaling features

catTransform_cols = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# categorical transformer pipeline: imputing missing values with most frequent value and one-hot encoding

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numTransformer, NumC),
        ('cat', catTransform_cols, catC)
    ]
)
# combining both numerical and categorical transformers into a single preprocessor

# Fitting Preprocessor
print("Fitting Preprocessor (MAY TAKE A LITTLE WHILE)")
preprocessor.fit(X)
print("Preprocessor Fitted Successfully.")
divider()
# fetch names of features after preprocessing
feature_names = get_Ft_Names(preprocessor, catC, NumC)
print(f"Total Features after Preprocessing: {len(feature_names)}")
divider()
# Train-Test Split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
# split X and Y into training and testing sets with 80-20 ratio
# stratify=Y ensures that the class distribution in train and test sets is similar to the original dataset.

print("Train-Test Split Completed.")
print(f"Training Set Shape: {X_train.shape}, Testing Set Shape: {X_test}.shape")
divider()
# Modelling and evaluation

models = {
    'LogisticRegression_pipeline': Pipeline(steps=[
        ('preprocessor', preprocessor),  # Ensure preprocessor is defined
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))  # Removed multi_class
    ]),
    # Logistic Regression is used for binary and multiclass classification problems.

    'RandomForest_pipeline': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    # Random Forest is used for both classification and regression tasks.
    # It is an ensemble learning method that constructs multiple decision trees during training.

    'GradientBoosting_pipeline': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    # Gradient Boosting is used for both classification and regression tasks.
    # It builds models sequentially, with each new model attempting to correct the errors of the previous models.
}

# Cross-Validation and Model Training
print("Cross-validating baseline models:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv is Stratified K-Fold Cross-Validation object.
# It splits the dataset into k folds while preserving the percentage of samples for each class.
# shuffle=True ensures that the data is shuffled before splitting into batches.

results = {}
for model_name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, Y_train, cv=cv, scoring='f1_macro')  # Use 'f1_macro' for multiclass
    results[model_name] = scores
    print(f"- {model_name}: F1-Score = {scores.mean():.4f} ± {scores.std():.4f}")
divider()

# Fit Models and Evaluate on Test Set
model_performance = {}
evalRes = {}
for model_name, pipeline in models.items():
    print(f"Training and Evaluating {model_name}:")
    pipeline.fit(X_train, Y_train)
    model_performance[model_name] = pipeline
    criteria = evaluate_model(pipeline, X_test, Y_test, name=model_name)  # Ensure evaluate_model is defined
    evalRes[model_name] = criteria
    divider()

# Parameter Tuning for RandomForest (using GridSearchCV)
print("Hyperparameter Tuning for RandomForest_pipeline:")
ForPar = models['RandomForest_pipeline']
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid = GridSearchCV(ForPar, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
grid.fit(X_train, Y_train)

print("GridSearch best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)
bestPar = grid.best_estimator_

# Checking tuned Random Forest on test values
tuned = evaluate_model(bestPar, X_test, Y_test, name="RandomForest_Tuned")
model_performance["RandomForest_Tuned"] = bestPar
evalRes["RandomForest_Tuned"] = tuned
divider()
# Selecting Best Model Based on F1-Score
best_model_name = max(evalRes.items(), key=lambda x: x[1]["F1-Score"])[0]
best_model = model_performance[best_model_name]

print(f"Best Model Selected Based on Test F1-Score: {best_model_name}")
divider()

# Feature Importance (only valid for tree-based models)
print("Checking Feature Importances (Applicable only for Tree-Based Models)\n")

# Fetching final feature names after preprocessing
feature_names = get_Ft_Names(preprocessor, catC, NumC)

# If model is pipeline, extract classifier part
clf = best_model.named_steps['classifier'] if isinstance(best_model, Pipeline) else best_model

if hasattr(clf, "feature_importances_"):
    # Transform training data through preprocessor to align importance array
    X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train)
    importances = clf.feature_importances_

    # Ensuring dimensional consistency
    if len(importances) == X_train_transformed.shape[1]:
        plot_feature_importance(feature_names, importances, N=20, save_path="feature_importance.png")
    else:
        print("Feature importance vector length does not match transformed dataset shape.\nSkipping feature importance plot.")
else:
    print(f"{best_model_name} does not support feature_importances_ (e.g., Logistic Regression). Skipping.")
divider()

# Saving Best Model and Evaluation Results
print("Saving Best Model and Evaluation Metrics:")

OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

final_model_path = os.path.join(OUT_DIR, f"{best_model_name}_model.joblib")
joblib.dump(best_model, final_model_path)
print(f"Model Saved Successfully at: {final_model_path}")

# Save evaluation metrics for reporting
pd.DataFrame(evalRes).T.to_csv(os.path.join(OUT_DIR, "evaluation_results.csv"))
print("Evaluation Results Saved to: models/evaluation_results.csv")

print("\nAll Steps Completed. Model Ready for Deployment and Integration.")
divider()