#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Analysis
# 
# ## Overview
# This project explores survival patterns among Titanic survivors using data analysis and machine learning.
# We clean the dataset, visualize survival trends, and build a model using **RandomForestClassifier**.
# 
# ## Dataset
# The dataset includes details like age, gender, ticket class, fare price, and survival outcomes.
# Our goal is to identify which factors had the most impact on survival.

# ## Data Loading and Initial Exploration
# First we load the data, inspect its structure, and check for missing values.
# 
# ### Steps
# * Import the necessary libraries (e.g., pandas)
# * Load the dataset ('titanic.csv')
# * Check column data types
# * Identify missing values

import pandas as pd

# Load the dataset
df = pd.read_csv('titanic.csv')

# # Titanic Data Exploration

# ## Dataset Shape
# Displays the number of rows and columns in the dataset
print("Dataset shape:", df.shape)

# ## Summary Statistics
# Provides key metrics like mean, standard deviation, and quartiles for numeric features
print("\nSummary statistics:")
print(df.describe())

# ## Column Types
# Shows data types for each feature and helps identify non-numeric columns
print("\nColumn data types:")
df.info()

# ## First Few Rows
# Gives a sample of the dataset to understand content and layout
print("\nFirst few rows:")
print(df.head())

# ## Number of Null Values
# Identifies missing values in each column to guide the cleaning process
print("\nMissing values per column:")
print(df.isnull().sum())

# ## Data Cleaning and Feature Engineering

# ### Handling Missing Values
# To ensure model accuracy and consistency, we addressed missing data:
# * **Age**: Replaced missing values with the column's **mean**
# * **Cabin**: Extracted the **first letter** to create a simplified **Cabin_Letter**
# * **Categorical Columns** ('Sex', 'Embarked'): Applied **one-hot encoding** using pd.get_dummies()

# ### Feature Creation
# We engineered several features to uncover survival patterns:
# * **Cabin_Letter**: Isolated the first character from 'Cabin' to capture possible deck locations
# * **AgeGroup**: Binned age into logical categories (Child, Teen, Young Adult, Middle-Aged, Senior, Elder, Very Elder)
# * **FareClass**: Combined 'Fare' * 'Pclass' to represent the economic weight of a passenger's ticket
# * **FamilySize**: Summed 'SibSp' + 'Parch' + 1 to represent the total number of family members aboard
# * **FareClassBin binned FareClass groups in categories

# Each of these features was designed to surface deeper patterns in passenger survival
# and to provide our model with more meaningful input.

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Feature engineering
df['Cabin_Letter'] = df['Cabin'].astype(str).str[0]
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['FareClass'] = df['Fare'] * df['Pclass']
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 35, 50, 65, 85, 90],
    labels=['Child', 'Teen', 'Young Adult', 'Middle-Aged', 'Senior', 'Elder', '90+'],
    right=False
)
df['FareClassBin'] = pd.cut(
    df['FareClass'],
    bins = [0, 50, 100, 150, 200, df['FareClass'].max() + 1],
    labels = ['0-50', '50-100', '100-150', '150-200', '200+'] ,
    include_lowest = True)


# Grouped survival rates by AgeGroup
print("\nSurvival rate by AgeGroup:")
print(df.groupby('AgeGroup', observed=True)['Survived'].mean())

# Grouped survival rate by FamilySize
print("\nSurvival rate by FamilySize:")
print(df.groupby('FamilySize')['Survived'].mean())

# Grouped survival rate by Sex and Pclass
print("\nSurvival rate by Sex and Pclass:")
print(df.groupby(['Sex', 'Pclass'])['Survived'].mean())

# Drop SibSp and Parch since we've combined them into FamilySize
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Preview the updated DataFrame
print("\nUpdated DataFrame preview:")
print(df.head())

# ## Exploratory Data Analysis
# 
# ## Key Insights
# We use visualizations to explore survival patterns using matplotlib and seaborn:
# 
# * FareClass vs. Survival (Boxplot) → higher FareClass improves survival odds
# * FamilySize and Pclass vs. Survival (Heatmap) → extreme family sizes have lower survival rates
# * Pclass vs. Survival (Barplot) → first class passengers have higher survival rates
# * FareClassBin and Sex vs. Survival(Barplot) → females dominated most bins with higher survival rates

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot: FareClass vs. Survival
plt.figure(figsize=(6, 4))
sns.boxplot(x='Survived', y='FareClass', data=df)
plt.title('FareClass vs. Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('FareClass')
plt.tight_layout()
plt.show()

# Heatmap: Survival Rate by Family Size and Pclass
family_class_survival = df.pivot_table(
    values='Survived',
    index='FamilySize',
    columns='Pclass',
    aggfunc='mean'
)

plt.figure(figsize=(8, 5))
sns.heatmap(family_class_survival, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Survival Rate by Family Size & Class')
plt.xlabel('Passenger Class')
plt.ylabel('Family Size')
plt.tight_layout()
plt.show()

# Bar Plot: Survival Rate by Passenger Class
plt.figure(figsize = (8, 5))
sns.barplot(data = df, x = 'Pclass', y = 'Survived', errorbar = None)
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()

# Bar Plot: Survival Rate by FareClassBin and Sex
plt.figure(figsize = (8, 5))
sns.barplot(data = df, x = 'FareClassBin', y = 'Survived', hue = 'Sex', errorbar = None)
plt.title('Survival Rate by FareClassBin and Sex')
plt.ylabel('Mean Survival Rate')
plt.xlabel('Fare Class Range')
plt.legend(title = 'Sex', loc = 'upper left', bbox_to_anchor = (1,1))
plt.tight_layout()
plt.show()

# ## Machine Learning — Random Forest

# ### Steps:
# 1. Import libraries
# 2. Define X (features) and y (target variable 'Survived')
# 3. Train-test split (80/20)
# 4. Train RandomForestClassifier (n_estimators = 100)
# 5. Evaluate model (accuracy, feature importance)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop columns that won't be used in modeling
df.drop(['Name', 'Ticket', 'Cabin', 'Cabin_Letter'], axis=1, inplace=True, errors='ignore')

# Define features and target
X = df[['Pclass','Age', 'Sex_male','Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional for Random Forest, but good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")

# Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 Feature Importances:")
print(importances.head(10))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Preview the Updated DataFrame
print("\nUpdated DataFrame preview:")
print(df.head())


# Print model summary
print("\nRandom Forest Model:")
print(model)

# Feature Importance from Random Forest
importances = model.feature_importances_
feature_names = X.columns
feature_importance = dict(zip(feature_names, importances))

# Sort features by importance
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
features, scores = zip(*sorted_features)

#Visualize Importance
plt.figure(figsize = (8, 5))
plt.barh(features, scores, color = 'skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

print(feature_importance)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(features, scores, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance in Survival Prediction')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

# Print original feature importances
importances = model.feature_importances_
feature_names = X.columns
feature_importance = dict(zip(feature_names, importances))
print("\nOriginal Feature Importances:")
print(feature_importance)
# ----------------------------------------
# Feature Engineering: Combine Pclass and Age
# ----------------------------------------
df['Pclass_Age'] = df['Pclass'] * df['Age']

# Define new feature set
X = df[['Pclass_Age', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model on new features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance after feature engineering
importances = model.feature_importances_
feature_names = X.columns
feature_importance = dict(zip(feature_names, importances))

print("\nFeature Importances After Adding Pclass_Age:")
print(feature_importance)

# Visualize updated feature importance
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
features, scores = zip(*sorted_features)

plt.figure(figsize=(8, 5))
plt.barh(features, scores, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance After Joining Columns')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ----------------------------------------
# Prepare for Hyperparameter Tuning
# ----------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid (to be filled in next)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize base model
model = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best parameters and model
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

best_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_model.fit(X_train, y_train)

# ----------------------------------------
# Evaluation
# ----------------------------------------

from sklearn.metrics import accuracy_score, confusion_matrix

# Predict and evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTuned Model Accuracy: {accuracy:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

# Standard classification metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nStandard Threshold (0.5):")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")

# ----------------------------------------
# Threshold Tuning
# ----------------------------------------

# Get predicted probabilities for the positive class (Survived = 1)
y_probs = best_model.predict_proba(X_test)[:, 1]

# Set custom threshold
threshold = 0.60  # Try 0.40, 0.50, 0.70 for comparison
y_pred_threshold = np.where(y_probs >= threshold, 1, 0)

# Recalculate metrics with new threshold
precision = precision_score(y_test, y_pred_threshold)
recall = recall_score(y_test, y_pred_threshold)
f1 = f1_score(y_test, y_pred_threshold)

print(f"\nCustom Threshold: {threshold}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")

# ----------------------------------------
# Conclusion and Next Steps
# ----------------------------------------

# ## Key Findings:
#
# * FareClass and Pclass were strong predictors of survival.
# * FamilySize showed mixed effects — both very large and very small families had lower survival rates.
# * The Random Forest model delivered solid predictive performance, and hyperparameter tuning improved accuracy.

# ## Future Enhancements:
#
# * Further hyperparameter tuning using GridSearchCV or RandomizedSearchCV
# * Explore alternative models (e.g., Gradient Boosting, XGBoost, or Logistic Regression)
# * Deploy the model as an interactive web app (e.g., using Streamlit or Flask)
# * Perform cross-validation and ROC curve analysis for deeper evaluation
