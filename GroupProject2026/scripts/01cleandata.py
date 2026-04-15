import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np   # linear algebra
import matplotlib.pyplot as plt  #graphs and plots
import seaborn as sns   #data visualizations
import plotly.express as px  #graphs and plots
import csv # Some extra functionalities for csv  files - reading it as a dictionary
from lightgbm import LGBMClassifier #sklearn is for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction

from sklearn.model_selection import train_test_split, cross_validate   #break up dataset into train and test sets

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# importing python library for working with missing data
import missingno as msno

# Load the dataset
df = pd.read_csv('GroupProject2026\data\diabetic_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Visualize missing values
msno.matrix(df)

# Drop columns with a high percentage of missing values
threshold = 0.5  # Set a threshold for dropping columns (e.g., 50% missing)
missing_percentage = df.isnull().mean()
columns_to_drop = missing_percentage[missing_percentage > threshold].index
df.drop(columns=columns_to_drop, inplace=True)

# Convert categorical variables to numeric using one-hot encoding
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Percentage of patients on Diabetes medication
diabetes_medication_percentage = (df['diabetesMed_Yes'].sum() / len
(df)) * 100
print(f"Percentage of patients on Diabetes medication: {diabetes_medication_percentage:.2f}%")

# Percentage of patients readmitted within 30 days
readmitted_within_30_days_percentage = (df['readmitted_<30'].sum() / len(df)) * 100
print(f"Percentage of patients readmitted within 30 days: {readmitted_within_30_days_percentage:.2f}%")


