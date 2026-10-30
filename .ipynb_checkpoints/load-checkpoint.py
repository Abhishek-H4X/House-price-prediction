import pandas as pd
from sklearn.datasets import fetch_california_housing
# Load dataset
housing = fetch_california_housing(as_frame=True)

# Convert into DataFrame
df = housing.frame

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical information
print("\nStatistical Summary:")
print(df.describe())