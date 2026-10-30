
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Step 1: Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

# Step 3: Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data:")
print(X_train_scaled[:5])

print("\nPreprocessing Completed!")
