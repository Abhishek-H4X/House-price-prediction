
import os
import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Create models folder
os.makedirs("models", exist_ok=True)

# 2. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 3. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# 7. Create models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000)
}

# 8. Train and save models
results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    predictions = model.predict(X_test_scaled)

    # Evaluate
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # Save model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, f"models/{filename}")

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# 9. Display results
comparison = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

print("\nAll models and scaler saved successfully!")
