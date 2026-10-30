
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create Lasso model
lasso_model = Lasso(alpha=0.01, max_iter=10000)

# 6. Train model
lasso_model.fit(X_train_scaled, y_train)

# 7. Make predictions
y_pred = lasso_model.predict(X_test_scaled)

# 8. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("LASSO REGRESSION RESULTS")
print("------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 9. Display coefficients
print("\nLasso Coefficients:")

for feature, coefficient in zip(X.columns, lasso_model.coef_):
    print(feature, ":", coefficient)

# 10. Count selected features
selected_features = sum(lasso_model.coef_ != 0)

print("\nTotal Features:", len(X.columns))
print("Selected Features:", selected_features)
print("Removed Features:", len(X.columns) - selected_features)

print("\nIntercept:", lasso_model.intercept_)
