import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Paths to save model and preprocessing assets
MODEL_PATH = "saved_xgb_model.pkl"
ENCODER_SCALER_PATH = "encoder_scaler.pkl"

# Load the dataset
print("Loading data...")
data = pd.read_csv("./data/Insurance_data.csv")
print("✅ Data loaded successfully.")
print(f"Columns in dataset: {data.columns.tolist()}")

# Define target and features
target = "TimeToResolutionDays"

# Check if target exists in the dataset
if target not in data.columns:
    raise ValueError(f"Target column '{target}' not found in the dataset. Available columns: {data.columns.tolist()}")

# Remove any columns that don't exist
important_features = []
for col in data.columns:
    if col != target:  # Skip the target column
        important_features.append(col)

print(f"Features to be used: {important_features}")

X = data[important_features].copy()
y = data[target].copy()
print(f"✅ Target: {target}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Encode categorical features if needed
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le
    print(f"✅ Encoded column: {col}")

# Save the feature names (important for prediction consistency)
feature_names = X.columns.tolist()
print(f"✅ Features used for training: {len(feature_names)}")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Scaling complete.")

# Train the XGBoost model
print("Training model...")
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("✅ Model training complete.")

# Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance:")
print(f"   - Mean Absolute Error: {mae:.2f} days")
print(f"   - Root Mean Squared Error: {rmse:.2f} days")
print(f"   - R² Score: {r2:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(len(importances)), importances[indices], color='skyblue')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")
print("✅ Feature importance plot saved.")

# Save the model
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

# Save encoders, scaler, and feature names together
joblib.dump({
    'encoders': encoders,
    'scaler': scaler,
    'features': feature_names
}, ENCODER_SCALER_PATH)
print(f"💾 Encoders, scaler & features saved to {ENCODER_SCALER_PATH}")