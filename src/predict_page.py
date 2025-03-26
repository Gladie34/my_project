import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

def predict_customer_info(df):
    st.subheader("⚙️ Data Preprocessing")
    st.info("Encoding categorical variables and scaling numeric ones...")
    X = df.drop(columns=["TimeToResolutionDays", "CustomerID"])
    important_features = [
    "CustomerFeedbackScore",
    "PolicyStartYear",
    "PolicyDiscounts",
    "PolicyUpgradesLastYear",
    "Age",
    "County",
    "Income",
    "PolicyDurationMonths",
    "NumberOfInquiriesLastYear",
    "CustomerSatisfaction"]
    X=X[important_features]
    y = df["TimeToResolutionDays"]

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #print(X_scaled)
    st.success("✅ Preprocessing Completed!")

    # Train/Test split
    st.subheader("🔀 Splitting Dataset")
    st.info("Splitting dataset into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.success("✅ Splitting Completed!")
    #print(X_train.columns)
    # Model Training

    st.subheader("🧠 Model Training")

    xgb_param_dist = {
    'n_estimators': [750, 800, 900, 1000, 1100],
    'learning_rate': [0.005, 0.008, 0.01, 0.012],
    'max_depth': [4, 5, 6, 7],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    xgb_random_search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=xgb_param_dist,
    n_iter=20,  # Try 20 random combinations
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
    )

    st.info("Training XGBRegressor...")
    model = xgb_random_search#RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    best_xgb = model.best_estimator_
    st.success("✅ Model Training Completed!")

    # Evaluation
    y_pred = model.predict(X_test)
    #print(y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R² Score):** {r2:.2f}")

    st.subheader(" Model Explanability") 

    explainer = shap.Explainer(best_xgb)

    # Compute SHAP values
    shap_values = explainer(X_test)

    # Display SHAP summary plot
    st.subheader("Feature Importance (SHAP Values)")
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=X.columns), plot_type="bar", show=False)
    st.pyplot(fig)


    # Prediction Section
    st.subheader("🔮 Make Predictions")

    st.markdown("Input new data below to predict TimeToResolutionDays:")

    input_data = {}
    for col in X.columns:
        if col in categorical_cols:
            options = df[col].unique().tolist()
            input_data[col] = st.selectbox(f"{col}", options)
        else:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    # Predict button
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        st.info("Processing input data and generating prediction...")
        # Encode categorical
        for col in categorical_cols:
            input_df[col] = le.fit(df[col]).transform(input_df[col])
        # Scale
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_df)
        st.success(f"Predicted TimeToResolutionDays: {round(prediction[0], 2)} days")

