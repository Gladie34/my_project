import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def predict_customer_info(df):
    st.subheader("⚙️ Data Preprocessing")
    st.info("Encoding categorical variables and scaling numeric ones...")
    X = df.drop(columns=["TimeToResolutionDays", "CustomerID"])
    y = df["TimeToResolutionDays"]

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("✅ Preprocessing Completed!")

    # Train/Test split
    st.subheader("🔀 Splitting Dataset")
    st.info("Splitting dataset into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    st.success("✅ Splitting Completed!")

    # Model Training
    st.subheader("🧠 Model Training")
    st.info("Training Random Forest Regressor...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    st.success("✅ Model Training Completed!")

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Evaluation")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R² Score):** {r2:.2f}")

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
        prediction = model.predict(input_scaled)
        st.success(f"Predicted TimeToResolutionDays: {round(prediction[0], 2)} days")

    st.sidebar.markdown("---")
    st.sidebar.write("App developed by Gladie")

