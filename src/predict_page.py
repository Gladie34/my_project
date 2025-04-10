
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

def predict_customer_info(df):
    tabs = st.tabs(["Data Preparation", "Model Training", "Model Evaluation", "Make Predictions"])

    important_features = [
        "CustomerFeedbackScore", "PolicyStartYear", "PolicyDiscounts", "PolicyUpgradesLastYear",
        "Age", "County", "Income", "PolicyDurationMonths", "NumberOfInquiriesLastYear", "CustomerSatisfaction"
    ]

    X = df[important_features].copy()
    y = df["TimeToResolutionDays"]

    le_dict = {}
    scaler = StandardScaler()
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler.fit(X)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        return model

    with st.spinner("Training model..."):
        model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Tab 0: Data Prep
    with tabs[0]:
        st.subheader("🔄 Data Preparation")
        st.markdown("#### Selected Features")
        st.code(", ".join(important_features))

        feature_info = pd.DataFrame({
            'Feature': important_features,
            'Type': [df[feat].dtype for feat in important_features],
            'Missing': [df[feat].isnull().sum() for feat in important_features],
            'Unique': [df[feat].nunique() for feat in important_features]
        })
        st.dataframe(feature_info)

        st.markdown("#### Preprocessing Steps")
        st.markdown("""
        1. Selected relevant features  
        2. Encoded categorical features  
        3. Scaled numeric features  
        4. Split into train/test (80/20)
        """)

    # Tab 1: Model Training
    with tabs[1]:
        st.subheader("🧠 Model Training")
        st.markdown("Random Forest Regressor used with the following parameters:")
        params = pd.DataFrame({
            'Parameter': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
            'Value': [100, 10, 5, 2, 'sqrt']
        })
        st.dataframe(params)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        st.plotly_chart(px.bar(
            x=list(range(1, 6)), y=-cv_scores, labels={'x': 'Fold', 'y': 'RMSE'},
            title="Cross-Validation RMSE by Fold", color=-cv_scores, color_continuous_scale='Viridis'
        ))

    # Tab 2: Evaluation
    with tabs[2]:
        st.subheader("📊 Model Evaluation")
        st.metric("MAE", f"{mae:.2f} days")
        st.metric("RMSE", f"{rmse:.2f} days")

        comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Error': y_test - y_pred})
        fig1 = px.scatter(comparison_df, x='Actual', y='Predicted', color='Error', title="Actual vs Predicted")
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                  mode="lines", line=dict(dash="dash", color="black")))
        st.plotly_chart(fig1)

        fig2 = px.histogram(comparison_df, x='Error', nbins=30, marginal='box',
                            title="Prediction Error Distribution", color_discrete_sequence=['#e74c3c'])
        st.plotly_chart(fig2)

    # Tab 3: Make Predictions
    with tabs[3]:
        st.subheader("🔮 Make Predictions")
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            input_data = {}

            with col1:
                input_data["Age"] = st.number_input("Age", 18, 100, int(df["Age"].mean()))
                input_data["CustomerFeedbackScore"] = st.slider("Feedback Score", 1, 10, 7)
                input_data["CustomerSatisfaction"] = st.slider("Satisfaction", 1, 10, 7)
                input_data["NumberOfInquiriesLastYear"] = st.number_input("Inquiries Last Year", 0, 50, 3)

            with col2:
                input_data["PolicyStartYear"] = st.number_input("Policy Start Year", 2000, 2025, 2020)
                input_data["PolicyDurationMonths"] = st.number_input("Duration (Months)", 1, 240, 24)
                input_data["PolicyDiscounts"] = st.number_input("Policy Discounts (%)", 0, 50, 10)
                input_data["PolicyUpgradesLastYear"] = st.number_input("Upgrades Last Year", 0, 10, 1)

            with col3:
                input_data["Income"] = st.selectbox("Income", sorted(df["Income"].unique()))
                input_data["County"] = st.selectbox("County", sorted(df["County"].unique()))

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([input_data])
            for col in categorical_cols:
                if col in input_df:
                    try:
                        input_df[col] = le_dict[col].transform(input_df[col])
                    except:
                        fallback = df[col].mode()[0]
                        input_df[col] = le_dict[col].transform([fallback])

            input_df = input_df[X.columns]
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_df)[0]

            avg_time = df["TimeToResolutionDays"].mean()
            diff = prediction - avg_time
            lower = max(0, prediction - rmse)
            upper = prediction + rmse

            st.metric("Predicted Time", f"{prediction:.1f} days")
            st.markdown(f"**Estimated range:** {lower:.1f} – {upper:.1f} days")

            fig = go.Figure()
            fig.add_vline(x=avg_time, line_width=2, line_dash="dash", line_color="gray", annotation_text="Average")
            fig.add_trace(go.Scatter(x=[prediction], y=[0.5], mode="markers", marker=dict(size=15, color="red")))
            fig.add_shape(type="rect", x0=lower, x1=upper, y0=0.4, y1=0.6,
                          fillcolor="rgba(231,76,60,0.2)", line=dict(color="rgba(231,76,60,0.5)"))
            fig.update_layout(title="Prediction Visualization", height=200, showlegend=False, yaxis_visible=False)
            st.plotly_chart(fig)

    return model  
