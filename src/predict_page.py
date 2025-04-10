# import streamlit as st
# import numpy as np
# import pandas as pd
# #import shap
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBRegressor

# def predict_customer_info(df):
#     st.subheader("⚙️ Data Preprocessing")
#     st.info("Encoding categorical variables and scaling numeric ones...")
#     X = df.drop(columns=["TimeToResolutionDays", "CustomerID"])
#     important_features = [
#     "CustomerFeedbackScore",
#     "PolicyStartYear",
#     "PolicyDiscounts",
#     "PolicyUpgradesLastYear",
#     "Age",
#     "County",
#     "Income",
#     "PolicyDurationMonths",
#     "NumberOfInquiriesLastYear",
#     "CustomerSatisfaction"]
#     X=X[important_features]
#     y = df["TimeToResolutionDays"]

#     # Encode categorical variables
#     categorical_cols = X.select_dtypes(include=['object']).columns
#     le = LabelEncoder()
#     for col in categorical_cols:
#         X[col] = le.fit_transform(X[col])

#     # Feature Scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     #print(X_scaled)
#     st.success("✅ Preprocessing Completed!")

#     # Train/Test split
#     st.subheader("🔀 Splitting Dataset")
#     st.info("Splitting dataset into Train and Test sets...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     st.success("✅ Splitting Completed!")
#     #print(X_train.columns)
#     # Model Training

#     st.subheader("🧠 Model Training")

#     xgb_param_dist = {
#     'n_estimators': [750, 800, 900, 1000, 1100],
#     'learning_rate': [0.005, 0.008, 0.01, 0.012],
#     'max_depth': [4, 5, 6, 7],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
#     }
    
#     xgb_random_search = RandomizedSearchCV(
#     estimator=XGBRegressor(random_state=42),
#     param_distributions=xgb_param_dist,
#     n_iter=20,  # Try 20 random combinations
#     scoring='neg_root_mean_squared_error',
#     cv=5,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
#     )

#     st.info("Training XGBRegressor...")
#     model = xgb_random_search#RandomForestRegressor(random_state=42)
#     model.fit(X_train, y_train)
#     best_xgb = model.best_estimator_
#     st.success("✅ Model Training Completed!")

#     # Evaluation
#     y_pred = model.predict(X_test)
#     #print(y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)

#     st.subheader("📈 Model Evaluation")
#     st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
#     st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
#     st.write(f"**R-squared (R² Score):** {r2:.2f}")

#     st.subheader(" Model Explanability") 

#     explainer = shap.Explainer(best_xgb)

#     # Compute SHAP values
#     shap_values = explainer(X_test)

#     # Display SHAP summary plot
#     st.subheader("Feature Importance (SHAP Values)")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=X.columns), plot_type="bar", show=False)
#     st.pyplot(fig)


#     # Prediction Section
#     st.subheader("🔮 Make Predictions")

#     st.markdown("Input new data below to predict TimeToResolutionDays:")

#     input_data = {}
#     for col in X.columns:
#         if col in categorical_cols:
#             options = df[col].unique().tolist()
#             input_data[col] = st.selectbox(f"{col}", options)
#         else:
#             input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

#     # Predict button
#     if st.button("Predict"):
#         input_df = pd.DataFrame([input_data])
#         st.info("Processing input data and generating prediction...")
#         # Encode categorical
#         for col in categorical_cols:
#             input_df[col] = le.fit(df[col]).transform(input_df[col])
#         # Scale
#         input_scaled = scaler.transform(input_df)
#         prediction = model.predict(input_df)
# #         st.success(f"Predicted TimeToResolutionDays: {round(prediction[0], 2)} days")


# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBRegressor
# import time

# def predict_customer_info(df):
#     # Create tabs for different stages of the modeling process
#     tabs = st.tabs(["Data Preparation", "Model Training", "Model Evaluation", "Make Predictions"])
    
#     # Important features selection
#     important_features = [
#         "CustomerFeedbackScore",
#         "PolicyStartYear",
#         "PolicyDiscounts",
#         "PolicyUpgradesLastYear",
#         "Age",
#         "County",
#         "Income",
#         "PolicyDurationMonths",
#         "NumberOfInquiriesLastYear",
#         "CustomerSatisfaction"
#     ]
    
#     # Prepare data
#     X = df[important_features].copy()  # Use only selected features
#     y = df["TimeToResolutionDays"]
    
#     # Encoder and scaler
#     le_dict = {}  # Dictionary to store fitted LabelEncoders
#     scaler = StandardScaler()
#     categorical_cols = X.select_dtypes(include=['object']).columns
    
#     # Encode categorical variables - store the fitted encoders
#     for col in categorical_cols:
#         le = LabelEncoder()
#         X[col] = le.fit_transform(X[col])
#         le_dict[col] = le  # Store the fitted encoder
    
#     # Train/Test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Feature Scaling - fit on all data (this is important for reusing later)
#     scaler.fit(X)
#     X_train_scaled = scaler.transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Setup model with caching to avoid retraining
#     @st.cache_resource
#     def train_model(X_train, y_train):
#         st.info("Training Random Forest model...")
        
#         # Create the Random Forest model with optimized parameters
#         rf_model = RandomForestRegressor(
#             n_estimators=100,
#             max_depth=10,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             max_features='sqrt',
#             n_jobs=-1,
#             random_state=42
#         )
        
#         # Fit the model
#         rf_model.fit(X_train, y_train)
        
#         return rf_model
    
#     # Train model
#     with st.spinner("Training model... This may take a moment"):
#         model = train_model(X_train, y_train)
    
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     # Calculate metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    
#     # Data Preparation Tab
#     with tabs[0]:
#         st.markdown("### 🔄 Data Preparation")
        
#         st.info("This section shows how the data is prepared for model training.")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("#### Selected Features")
#             st.code(f"Features used: {', '.join(important_features)}")
            
#             st.markdown("#### Feature Information")
#             feature_info = pd.DataFrame({
#                 'Feature': important_features,
#                 'Type': [df[feat].dtype for feat in important_features],
#                 'Missing Values': [df[feat].isnull().sum() for feat in important_features],
#                 'Unique Values': [df[feat].nunique() for feat in important_features]
#             })
#             st.dataframe(feature_info, use_container_width=True)
        
#         with col2:
#             st.markdown("#### Preprocessing Steps")
#             st.markdown("""
#             1. **Feature Selection**: Selected the most important features for prediction
#             2. **Categorical Encoding**: Converted categorical variables to numeric using Label Encoding
#             3. **Feature Scaling**: Standardized numeric features to have mean=0 and std=1
#             4. **Train-Test Split**: Split data into 80% training and 20% testing sets
#             """)
            
#             st.markdown("#### Dataset Split")
#             split_data = {
#                 'Dataset': ['Training Set', 'Testing Set', 'Total'],
#                 'Size': [X_train.shape[0], X_test.shape[0], X.shape[0]],
#                 'Percentage': [
#                     f"{X_train.shape[0]/X.shape[0]*100:.1f}%", 
#                     f"{X_test.shape[0]/X.shape[0]*100:.1f}%",
#                     "100%"
#                 ]
#             }
#             st.dataframe(pd.DataFrame(split_data), use_container_width=True)
    
#     # Model Training Tab
#     with tabs[1]:
#         st.markdown("### 🧠 Model Training")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("#### Model Type")
#             st.markdown("""
#             **Random Forest Regressor** is used for this prediction task. Random Forest is an ensemble learning
#             method that builds multiple decision trees and merges their predictions.
            
#             Benefits:
#             - High prediction accuracy
#             - Robust to outliers
#             - Handles non-linear relationships well
#             - Built-in feature importance
#             - Less prone to overfitting
#             """)
            
#             st.markdown("#### Model Parameters")
#             params_df = pd.DataFrame({
#                 'Parameter': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
#                 'Value': [100, 10, 5, 2, 'sqrt'],
#                 'Description': [
#                     'Number of trees in the forest',
#                     'Maximum depth of each tree',
#                     'Minimum samples required to split a node',
#                     'Minimum samples required at each leaf node',
#                     'Number of features to consider for best split'
#                 ]
#             })
#             st.dataframe(params_df, use_container_width=True)
        
#         with col2:
#             st.markdown("#### Training Process")
#             st.markdown("""
#             The Random Forest model was trained using the following process:
            
#             1. The data was split into training (80%) and testing (20%) sets
#             2. Categorical variables were encoded
#             3. The model was trained on the training set with optimized parameters
#             4. Performance was evaluated on the test set
#             """)
            
#             st.markdown("#### Cross-Validation Results")
#             cv_scores = cross_val_score(
#                 model, 
#                 X_train, 
#                 y_train, 
#                 cv=5, 
#                 scoring='neg_root_mean_squared_error'
#             )
            
#             cv_df = pd.DataFrame({
#                 'Fold': range(1, 6),
#                 'RMSE': -cv_scores
#             })
            
#             fig = px.bar(
#                 cv_df,
#                 x='Fold',
#                 y='RMSE',
#                 title='Cross-Validation RMSE by Fold',
#                 color='RMSE',
#                 color_continuous_scale='RdYlGn_r'
#             )
#             st.plotly_chart(fig, use_container_width=True)
    
#     # Model Evaluation Tab
#     with tabs[2]:
#         st.markdown("### 📊 Model Evaluation")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("#### Performance Metrics")
            
#             metrics_df = pd.DataFrame({
#                 'Metric': ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)'],
#                 'Value': [f"{mae:.2f} days", f"{rmse:.2f} days"],
#                 'Description': [
#                     'Average absolute difference between predicted and actual values',
#                     'Root of the average squared difference (penalizes large errors)'
#                 ]
#             })
            
#             st.dataframe(metrics_df, use_container_width=True)
            
#             # Visualize metrics
#             cols = st.columns(3)
#             with cols[0]:
#                 st.metric("MAE", f"{mae:.2f} days")
#             with cols[1]:
#                 st.metric("RMSE", f"{rmse:.2f} days")
          
#         with col2:
#             st.markdown("#### Predicted vs Actual")
            
#             # Create dataframe for comparison
#             comparison_df = pd.DataFrame({
#                 'Actual': y_test,
#                 'Predicted': y_pred,
#                 'Error': y_test - y_pred
#             })
            
#             # Plot predicted vs actual
#             fig = px.scatter(
#                 comparison_df,
#                 x='Actual',
#                 y='Predicted',
#                 color='Error',
#                 color_continuous_scale='RdBu_r',
#                 title='Predicted vs Actual Resolution Time',
#                 labels={'Actual': 'Actual Time (Days)', 'Predicted': 'Predicted Time (Days)'}
#             )
            
#             # Add perfect prediction line
#             min_val = min(comparison_df['Actual'].min(), comparison_df['Predicted'].min())
#             max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
#             fig.add_trace(
#                 go.Scatter(
#                     x=[min_val, max_val],
#                     y=[min_val, max_val],
#                     mode='lines',
#                     line=dict(color='black', dash='dash'),
#                     name='Perfect Prediction'
#                 )
#             )
            
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Error distribution
#         st.markdown("#### Error Distribution")
        
#         error_fig = px.histogram(
#             comparison_df,
#             x='Error',
#             nbins=30,
#             marginal="box",
#             title="Distribution of Prediction Errors",
#             labels={'Error': 'Prediction Error (Days)'},
#             color_discrete_sequence=['#e74c3c']
#         )
        
#         st.plotly_chart(error_fig, use_container_width=True)
    
#     # Make Predictions Tab
#     with tabs[3]:
#         st.markdown("### 🔮 Make Predictions for New Claims")
        
#         st.info("Enter customer and policy information to predict the expected resolution time.")
        
#         # Create a form for better user experience
#         with st.form("prediction_form"):
#             st.markdown("#### Enter Claim Details")
            
#             # Organize input fields in columns
#             col1, col2, col3 = st.columns(3)
            
#             input_data = {}
            
#             with col1:
#                 input_data["Age"] = st.number_input("Customer Age", 
#                                                   min_value=18, 
#                                                   max_value=100, 
#                                                   value=int(df["Age"].mean()),
#                                                   help="Age of the customer in years")
                
#                 input_data["CustomerFeedbackScore"] = st.slider("Customer Feedback Score", 
#                                                               min_value=1, 
#                                                               max_value=10, 
#                                                               value=7,
#                                                               help="Customer's previous feedback score (1-10)")
                
#                 input_data["CustomerSatisfaction"] = st.slider("Customer Satisfaction", 
#                                                             min_value=1, 
#                                                             max_value=10, 
#                                                             value=7,
#                                                             help="Customer's overall satisfaction score (1-10)")
                
#                 input_data["NumberOfInquiriesLastYear"] = st.number_input("Number of Inquiries Last Year", 
#                                                                          min_value=0, 
#                                                                          max_value=50, 
#                                                                          value=3,
#                                                                          help="How many inquiries the customer made last year")
            
#             with col2:
#                 input_data["PolicyStartYear"] = st.number_input("Policy Start Year", 
#                                                               min_value=2000, 
#                                                               max_value=2025, 
#                                                               value=2020,
#                                                               help="Year when the policy was initiated")
                
#                 input_data["PolicyDurationMonths"] = st.number_input("Policy Duration (Months)", 
#                                                                    min_value=1, 
#                                                                    max_value=240, 
#                                                                    value=24,
#                                                                    help="How long the policy has been active (in months)")
                
#                 input_data["PolicyDiscounts"] = st.number_input("Policy Discounts (%)", 
#                                                               min_value=0, 
#                                                               max_value=50, 
#                                                               value=10,
#                                                               help="Percentage discount applied to the policy")
                
#                 input_data["PolicyUpgradesLastYear"] = st.number_input("Policy Upgrades Last Year", 
#                                                                      min_value=0, 
#                                                                      max_value=10, 
#                                                                      value=1,
#                                                                      help="Number of times the policy was upgraded last year")
            
#             with col3:
#                 # For categorical variables, show only unique values in the dataset
#                 unique_incomes = df["Income"].unique().tolist()
#                 input_data["Income"] = st.selectbox("Income Category", 
#                                                   options=sorted(unique_incomes),
#                                                   index=0,
#                                                   help="Income bracket of the customer")
                
#                 # County selection with search
#                 unique_counties = sorted(df["County"].unique().tolist())
#                 input_data["County"] = st.selectbox("County", 
#                                                   options=unique_counties,
#                                                   index=0,
#                                                   help="Customer's county of residence")
            
#             # Submit button
#             submitted = st.form_submit_button("Predict Resolution Time", type="primary")
        
#         # Process form submission
#         if submitted:
#             # Create a DataFrame from the input
#             input_df = pd.DataFrame([input_data])
            
#             # Display a spinner while processing
#             with st.spinner("Calculating prediction..."):
#                 # Add a slight delay for UX purposes
#                 time.sleep(0.5)
                
#                 # Encode categorical variables - THIS IS THE FIX
#                 for col in categorical_cols:
#                     if col in input_df.columns:
#                         # Use transform only (not fit_transform)
#                         if col in le_dict:
#                             # Get the encoder that was already fitted
#                             le = le_dict[col]
#                             # Transform the new data (don't fit again)
#                             try:
#                                 input_df[col] = le.transform(input_df[col])
#                             except ValueError:
#                                 # Handle values not seen during training
#                                 st.warning(f"Warning: The value for {col} wasn't in the training data. Using most common value instead.")
#                                 # Fall back to the most common value
#                                 most_common_value = df[col].mode()[0]
#                                 most_common_encoded = le.transform([most_common_value])[0]
#                                 input_df[col] = most_common_encoded
                
#                 # Make sure columns are in the same order as during training
#                 input_df = input_df[X.columns]
                
#                 # Scale features using the already fitted scaler
#                 input_scaled = scaler.transform(input_df)
                
#                 # Make prediction
#                 # We can use either the raw DataFrame or the scaled version depending on how the model was trained
#                 prediction = model.predict(input_df)[0]
                
#                 # Display prediction with nice formatting
#                 st.markdown("#### Prediction Result")
                
#                 col1, col2 = st.columns([1, 2])
                
#                 with col1:
#                     st.metric(
#                         label="Predicted Resolution Time",
#                         value=f"{prediction:.1f} days",
#                     )
                
#                 with col2:
#                     # Compare to average
#                     avg_time = df["TimeToResolutionDays"].mean()
#                     diff = prediction - avg_time
                    
#                     if diff < 0:
#                         st.success(f"⚡ This claim is expected to be resolved {abs(diff):.1f} days faster than average!")
#                     elif diff > 0:
#                         st.warning(f"⏳ This claim may take {diff:.1f} days longer than average to resolve.")
#                     else:
#                         st.info("📊 This claim is expected to take about the average time to resolve.")
                
#                 # Confidence interval (simplified approach)
#                 lower_bound = max(0, prediction - rmse)
#                 upper_bound = prediction + rmse
                
#                 st.markdown(f"**Estimated range:** Between {lower_bound:.1f} and {upper_bound:.1f} days")
                
#                 # Visualization
#                 fig = go.Figure()
                
#                 # Add average line
#                 fig.add_vline(x=avg_time, line_width=2, line_dash="dash", line_color="gray", annotation_text="Average")
                
#                 # Add prediction marker
#                 fig.add_trace(go.Scatter(
#                     x=[prediction],
#                     y=[0.5],
#                     mode="markers",
#                     marker=dict(size=15, color="red"),
#                     name="Prediction"
#                 ))
                
#                 # Add prediction range
#                 fig.add_shape(
#                     type="rect",
#                     x0=lower_bound,
#                     x1=upper_bound,
#                     y0=0.4,
#                     y1=0.6,
#                     fillcolor="rgba(231, 76, 60, 0.2)",
#                     line=dict(color="rgba(231, 76, 60, 0.5)")
#                 )
                
#                 fig.update_layout(
#                     title="Prediction Visualization",
#                     xaxis_title="Time to Resolution (Days)",
#                     yaxis_visible=False,
#                     height=200,
#                     margin=dict(l=20, r=20, t=40, b=20),
#                     showlegend=False
#                 )
                
#             st.plotly_chart(fig, use_container_width=True)

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
