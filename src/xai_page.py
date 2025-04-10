# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.inspection import permutation_importance


#     """
#     Creates the Explainable AI page to provide insights into model predictions
#     and feature importance.
#     """
#     # Create tabs for different XAI approaches
#     tabs = st.tabs(["Feature Importance", "What-If Analysis"])
    
#     # Important features selection (same as in predict_page)
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
    
#     # Prepare data (same preparation steps as in predict_page)
#     X = df[important_features].copy()
#     y = df["TimeToResolutionDays"]
    
#     # Encoder and scaler
#     le = LabelEncoder()
#     scaler = StandardScaler()
#     categorical_cols = X.select_dtypes(include=['object']).columns
    
#     # Encode categorical variables
#     for col in categorical_cols:
#         X[col] = le.fit_transform(X[col])
    
#     # Train/Test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train model (cached to avoid retraining)
#     @st.cache_resource
#     def train_model(X_train, y_train):
#         st.info("Training model for explanations...")
        
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
#     with st.spinner("Preparing model explanations... This may take a moment"):
#         model = train_model(X_train, y_train)
    
#     # Make predictions
#     y_pred = model.predict(X_test)
    
#     # Feature Importance Tab
#     with tabs[0]:
#         st.markdown("### 🏆 Feature Importance Analysis")
        
#         st.markdown("""
#         Feature importance helps identify which features have the greatest impact on the model's predictions.
#         Understanding these key drivers can help insurance professionals focus on the right factors.
#         """)
        
#         # Add info box explaining XAI importance
#         st.markdown("""
#         <div class='info-box'>
#             <h4>Why Explainable AI Matters</h4>
#             <p>Explainable AI allows insurance professionals to:</p>
#             <ul>
#                 <li>Understand and trust the model's predictions</li>
#                 <li>Identify which factors drive resolution times</li>
#                 <li>Explain predictions to customers and stakeholders</li>
#                 <li>Meet regulatory requirements for model transparency</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("#### Built-in Feature Importance")
#             st.markdown("""
#             Random Forest's built-in feature importance measures how much each feature
#             decreases impurity (variance) across all trees in the forest.
#             """)
            
#             # Get feature importances
#             importances = model.feature_importances_
            
#             # Create dataframe for visualization
#             feature_importance_df = pd.DataFrame({
#                 'Feature': important_features,
#                 'Importance': importances
#             }).sort_values('Importance', ascending=False)
            
#             # Plot feature importances
#             fig = px.bar(
#                 feature_importance_df,
#                 x='Importance',
#                 y='Feature',
#                 orientation='h',
#                 title='Built-in Feature Importance',
#                 color='Importance',
#                 color_continuous_scale='viridis'
#             )
            
#             fig.update_layout(height=500)
#             st.plotly_chart(fig, use_container_width=True)
            
#         with col2:
#             st.markdown("#### Permutation Importance")
#             st.markdown("""
#             Permutation importance measures how much model performance decreases when a 
#             feature's values are randomly shuffled. This approach is less biased towards
#             high-cardinality features.
#             """)
            
#             # Calculate permutation importance
#             @st.cache_resource
#             def get_permutation_importance(model, X_test, y_test):
#                 perm_importance = permutation_importance(
#                     model, X_test, y_test, 
#                     n_repeats=5, 
#                     random_state=42
#                 )
#                 return perm_importance
            
#             with st.spinner("Calculating permutation importance..."):
#                 perm_importance = get_permutation_importance(model, X_test, y_test)
            
#             # Create dataframe for visualization
#             perm_importance_df = pd.DataFrame({
#                 'Feature': important_features,
#                 'Importance': perm_importance.importances_mean,
#                 'Std': perm_importance.importances_std
#             }).sort_values('Importance', ascending=False)
            
#             # Plot permutation importances
#             fig = px.bar(
#                 perm_importance_df,
#                 x='Importance',
#                 y='Feature',
#                 orientation='h',
#                 title='Permutation Feature Importance',
#                 color='Importance',
#                 color_continuous_scale='viridis',
#                 error_x='Std'
#             )
            
#             fig.update_layout(height=500)
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Feature importance summary
#         st.markdown("#### Key Insights")
        
#         # Get top 3 features from each method
#         top_builtin = feature_importance_df['Feature'].tolist()[:3]
#         top_perm = perm_importance_df['Feature'].tolist()[:3]
        
#         st.markdown(f"""
#         <div class='success-box'>
#             <h4>Top Features by Built-in Importance:</h4>
#             <ol>
#                 <li>{top_builtin[0]}</li>
#                 <li>{top_builtin[1]}</li>
#                 <li>{top_builtin[2]}</li>
#             </ol>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown(f"""
#         <div class='success-box'>
#             <h4>Top Features by Permutation Importance:</h4>
#             <ol>
#                 <li>{top_perm[0]}</li>
#                 <li>{top_perm[1]}</li>
#                 <li>{top_perm[2]}</li>
#             </ol>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # What-If Analysis Tab
#     with tabs[1]:
#         st.markdown("### 🔮 What-If Analysis Tool")
        
#         st.markdown("""
#         The What-If Analysis tool allows you to explore how changing feature values
#         affects predictions. This helps understand model sensitivity and causal relationships.
#         """)
        
#         # Add info box
#         st.markdown("""
#         <div class='info-box'>
#             <h4>How to Use This Tool</h4>
#             <p>1. Select a data instance or enter custom values</p>
#             <p>2. Adjust feature values using the sliders</p>
#             <p>3. See how the predicted resolution time changes</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Choose how to create the baseline instance
#         baseline_option = st.radio(
#             "Select baseline instance:",
#             ["Random from dataset", "Average values"]
#         )
        
#         # Initialize baseline instance
#         if baseline_option == "Random from dataset":
#             # Use a random instance from the dataset
#             random_idx = np.random.randint(0, len(X))
#             baseline_instance = X.iloc[random_idx].to_dict()
#             original_prediction = model.predict([list(baseline_instance.values())])[0]
            
#             st.markdown(f"Selected random instance with predicted resolution time: **{original_prediction:.2f} days**")
            
#         else:  # Average values
#             # Use average values for each feature
#             baseline_instance = {}
#             for feature in important_features:
#                 if X[feature].dtype.kind in 'bifc':  # numeric
#                     baseline_instance[feature] = X[feature].mean()
#                 else:  # categorical
#                     baseline_instance[feature] = X[feature].mode()[0]
            
#             original_prediction = model.predict([list(baseline_instance.values())])[0]
#             st.markdown(f"Using average values with predicted resolution time: **{original_prediction:.2f} days**")
        
#         # Create sliders for changing feature values
#         st.markdown("#### Adjust Feature Values")
        
#         # Display in columns for better layout
#         modified_instance = baseline_instance.copy()
#         col1, col2 = st.columns(2)
        
#         # Create sliders in two columns
#         features_col1 = important_features[:len(important_features)//2]
#         features_col2 = important_features[len(important_features)//2:]
        
#         with col1:
#             for feature in features_col1:
#                 if X[feature].dtype.kind in 'bifc':  # numeric
#                     min_val = float(X[feature].min())
#                     max_val = float(X[feature].max())
#                     step = (max_val - min_val) / 100
                    
#                     modified_instance[feature] = st.slider(
#                         f"{feature}:",
#                         min_value=min_val,
#                         max_value=max_val,
#                         value=float(baseline_instance[feature]),
#                         step=step,
#                         key=f"slider_{feature}"
#                     )
#                 else:  # categorical
#                     unique_values = X[feature].unique().tolist()
#                     modified_instance[feature] = st.selectbox(
#                         f"{feature}:",
#                         options=unique_values,
#                         index=unique_values.index(baseline_instance[feature]) if baseline_instance[feature] in unique_values else 0,
#                         key=f"select_{feature}"
#                     )
        
#         with col2:
#             for feature in features_col2:
#                 if X[feature].dtype.kind in 'bifc':  # numeric
#                     min_val = float(X[feature].min())
#                     max_val = float(X[feature].max())
#                     step = (max_val - min_val) / 100
                    
#                     modified_instance[feature] = st.slider(
#                         f"{feature}:",
#                         min_value=min_val,
#                         max_value=max_val,
#                         value=float(baseline_instance[feature]),
#                         step=step,
#                         key=f"slider_{feature}"
#                     )
#                 else:  # categorical
#                     unique_values = X[feature].unique().tolist()
#                     modified_instance[feature] = st.selectbox(
#                         f"{feature}:",
#                         options=unique_values,
#                         index=unique_values.index(baseline_instance[feature]) if baseline_instance[feature] in unique_values else 0,
#                         key=f"select_{feature}"
#                     )
        
#         # Calculate new prediction
#         new_prediction = model.predict([list(modified_instance.values())])[0]
        
#         # Display comparison
#         st.markdown("#### Prediction Comparison")
        
#         col1, col2, col3 = st.columns([2, 2, 3])
        
#         with col1:
#             st.metric(
#                 label="Original Prediction",
#                 value=f"{original_prediction:.2f} days"
#             )
        
#         with col2:
#             st.metric(
#                 label="New Prediction",
#                 value=f"{new_prediction:.2f} days",
#                 delta=f"{new_prediction - original_prediction:.2f} days"
#             )
        
#         with col3:
#             # Calculate percent change
#             percent_change = ((new_prediction - original_prediction) / original_prediction) * 100
            
#             if percent_change < 0:
#                 st.success(f"🏎️ Resolution time decreased by {abs(percent_change):.1f}%")
#             elif percent_change > 0:
#                 st.warning(f"⏳ Resolution time increased by {percent_change:.1f}%")
#             else:
#                 st.info("🔄 No change in resolution time")
        
#         # Visualize the comparison
#         fig = go.Figure()
        
#         fig.add_trace(go.Bar(
#             x=['Original', 'New'],
#             y=[original_prediction, new_prediction],
#             marker_color=['#3498db', '#e74c3c']
#         ))
        
#         avg_time = df["TimeToResolutionDays"].mean()
#         fig.add_hline(
#             y=avg_time,
#             line_dash="dash",
#             line_color="gray",
#             annotation_text="Dataset Average"
#         )
        
#         fig.update_layout(
#             title="Comparing Predictions",
#             xaxis_title="Scenario",
#             yaxis_title="Resolution Time (Days)",
#             height=400
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Feature contribution to the change
#         st.markdown("#### Feature Contributions to Change")
        
#         # Calculate approximate contributions
#         contributions = {}
        
#         for feature in important_features:
#             # Create a new instance with only this feature changed
#             test_instance = baseline_instance.copy()
#             test_instance[feature] = modified_instance[feature]
            
#             # Predict
#             test_prediction = model.predict([list(test_instance.values())])[0]
            
#             # Calculate contribution
#             contribution = test_prediction - original_prediction
#             contributions[feature] = contribution
        
#         # Create dataframe for visualization
#         contribution_df = pd.DataFrame({
#             'Feature': list(contributions.keys()),
#             'Contribution': list(contributions.values())
#         }).sort_values('Contribution', key=abs, ascending=False)
        
#         # Add colorscale
#         fig = px.bar(
#             contribution_df,
#             x='Feature',
#             y='Contribution',
#             title='Approximate Feature Contributions to Prediction Change',
#             color='Contribution',
#             color_continuous_scale='RdBu_r'
#         )
        
#         fig.update_layout(height=500)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Final insights
#         st.markdown("#### AI-Generated Insights")
        
#         # Simple insights based on feature changes
#         major_contributors = contribution_df.head(3)
        
#         insights = """
#         <div class='success-box'>
#             <h4>Key Insights</h4>
#             <ul>
#         """
        
#         for _, row in major_contributors.iterrows():
#             feature = row['Feature']
#             contribution = row['Contribution']
            
#             if contribution < 0:
#                 insights += f"<li>Decreasing <b>{feature}</b> reduced resolution time by {abs(contribution):.2f} days</li>"
#             elif contribution > 0:
#                 insights += f"<li>Increasing <b>{feature}</b> increased resolution time by {contribution:.2f} days</li>"
        
#         insights += """
#             </ul>
#             <h4>Recommendations</h4>
#             <ul>
#         """
        
#         # Add recommendations based on the insights
#         if new_prediction < original_prediction:
#             insights += "<li>This combination of factors could help reduce resolution times for similar claims</li>"
            
#             # Find the most impactful negative contributor
#             most_impactful = contribution_df[contribution_df['Contribution'] < 0].iloc[0] if len(contribution_df[contribution_df['Contribution'] < 0]) > 0 else None
#             if most_impactful is not None:
#                 insights += f"<li>Focus on optimizing <b>{most_impactful['Feature']}</b> for maximum impact</li>"
#         else:
#             insights += "<li>Watch for these risk factors that may increase resolution times</li>"
            
#             # Find the most impactful positive contributor
#             most_impactful = contribution_df[contribution_df['Contribution'] > 0].iloc[0] if len(contribution_df[contribution_df['Contribution'] > 0]) > 0 else None
#             if most_impactful is not None:
#                 insights += f"<li>Monitor <b>{most_impactful['Feature']}</b> closely as it significantly impacts resolution time</li>"
        
#         insights += """
#             </ul>
#         </div>
#         """
        
#         st.markdown(insights, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance

def explain_model(df, model):
    st.title("Explainable AI (XAI)")

    tabs = st.tabs(["SHAP Analysis", "Feature Importance", "What-If Analysis"])

    important_features = [
        "CustomerFeedbackScore", "PolicyStartYear", "PolicyDiscounts",
        "PolicyUpgradesLastYear", "Age", "County", "Income",
        "PolicyDurationMonths", "NumberOfInquiriesLastYear", "CustomerSatisfaction"
    ]

    X = df[important_features].copy()
    y = df["TimeToResolutionDays"]

    # Encode categorical
    le_dict = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    # Standard scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------- SHAP ANALYSIS TAB ---------------------------- #
    with tabs[0]:
        st.subheader("SHAP Summary Plot")
        st.markdown("This explains the impact of each feature on individual predictions.")

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.summary_plot(shap_values, X, plot_type="bar")
        st.pyplot(bbox_inches='tight')

        st.subheader("SHAP Dependence Plot")
        feature = st.selectbox("Select feature for SHAP dependence plot", X.columns)
        shap.dependence_plot(feature, shap_values.values, X, interaction_index=None)
        st.pyplot(bbox_inches='tight')

    # ------------------------ BUILT-IN IMPORTANCE TAB ------------------------ #
    with tabs[1]:
        st.subheader("Feature Importance")
        importances = model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(feature_df, x="Importance", y="Feature", orientation="h",
                     title="Built-in Feature Importance", color="Importance",
                     color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Permutation Importance")
        with st.spinner("Calculating permutation importance..."):
            perm = permutation_importance(model, X, y, n_repeats=5, random_state=42)

        perm_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": perm.importances_mean,
            "Std": perm.importances_std
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(perm_df, x="Importance", y="Feature", orientation="h",
                      title="Permutation Feature Importance",
                      error_x="Std", color="Importance",
                      color_continuous_scale="viridis")
        st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------- WHAT-IF TAB ----------------------------- #
    with tabs[2]:
        st.subheader("What-If Analysis")
        st.markdown("Simulate changes in customer or policy features to observe predicted outcome.")

        default_instance = X.iloc[np.random.randint(0, X.shape[0])].copy()
        st.write("Starting with the following instance:")
        st.dataframe(default_instance.to_frame().T)

        user_inputs = {}
        col1, col2 = st.columns(2)
        split = len(X.columns) // 2
        for i, col in enumerate(X.columns):
            col_container = col1 if i < split else col2
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            step = (max_val - min_val) / 100 if max_val > min_val else 1
            user_inputs[col] = col_container.slider(col, min_val, max_val, float(default_instance[col]), step=step)

        modified_instance = pd.DataFrame([user_inputs])
        original_pred = model.predict([default_instance])[0]
        new_pred = model.predict(modified_instance)[0]

        st.metric("Original Prediction", f"{original_pred:.2f} days")
        st.metric("Modified Prediction", f"{new_pred:.2f} days", delta=f"{new_pred - original_pred:.2f} days")

        st.markdown("#### Feature Contribution to Change")
        contributions = {}
        for feature in X.columns:
            temp_instance = default_instance.copy()
            temp_instance[feature] = user_inputs[feature]
            contrib_pred = model.predict([temp_instance])[0]
            contributions[feature] = contrib_pred - original_pred

        contrib_df = pd.DataFrame({
            "Feature": list(contributions.keys()),
            "Contribution": list(contributions.values())
        }).sort_values(by="Contribution", key=abs, ascending=False)

        fig3 = px.bar(contrib_df, x="Feature", y="Contribution",
                      title="Feature Contributions to Change",
                      color="Contribution", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig3, use_container_width=True)
