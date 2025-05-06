import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import sys
import traceback
from src.utils import load_model_assets

# Add robust error handling
try:
    # Load model and assets
    model, encoders, scaler, feature_names = load_model_assets()
    
    # Try to import SHAP
    try:
        import shap
        SHAP_AVAILABLE = True
    except ImportError:
        SHAP_AVAILABLE = False
        st.warning("SHAP library is not installed. Some interpretation features will be limited.")
    except Exception as e:
        SHAP_AVAILABLE = False
        st.error(f"Error importing SHAP: {str(e)}")
except Exception as e:
    st.error(f"Error loading model assets: {str(e)}")
    traceback.print_exc()

# Function to get readable feature names
def get_readable_feature_names(original_features):
    """Convert feature codes to readable names."""
    # This dictionary maps feature codes to human-readable names
    # Update this with your actual feature mappings
    feature_name_mapping = {
        # Examples - update with your actual feature mappings
        "ClaimAmount": "Claim Amount ($)",
        "Age": "Customer Age",
        "Gender": "Customer Gender",
        "PolicyDuration": "Policy Length (months)",
        "PremiumMonthly": "Premium Amount ($)",
        "PolicyType": "Type of Policy",
        "ClaimComplexity": "Claim Complexity Score",
        "Region": "Geographic Region",
        "DaysToFile": "Days to File Claim",
        "ClaimsRejected": "Number of Rejected Claims",
        "Income": "Customer Income",
        "CustomerID": "Customer ID",
        # Add more mappings as needed
    }
    
    readable_features = []
    for feature in original_features:
        # If we have a specific mapping, use it; otherwise keep the original
        readable_features.append(feature_name_mapping.get(feature, feature))
    
    return readable_features

def explain_model(df, model):
    st.write("### Understanding Model Predictions")
    
    # Debug information
    st.write(f"DataFrame shape: {df.shape}")
    st.write(f"Model type: {type(model).__name__}")
    
    # Error handling for the entire function
    try:
        st.info("This page helps interpret how the model makes decisions for insurance claim resolution time.")
        
        # Get readable feature names
        try:
            readable_feature_names = get_readable_feature_names(feature_names)
        except Exception as e:
            st.error(f"Error getting readable feature names: {str(e)}")
            readable_feature_names = feature_names
        
        tabs = st.tabs(["Feature Importance", "SHAP Values", "What-If Analysis"])
        
        with tabs[0]:
            st.write("#### Feature Importance")
            st.write("These are the most influential factors in determining claim resolution time.")
            
            # Get feature importance from model (assuming XGBoost model)
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'Feature': readable_feature_names,
                        'Importance': importances
                    }).sort_values(by='Importance', ascending=False)
                    
                    fig = px.bar(
                        feature_importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance (Higher = More Impact)',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Model doesn't provide direct feature importance. Try SHAP values instead.")
            except Exception as e:
                st.error(f"Error creating feature importance plot: {str(e)}")
                traceback.print_exc()
        
        with tabs[1]:
            st.write("#### SHAP Values Explanation")
            st.write("SHAP values show how each feature contributes to individual predictions.")
            
            if not SHAP_AVAILABLE:
                st.warning("SHAP library is not installed or incompatible. Please install with: `pip install shap`")
                st.info("If you've already installed SHAP, there might be a compatibility issue with your environment.")
            else:
                # Simplified SHAP approach with error handling
                if st.button("Generate SHAP Analysis"):
                    with st.spinner("Calculating SHAP values (this may take a while)..."):
                        try:
                            # Get a sample of the data (for performance)
                            sample_size = min(50, len(df))
                            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
                            
                            # Prepare the data
                            X = sample_df.copy()
                            
                            # Handle categorical features
                            for col in encoders:
                                if col in X.columns:
                                    X[col] = encoders[col].transform(X[col])
                            
                            # Ensure only appropriate columns are used
                            X = X[feature_names]
                            
                            # Scale the data
                            X_scaled = scaler.transform(X)
                            
                            # Create DataFrame with readable feature names
                            X_display = pd.DataFrame(X_scaled, columns=readable_feature_names)
                            
                            # Use TreeExplainer for robustness
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_scaled)
                            
                            # Convert to DataFrame for visualization
                            shap_df = pd.DataFrame(shap_values, columns=feature_names)
                            
                            # Calculate average absolute SHAP value for each feature
                            feature_importance = pd.DataFrame({
                                'Feature': readable_feature_names,
                                'Importance': np.abs(shap_df).mean().values
                            }).sort_values('Importance', ascending=False)
                            
                            # Create a bar chart with Plotly (more robust than matplotlib)
                            fig = px.bar(
                                feature_importance,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='SHAP Feature Importance',
                                color='Importance',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating SHAP analysis: {str(e)}")
                            st.info("Full error details (helpful for debugging):")
                            st.code(traceback.format_exc())
                            
                            # Fallback to basic feature importance
                            st.write("#### Fallback: Basic Feature Importance")
                            if hasattr(model, 'feature_importances_'):
                                importances = model.feature_importances_
                                feature_importance_df = pd.DataFrame({
                                    'Feature': readable_feature_names,
                                    'Importance': importances
                                }).sort_values(by='Importance', ascending=False)
                                
                                fig = px.bar(
                                    feature_importance_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='Feature Importance (Higher = More Impact)',
                                    color='Importance',
                                    color_continuous_scale='viridis'
                                )
                                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            st.write("#### What-If Analysis")
            st.write("Explore how changing input values affects predictions.")
            
            try:
                st.subheader("Select Features to Compare")
                
                # Let user select features to vary
                cols = st.columns(2)
                with cols[0]:
                    feature1 = st.selectbox("Feature 1:", feature_names, index=0)
                with cols[1]:
                    # Default to a different feature for second selection
                    default_idx = 1 if len(feature_names) > 1 else 0
                    feature2 = st.selectbox("Feature 2:", feature_names, index=default_idx)
                
                if feature1 == feature2:
                    st.warning("Please select two different features.")
                else:
                    # Create inputs for other features
                    st.subheader("Set Values for Other Features")
                    other_inputs = {}
                    
                    # Create 2-column layout for controls
                    col_features = st.columns(2)
                    
                    # Add other features
                    feature_idx = 0
                    for i, feature in enumerate(feature_names):
                        if feature not in [feature1, feature2]:
                            col_idx = feature_idx % 2
                            with col_features[col_idx]:
                                if feature in ['CustomerID']:
                                    other_inputs[feature] = 0
                                elif feature in encoders:
                                    options = encoders[feature].classes_
                                    other_inputs[feature] = st.selectbox(f"{feature}", options)
                                else:
                                    from src.utils import get_range_for_feature
                                    min_val, max_val, step_val = get_range_for_feature(feature)
                                    other_inputs[feature] = st.slider(
                                        f"{feature}", 
                                        min_value=min_val,
                                        max_value=max_val,
                                        step=step_val
                                    )
                            feature_idx += 1
                    
                    # Define ranges for the two features
                    if st.button("Generate What-If Analysis"):
                        with st.spinner("Generating analysis..."):
                            # Create ranges for the two features
                            if feature1 in encoders:
                                range1 = encoders[feature1].classes_
                            else:
                                from src.utils import get_range_for_feature
                                min_val, max_val, step_val = get_range_for_feature(feature1)
                                range1 = np.linspace(min_val, max_val, 10)
                            
                            if feature2 in encoders:
                                range2 = encoders[feature2].classes_
                            else:
                                from src.utils import get_range_for_feature
                                min_val, max_val, step_val = get_range_for_feature(feature2)
                                range2 = np.linspace(min_val, max_val, 10)
                            
                            # Create grid of predictions
                            results = []
                            
                            for val1 in range1:
                                for val2 in range2:
                                    # Create input dictionary
                                    input_data = other_inputs.copy()
                                    input_data[feature1] = val1
                                    input_data[feature2] = val2
                                    
                                    # Create dataframe
                                    input_df = pd.DataFrame([input_data])
                                    
                                    # Apply encoders
                                    for col, encoder in encoders.items():
                                        if col in input_df.columns:
                                            input_df[col] = encoder.transform(input_df[col])
                                    
                                    # Ensure column order
                                    input_df = input_df[feature_names]
                                    
                                    # Scale data
                                    input_scaled = scaler.transform(input_df)
                                    
                                    # Predict
                                    prediction = model.predict(input_scaled)[0]
                                    
                                    # Decode categorical values for display
                                    val1_display = val1
                                    val2_display = val2
                                    
                                    results.append({
                                        feature1: val1_display,
                                        feature2: val2_display,
                                        'Prediction': prediction
                                    })
                            
                            # Convert to dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Create heatmap or scatter plot
                            if len(range1) <= 20 and len(range2) <= 20:
                                # Create pivot table for heatmap
                                pivot_df = results_df.pivot(index=feature2, columns=feature1, values='Prediction')
                                
                                fig = px.imshow(
                                    pivot_df,
                                    labels=dict(x=feature1, y=feature2, color="Predicted Resolution Time (Days)"),
                                    title=f"How {feature1} and {feature2} Affect Prediction",
                                    color_continuous_scale="viridis"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Create 3D scatter plot
                                fig = px.scatter_3d(
                                    results_df,
                                    x=feature1,
                                    y=feature2,
                                    z='Prediction',
                                    color='Prediction',
                                    title=f"How {feature1} and {feature2} Affect Prediction",
                                    color_continuous_scale="viridis"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show data table
                            with st.expander("View Raw Prediction Data"):
                                st.dataframe(results_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error in What-If Analysis: {str(e)}")
                st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error in model interpretation: {str(e)}")
        st.info("Full error traceback (helpful for debugging):")
        st.code(traceback.format_exc())