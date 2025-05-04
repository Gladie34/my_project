import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from src.utils import load_model_assets

# Load model and assets
model, encoders, scaler, feature_names = load_model_assets()

# Try to fix PIL issue before importing SHAP
try:
    import PIL
    from PIL import Image
    if not hasattr(Image, 'Resampling'):  # Check if Resampling is missing
        # Add Resampling enum to PIL.Image if it's missing
        # This is a compatibility fix for older Pillow versions
        Image.Resampling = type('Resampling', (), {
            'NEAREST': PIL.Image.NEAREST,
            'BILINEAR': PIL.Image.BILINEAR,
            'BICUBIC': PIL.Image.BICUBIC,
            'LANCZOS': PIL.Image.LANCZOS
        })
except Exception as e:
    st.error(f"Error with PIL: {e}")

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
except Exception as e:
    st.error(f"Error importing SHAP: {e}")
    SHAP_AVAILABLE = False

# Function to get readable feature names
def get_readable_feature_names(original_features):
    """Convert feature codes to readable names."""
    # This dictionary maps feature codes to human-readable names
    # Update this with your actual feature names based on your dataset
    feature_name_mapping = {
        # Examples - update with your actual feature mappings
        "ClaimAmount": "Claim Amount ($)",
        "Age": "Customer Age",
        "Gender": "Customer Gender",
        "PolicyDuration": "Policy Length (months)",
        "PremiumAmount": "Premium Amount ($)",
        "ClaimType": "Type of Claim",
        "ClaimComplexity": "Claim Complexity Score",
        "Region": "Geographic Region",
        "DaysToFile": "Days to File Claim",
        "PriorClaims": "Number of Prior Claims",
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
    st.write("### 🧠 Understanding Model Predictions")
    st.info("This page helps interpret how the model makes decisions for insurance claim resolution time.")
    
    # Get readable feature names
    readable_feature_names = get_readable_feature_names(feature_names)
    
    tabs = st.tabs(["Feature Importance", "SHAP Values", "What-If Analysis"])
    
    with tabs[0]:
        st.write("#### 📊 Global Feature Importance")
        st.write("These are the most influential factors in determining claim resolution time.")
        
        # Get feature importance from model (assuming XGBoost model)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': readable_feature_names,  # Use readable names
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
            st.warning("⚠️ Model doesn't provide direct feature importance. Try SHAP values instead.")
    
    with tabs[1]:
        st.write("#### 🔍 SHAP Values Explanation")
        st.write("SHAP values show how each feature contributes to individual predictions.")
        
        if not SHAP_AVAILABLE:
            st.warning("⚠️ SHAP library not installed or incompatible. Please install with: `pip install shap`")
            st.info("If you've already installed SHAP, there might be a compatibility issue with your environment.")
        else:
            # Add an option to use alternate approach
            use_alt_approach = st.checkbox("Use alternative SHAP approach (try this if you get errors)")
            
            # User can choose between different SHAP visualizations
            shap_view = st.radio(
                "Choose SHAP visualization type:",
                ["Summary Plot", "Detailed Beeswarm", "Waterfall Plot (Single Instance)"],
                horizontal=True
            )
            
            if st.button("Generate SHAP Analysis"):
                with st.spinner("Calculating SHAP values (this may take a while)..."):
                    try:
                        # Get a sample of the data (for performance)
                        if len(df) > 100:
                            sample_size = st.slider("Sample size (smaller is faster):", 10, min(100, len(df)), 50)
                            sample_df = df.sample(sample_size, random_state=42)
                        else:
                            sample_df = df
                        
                        # Prepare the data
                        X = sample_df.copy()
                        for col in encoders:
                            if col in X.columns:
                                X[col] = encoders[col].transform(X[col])
                        
                        # Ensure only appropriate columns are used
                        X = X[feature_names]
                        
                        # Scale the data
                        X_scaled = scaler.transform(X)
                        
                        # Create DataFrame with READABLE feature names for better visualization
                        X_display = pd.DataFrame(X_scaled, columns=readable_feature_names)
                        
                        if use_alt_approach:
                            # Alternative approach using TreeExplainer directly
                            st.write("Using alternative TreeExplainer approach...")
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_scaled)
                            
                            # Convert to DataFrame for visualization
                            shap_df = pd.DataFrame(shap_values, columns=readable_feature_names)  # Use readable names
                            
                            # Create simpler visualization with Plotly instead of matplotlib
                            if shap_view == "Summary Plot" or shap_view == "Detailed Beeswarm":
                                # Calculate average absolute SHAP value for each feature
                                feature_importance = pd.DataFrame({
                                    'Feature': readable_feature_names,  # Use readable names
                                    'Importance': np.abs(shap_df).mean().values
                                }).sort_values('Importance', ascending=False)
                                
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
                                
                                # Also show a heatmap of SHAP values
                                st.write("#### SHAP Values Heatmap")
                                st.write("Red = positive impact, Blue = negative impact")
                                
                                # Sample a few rows for the heatmap (too many makes it unreadable)
                                heat_rows = min(20, shap_df.shape[0])
                                heat_data = shap_df.iloc[:heat_rows, :]
                                
                                heat_fig = px.imshow(
                                    heat_data,
                                    labels=dict(x="Feature", y="Instance", color="SHAP Value"),
                                    x=readable_feature_names,  # Use readable names
                                    color_continuous_scale='RdBu_r',
                                    aspect="auto"
                                )
                                st.plotly_chart(heat_fig, use_container_width=True)
                                
                            elif shap_view == "Waterfall Plot (Single Instance)":
                                # Let user select an instance to analyze
                                instance_index = st.number_input(
                                    "Select instance to explain (row number):", 
                                    min_value=0, 
                                    max_value=len(X_scaled)-1, 
                                    value=0
                                )
                                
                                # Get SHAP values for this instance
                                instance_shap = shap_df.iloc[instance_index]
                                
                                # Sort by absolute value
                                sorted_idx = np.argsort(np.abs(instance_shap.values))[::-1]
                                sorted_features = [readable_feature_names[i] for i in sorted_idx]  # Use readable names
                                sorted_values = instance_shap.values[sorted_idx]
                                
                                # Create waterfall-like chart with Plotly
                                waterfall_df = pd.DataFrame({
                                    'Feature': sorted_features,
                                    'SHAP Value': sorted_values
                                })
                                
                                fig = px.bar(
                                    waterfall_df,
                                    x='SHAP Value',
                                    y='Feature',
                                    orientation='h',
                                    title=f'Feature Impact for Instance #{instance_index}',
                                    color='SHAP Value',
                                    color_continuous_scale='RdBu_r'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show the actual instance data
                                st.write("#### Instance Details:")
                                instance_data = {}
                                for i, feature in enumerate(feature_names):
                                    # Try to decode categorical features
                                    if feature in encoders:
                                        # Find the original value that most closely matches the encoded value
                                        original_values = encoders[feature].classes_
                                        encoded_val = X[feature].iloc[instance_index]
                                        # Find closest match for the encoded value
                                        closest_idx = np.argmin(np.abs(encoders[feature].transform(original_values) - encoded_val))
                                        instance_data[readable_feature_names[i]] = original_values[closest_idx]
                                    else:
                                        # For numerical features, show original value
                                        instance_data[readable_feature_names[i]] = X[feature].iloc[instance_index]
                                
                                # Display as a DataFrame for better formatting
                                st.dataframe(pd.DataFrame([instance_data]), use_container_width=True)
                                
                        else:
                            # Standard SHAP approach
                            try:
                                # Create explainer
                                explainer = shap.Explainer(model)
                                shap_values = explainer(X_scaled)
                                
                                # Override the feature names with readable ones
                                shap_values.feature_names = readable_feature_names
                                
                                # Display the selected SHAP visualization
                                if shap_view == "Summary Plot":
                                    plt.figure(figsize=(10, 8))
                                    shap.summary_plot(shap_values, X_display, feature_names=readable_feature_names)
                                    st.pyplot(plt.gcf(), clear_figure=True)
                                    
                                elif shap_view == "Detailed Beeswarm":
                                    plt.figure(figsize=(10, 8))
                                    shap.plots.beeswarm(shap_values, max_display=15)
                                    st.pyplot(plt.gcf(), clear_figure=True)
                                    
                                elif shap_view == "Waterfall Plot (Single Instance)":
                                    # Let user select an instance to analyze
                                    instance_index = st.number_input(
                                        "Select instance to explain (row number):", 
                                        min_value=0, 
                                        max_value=len(X_scaled)-1, 
                                        value=0
                                    )
                                    
                                    plt.figure(figsize=(10, 8))
                                    shap.plots.waterfall(shap_values[instance_index], max_display=15)
                                    st.pyplot(plt.gcf(), clear_figure=True)
                                
                            except Exception as e:
                                st.error(f"Error with standard SHAP approach: {e}")
                                st.info("Try checking the 'Use alternative SHAP approach' option above.")
                        
                        # Interpretation guide
                        st.write("**Interpretation Guide:**")
                        st.write("""
                        - **Red/positive values** push the prediction higher (longer resolution time)
                        - **Blue/negative values** push the prediction lower (shorter resolution time)
                        - **Larger magnitude** means stronger impact on the prediction
                        """)
                        
                    except Exception as e:
                        st.error(f"Error generating SHAP analysis: {e}")
                        st.info("Try using a smaller dataset or check if SHAP is compatible with your environment.")
                        
                        # Fallback to basic feature importance
                        st.write("#### Fallback: Basic Feature Importance")
                        if hasattr(model, 'feature_importances_'):
                            importances = model.feature_importances_
                            feature_importance_df = pd.DataFrame({
                                'Feature': readable_feature_names,  # Use readable names
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
        st.write("#### 🔄 What-If Analysis")
        st.write("Explore how changing input values affects predictions.")
        
        st.subheader("Select Features to Compare")
        
        # Let user select features to vary
        cols = st.columns(2)
        with cols[0]:
            feature1_idx = st.selectbox("Feature 1:", range(len(readable_feature_names)), 
                                      format_func=lambda i: readable_feature_names[i])
            feature1 = feature_names[feature1_idx]
        with cols[1]:
            # Default to a different feature for second selection
            default_idx = min(1, len(readable_feature_names)-1)
            if default_idx == feature1_idx:
                default_idx = min(0, len(readable_feature_names)-1)
                
            feature2_idx = st.selectbox("Feature 2:", range(len(readable_feature_names)), 
                                      index=default_idx,
                                      format_func=lambda i: readable_feature_names[i])
            feature2 = feature_names[feature2_idx]
        
        if feature1 == feature2:
            st.warning("⚠️ Please select two different features.")
        else:
            # Create inputs for other features
            st.subheader("Set Values for Other Features")
            other_inputs = {}
            
            # Create 2-column layout for controls
            num_other_features = len(feature_names) - 2
            col_features = st.columns(2)
            
            feature_idx = 0
            for i, feature in enumerate(feature_names):
                if feature not in [feature1, feature2]:
                    col_idx = feature_idx % 2
                    with col_features[col_idx]:
                        if feature in ['CustomerID']:
                            other_inputs[feature] = 0
                        elif feature in encoders:
                            options = encoders[feature].classes_
                            other_inputs[feature] = st.selectbox(f"{readable_feature_names[i]}", options)
                        else:
                            # Use slider with appropriate range
                            from src.utils import get_range_for_feature
                            min_val, max_val, step_val = get_range_for_feature(feature)
                            other_inputs[feature] = st.slider(
                                f"{readable_feature_names[i]}", 
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
                            
                            # Store result with readable feature names
                            feature1_readable = readable_feature_names[feature_names.index(feature1)]
                            feature2_readable = readable_feature_names[feature_names.index(feature2)]
                            
                            results.append({
                                feature1_readable: val1,
                                feature2_readable: val2,
                                'Prediction': prediction
                            })
                    
                    # Convert to dataframe
                    results_df = pd.DataFrame(results)
                    
                    # Create heatmap or scatter plot
                    feature1_readable = readable_feature_names[feature_names.index(feature1)]
                    feature2_readable = readable_feature_names[feature_names.index(feature2)]
                    
                    if len(range1) <= 20 and len(range2) <= 20:
                        # Create pivot table for heatmap
                        pivot_df = results_df.pivot(index=feature2_readable, columns=feature1_readable, values='Prediction')
                        
                        fig = px.imshow(
                            pivot_df,
                            labels=dict(x=feature1_readable, y=feature2_readable, color="Predicted Resolution Time (Days)"),
                            title=f"How {feature1_readable} and {feature2_readable} Affect Prediction",
                            color_continuous_scale="viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Create 3D scatter plot
                        fig = px.scatter_3d(
                            results_df,
                            x=feature1_readable,
                            y=feature2_readable,
                            z='Prediction',
                            color='Prediction',
                            title=f"How {feature1_readable} and {feature2_readable} Affect Prediction",
                            color_continuous_scale="viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    with st.expander("View Raw Prediction Data"):
                        st.dataframe(results_df, use_container_width=True)