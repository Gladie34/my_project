import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_model_assets, get_range_for_feature

# Load model and assets
model, encoders, scaler, feature_names = load_model_assets()

def predict_customer_info():
    st.info("📝 Please fill in the details below and click Predict.")
    
    # Create a form to keep the UI organized
    with st.form("prediction_form"):
        input_data = {}
        
        # Create feature categories
        feature_categories = {
            "Customer Info": ["CustomerID", "Age", "Gender", "Income", "Region", "CustomerSatisfaction"],
            "Policy Details": ["PolicyType", "PolicyStartYear", "PremiumMonthly", "PolicyDuration", "RenewalStatus", 
                              "PremiumPaymentFrequency", "PolicyDiscounts"],
            "Claim Information": ["ClaimAmount", "ClaimsRejected", "DaysToFile", "ClaimComplexity", "ClaimReasons", 
                                 "LastClaimAmount", "ClaimsMade", "FraudulentClaims"]
        }
        
        # Add additional categories if needed based on your dataset
        additional_categories = {
            "Service Details": ["CustomerServiceFrequency", "ServiceNature", "CustomerFeedbackScore", "NumberOfInquiries"],
            "Property Information": ["HouseType", "LifeInsuranceType", "HealthInsuranceType", "LifeInsurancePlan"]
        }
        
        # Merge all categories
        all_categories = {**feature_categories, **additional_categories}
        
        # Get all features from categories
        categorized_features = []
        for features in all_categories.values():
            categorized_features.extend(features)
        
        # Create a mapping of feature to category
        feature_to_category = {}
        for category, features in all_categories.items():
            for feature in features:
                feature_to_category[feature] = category
        
        # For any feature not in a category, assign to one of the existing categories
        for feature in feature_names:
            if feature not in categorized_features:
                # Assign to customer info by default, but you can change this logic
                feature_to_category[feature] = "Additional Features"
                
        # Get all unique categories used in the dataset
        used_categories = set(feature_to_category.values())
        
        # Display features by category in a horizontal layout
        for category in used_categories:
            # Get features in this category that are in the dataset
            category_features = [f for f in feature_names if feature_to_category.get(f) == category]
            
            if category_features:  # Only show categories that have features in the dataset
                st.markdown(f"**{category}**")
                
                # Determine how many features to display per row
                features_per_row = 4  # Adjust this number as needed
                
                # Create rows of features
                for i in range(0, len(category_features), features_per_row):
                    row_features = category_features[i:i+features_per_row]
                    cols = st.columns(features_per_row)
                    
                    for j, feature in enumerate(row_features):
                        with cols[j]:
                            if feature in ['CustomerID']:
                                input_data[feature] = 0  # Use dummy/default
                            elif feature in encoders:
                                options = encoders[feature].classes_
                                input_data[feature] = st.selectbox(
                                    f"{feature}", 
                                    options,
                                    label_visibility="visible"
                                )
                            else:
                                # Use number input for numeric features
                                min_val, max_val, step_val = get_range_for_feature(feature)
                                default_val = (min_val + max_val) / 2
                                input_data[feature] = st.number_input(
                                    f"{feature}", 
                                    min_value=None,  # Remove min constraint
                                    max_value=None,  # Remove max constraint
                                    value=float(default_val),
                                    step=step_val,
                                    label_visibility="visible"
                                )
        
        # Create a centered predict button
        cols = st.columns([1, 1, 1])
        with cols[1]:
            predict_button = st.form_submit_button(
                "Predict",
                help="Calculate the predicted resolution time"
            )
    
    # Process prediction if button is clicked
    if predict_button:
        # Validate inputs
        valid_inputs = True
        for feature, value in input_data.items():
            if feature not in encoders and value < 0:
                st.error(f"❌ {feature} cannot be negative. Please enter a valid value.")
                valid_inputs = False
                
        if valid_inputs:
            # Create results container
            result_container = st.container()
            
            with result_container:
                # Display a spinner while processing
                with st.spinner("Processing prediction..."):
                    input_df = pd.DataFrame([input_data])
                    
                    # Apply encoders
                    for col, encoder in encoders.items():
                        input_df[col] = encoder.transform(input_df[col])
                    
                    # Align column order
                    input_df = input_df[feature_names]
                    
                    # Scale
                    input_scaled = scaler.transform(input_df)
                    
                    # Predict
                    prediction = model.predict(input_scaled)[0]
                
                # Show results in an attractive format
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                    <div style='background-color: #f0f7ff; padding: 20px; border-radius: 10px; 
                    border-left: 5px solid #1E88E5; text-align: center;'>
                        <h3 style='margin-top: 0; color: #1E88E5;'>Prediction Results</h3>
                        <h2 style='font-size: 2.5rem; margin: 10px 0; color: #333;'>
                            {:.2f} days
                        </h2>
                        <p style='margin-bottom: 0;'>Estimated time to resolve this claim</p>
                    </div>
                    """.format(prediction), unsafe_allow_html=True)
                
                # Display additional information based on prediction
                if prediction <= 7:
                    st.success("This claim is expected to be resolved quickly.")
                elif prediction <= 30:
                    st.info("This claim has a standard resolution timeframe.")
                else:
                    st.warning("This claim may take longer than average to resolve.")
    
    # Return the model so it can be stored in session state
    return model