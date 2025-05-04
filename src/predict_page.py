import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_model_assets, get_range_for_feature

# Load model and assets
model, encoders, scaler, feature_names = load_model_assets()

def predict_customer_info():
    st.info("📝 Please fill in the details below and click Predict.")
    
    input_data = {}
    for feature in feature_names:
        if feature in ['CustomerID']:
            input_data[feature] = 0  # Use dummy/default
        elif feature in encoders:
            options = encoders[feature].classes_
            input_data[feature] = st.selectbox(f"{feature}", options)
        else:
            # Use number input for numeric features instead of sliders
            min_val, max_val, step_val = get_range_for_feature(feature)
            # Default value is the middle of the range
            default_val = (min_val + max_val) / 2
            
            # Add a hint about typical values
            hint = ""
            if "Income" in feature:
                hint = " (e.g., 75000)"
            elif "Amount" in feature:
                hint = " (e.g., 5000)"
            elif "Age" in feature:
                hint = " (e.g., 35)"
                
            # Use number input with appropriate step size
            input_data[feature] = st.number_input(
                f"{feature}{hint}", 
                min_value=None,  # Remove min constraint
                max_value=None,  # Remove max constraint
                value=float(default_val),
                step=step_val
            )
    
    if st.button("🚀 Predict"):
        # Validate inputs
        valid_inputs = True
        for feature, value in input_data.items():
            if feature not in encoders and value < 0:
                st.error(f"❌ {feature} cannot be negative. Please enter a valid value.")
                valid_inputs = False
                
        if valid_inputs:
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
            st.success(f"✅ Predicted Time to Resolution: **{prediction:.2f} days**")
            
            # Display additional information about the prediction
            if prediction <= 7:
                st.info("📊 This claim is expected to be resolved quickly.")
            elif prediction <= 30:
                st.info("📊 This claim has a standard resolution timeframe.")
            else:
                st.warning("📊 This claim may take longer than average to resolve.")
    
    # Return the model so it can be stored in session state
    return model