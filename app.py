import streamlit as st
import pandas as pd
import os
from src.explore_page import run_exploration
from src.predict_page import predict_customer_info
from src.xai_page import explain_model

# Function to load CSS
def load_css(css_file):
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file {css_file} not found.")

def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully.")
    else:
        df = pd.read_csv("data/Insurance_data.csv")
        st.info("Using default dataset.")
    return df

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Time for Claim Resolution Predictor",
        page_icon="⏱️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS (updated path)
    load_css("style.css")  # Loading from root directory
    
    # App header with clock emoji and updated title
    st.title("⏱️ Time for Claim Resolution Predictor")
    st.markdown("#### Predict resolution time for insurance claims using machine learning")
    
    # Add a separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.markdown("## Upload Data")
        df = load_data()

        st.markdown("## Navigate to:")
        section = st.radio("", ["Explore Data", "Predict", "Interpretation"])

        st.markdown("----")
        st.markdown("**Version:** 2.3")
        st.markdown("**Last updated:** April 2025")
        
        # About section in sidebar
        with st.expander("About this app"):
            st.markdown("""
            This application uses machine learning to predict how long it will 
            take to resolve insurance claims based on various factors.
            
            **Features:**
            - Data visualization and exploration
            - Custom prediction inputs
            - Model interpretation with SHAP values
            """)

    # Main content area
    if section == "Explore Data":
        st.markdown("<h2 class='section-header'>Data Exploration</h2>", unsafe_allow_html=True)
        run_exploration(df)

    elif section == "Predict":
        st.markdown("<h2 class='section-header'>Prediction</h2>", unsafe_allow_html=True)
        
        # Instructions card
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; 
        border-left: 5px solid #1E88E5; margin-bottom: 20px;'>
            <h4 style='margin-top: 0'>How to use the predictor:</h4>
            <p>1. Enter the claim details in the form below</p>
            <p>2. Click the Predict button to get the estimated resolution time</p>
            <p>3. Check the Interpretation tab to understand what factors influence the prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        model = predict_customer_info()
        if model:
            st.session_state.model = model

    elif section == "Interpretation":
        st.markdown("<h2 class='section-header'>Model Interpretation</h2>", unsafe_allow_html=True)
        
        if "model" in st.session_state:
            explain_model(df, st.session_state.model)
        else:
            st.warning("""
            Please run a prediction first to load the model.
            
            Go to the Predict section and submit a prediction to enable model interpretation.
            """)
            
            if st.button("Go to Prediction Page"):
                st.session_state.section = "Predict"
                st.experimental_rerun()
    
    # Footer
    st.markdown("<hr style='margin-top: 30px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8rem;'>
        Developed with Streamlit and Machine Learning | © 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()