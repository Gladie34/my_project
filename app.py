
import streamlit as st
import pandas as pd

from src.explore_page import eda, plot_time_to_resolution_distribution, corelation_map
from src.predict_page import predict_customer_info

# Page config
st.set_page_config(page_title="Insurance Time Resolution Predictor", layout="wide")
st.title("🕒 Insurance Claim TimeToResolutionDays Predictor")

# Sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("📂 Upload your input CSV file", type=["csv"])

    def load_data(file):
        return pd.read_csv(file)

    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = pd.read_csv("./data/Insurance_data.csv")

# Main Section - Select Action
st.markdown("### Select an Action")
action = st.radio("Do you want to explore the data or run the model?", 
                  ('Explore Data', 'Predict TimeToResolutionDays'), index=0)

st.divider()  # Adds a visual separator

# Display the chosen section
if action == 'Explore Data':
    st.subheader("🔍 Data Exploration")
    eda(df)
    plot_time_to_resolution_distribution(df)
    corelation_map(df)

elif action == 'Predict TimeToResolutionDays':
    st.subheader("📊 Prediction")
    predict_customer_info(df)