
import streamlit as st
import pandas as pd


from src.explore_page import eda
from src.predict_page import predict_customer_info

# Page config
st.set_page_config(page_title="Insurance Time Resolution Predictor", layout="wide")
st.title("🕒 Insurance Claim TimeToResolutionDays Predictor")

with st.sidebar:
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    def load_data(file):
        data = pd.read_csv(file)
        return data

    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = pd.read_csv("./data/Insurance_data.csv")

    st.title("Do you want to explore the data or run the model?")
    action = st.radio("""Please, Choose""", ('','Explore Data', 'Predict TimeToResolutionDays'))
if action == 'Explore Data':

    eda(df)

elif action == 'Predict TimeToResolutionDays':

    predict_customer_info(df)
