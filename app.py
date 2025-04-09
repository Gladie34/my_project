
import streamlit as st
import pandas as pd

from src.explore_page import eda, plot_time_to_resolution_distribution, corelation_map
from src.predict_page import predict_customer_info



def replace_county_with_code(Insurance_data1,county_column='Region'):
    try:
        # Define county name to county code mapping
        county_mapping = {
            "Mombasa": 1, "Kwale": 2, "Kilifi": 3, "Tana River": 4, "Lamu": 5, "Taita Taveta": 6,
            "Garissa": 7, "Wajir": 8, "Mandera": 9, "Marsabit": 10, "Isiolo": 11, "Meru": 12,
            "Tharaka Nithi": 13, "Embu": 14, "Kitui": 15, "Machakos": 16, "Makueni": 17,
            "Nyandarua": 18, "Nyeri": 19, "Kirinyaga": 20, "Murang'a": 21, "Kiambu": 22,
            "Turkana": 23, "West Pokot": 24, "Samburu": 25, "Trans Nzoia": 26, "Eldoret": 27,
            "Elgeyo Marakwet": 28, "Nandi": 29, "Baringo": 30, "Laikipia": 31, "Nakuru": 32,
            "Narok": 33, "Kajiado": 34, "Kericho": 35, "Bomet": 36, "Kakamega": 37, "Vihiga": 38,
            "Bungoma": 39, "Busia": 40, "Siaya": 41, "Kisumu": 42, "Homa Bay": 43, "Migori": 44,
            "Kisii": 45, "Nyamira": 46, "Nairobi": 47
        }

        # Replace county names with county codes
        Insurance_data1['Region'] = Insurance_data1[county_column].map(county_mapping)

        # Drop the original county name column
        Insurance_data1 = Insurance_data1.drop(columns=[county_column])

        print("✅ County names replaced with codes successfully!")

        return Insurance_data1

    except Exception as e:
        print(f"Error replacing county names with codes: {e}")
        return None, None
    
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
        df = replace_county_with_code(df,county_column='Region')
        print(df.columns)

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


