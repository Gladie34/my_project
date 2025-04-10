import streamlit as st
import pandas as pd
from src.explore_page import eda, plot_time_to_resolution_distribution, corelation_map
from src.predict_page import predict_customer_info
from src.xai_page import explain_model  # Requires updated explain_model(df, model)

# Set page styling
def set_page_styling():
    st.markdown("""
        <style>
        .stApp { background-color: #f8f9fa; }
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid #3498db;
        }
        .section-header {
            font-size: 1.8rem;
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 10px;
            margin: 1.5rem 0;
        }
        .info-box {
            background-color: #e8f4f8;
            border-left: 5px solid #3498db;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .success-box {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Replace county names with codes
def replace_county_with_code(df, county_column='Region'):
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
    df['County'] = df[county_column].map(county_mapping)
    return df

def main():
    st.set_page_config(page_title="Insurance Claims Resolution Predictor", page_icon="⏱️", layout="wide")
    set_page_styling()
    st.markdown("<h1 class='main-header'>⏱️ Insurance Claim Resolution Time Predictor</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📂 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        st.markdown("---")
        section = st.radio("Choose a section", ["📊 Explore Data", "🧠 Train & Predict", "🔍 Explainable AI"])
        st.markdown("---")
        st.markdown("**Version**: 2.0  \n**Last updated**: April 2025")

    # Load and prepare data
    @st.cache_data
    def load_data(path):
        return pd.read_csv(path)

    if uploaded_file:
        df = load_data(uploaded_file)
        st.success("✅ File uploaded successfully.")
    else:
        df = load_data("./data/Insurance_data.csv")
        df = replace_county_with_code(df, 'Region')
        st.info("Using default dataset.")

    # Section: Explore Data
    if section == "📊 Explore Data":
        st.markdown("<h2 class='section-header'>📊 Data Exploration</h2>", unsafe_allow_html=True)
        with st.expander("Dataset Overview", expanded=True): eda(df)
        with st.expander("Time to Resolution Distribution", expanded=True): plot_time_to_resolution_distribution(df)
        with st.expander("Correlation Map", expanded=True): corelation_map(df)

    # Section: Train & Predict
    elif section == "🧠 Train & Predict":
        st.markdown("<h2 class='section-header'>🧠 Train & Predict</h2>", unsafe_allow_html=True)
        st.session_state.model = predict_customer_info(df)  # Store trained model in session_state

    # Section: Explainable AI
    elif section == "🔍 Explainable AI":
        st.markdown("<h2 class='section-header'>🔍 Explainable AI Insights</h2>", unsafe_allow_html=True)
        if "model" in st.session_state:
            explain_model(df, st.session_state.model)
        else:
            st.warning("⚠️ Please train the model first under 'Train & Predict' section.")

if __name__ == "__main__":
    main()
