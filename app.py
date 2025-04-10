
# # import streamlit as st
# # import pandas as pd

# # from src.explore_page import eda, plot_time_to_resolution_distribution, corelation_map
# # from src.predict_page import predict_customer_info



# # def replace_county_with_code(Insurance_data1,county_column='Region'):
# #     try:
# #         # Define county name to county code mapping
# #         county_mapping = {
# #             "Mombasa": 1, "Kwale": 2, "Kilifi": 3, "Tana River": 4, "Lamu": 5, "Taita Taveta": 6,
# #             "Garissa": 7, "Wajir": 8, "Mandera": 9, "Marsabit": 10, "Isiolo": 11, "Meru": 12,
# #             "Tharaka Nithi": 13, "Embu": 14, "Kitui": 15, "Machakos": 16, "Makueni": 17,
# #             "Nyandarua": 18, "Nyeri": 19, "Kirinyaga": 20, "Murang'a": 21, "Kiambu": 22,
# #             "Turkana": 23, "West Pokot": 24, "Samburu": 25, "Trans Nzoia": 26, "Eldoret": 27,
# #             "Elgeyo Marakwet": 28, "Nandi": 29, "Baringo": 30, "Laikipia": 31, "Nakuru": 32,
# #             "Narok": 33, "Kajiado": 34, "Kericho": 35, "Bomet": 36, "Kakamega": 37, "Vihiga": 38,
# #             "Bungoma": 39, "Busia": 40, "Siaya": 41, "Kisumu": 42, "Homa Bay": 43, "Migori": 44,
# #             "Kisii": 45, "Nyamira": 46, "Nairobi": 47
# #         }

# #         # Replace county names with county codes
# #         Insurance_data1['Region'] = Insurance_data1[county_column].map(county_mapping)

# #         # Drop the original county name column
# #         Insurance_data1 = Insurance_data1.drop(columns=[county_column])

# #         print("✅ County names replaced with codes successfully!")

# #         return Insurance_data1

# #     except Exception as e:
# #         print(f"Error replacing county names with codes: {e}")
# #         return None, None
    
# # # Page config
# # st.set_page_config(page_title="Insurance Time Resolution Predictor", layout="wide")
# # st.title("🕒 Insurance Claim TimeToResolutionDays Predictor")

# # # Sidebar for file upload
# # with st.sidebar:
# #     uploaded_file = st.file_uploader("📂 Upload your input CSV file", type=["csv"])

# #     def load_data(file):
# #         return pd.read_csv(file)

# #     if uploaded_file is not None:
# #         df = load_data(uploaded_file)
# #     else:
# #         df = pd.read_csv("./data/Insurance_data.csv")
# #         df = replace_county_with_code(df,county_column='Region')
# #         print(df.columns)

# # # Main Section - Select Action
# # st.markdown("### Select an Action")
# # action = st.radio("Do you want to explore the data or run the model?", 
# #                   ('Explore Data', 'Predict TimeToResolutionDays'), index=0)

# # st.divider()  # Adds a visual separator

# # # Display the chosen section
# # if action == 'Explore Data':
# #     st.subheader("🔍 Data Exploration")
# #     eda(df)
# #     plot_time_to_resolution_distribution(df)
# #     corelation_map(df)

# # elif action == 'Predict TimeToResolutionDays':
# #     st.subheader("📊 Prediction")
# #     predict_customer_info(df)


# # import streamlit as st
# # import pandas as pd

# # from src.explore_page import eda, plot_time_to_resolution_distribution, corelation_map
# # from src.predict_page import predict_customer_info
# # # from src.xai_page import explain_model

# # # Set app-wide theme and styling
# # def set_page_styling():
# #     # Custom CSS to improve aesthetics
# #     st.markdown(
# #         """
# #         <style>
# #         .stApp {
# #             background-color: #f8f9fa;
# #         }
# #         .main-header {
# #             font-size: 2.5rem;
# #             color: #2c3e50;
# #             text-align: center;
# #             padding: 1rem 0;
# #             margin-bottom: 2rem;
# #             border-bottom: 2px solid #3498db;
# #         }
# #         .section-header {
# #             font-size: 1.8rem;
# #             color: #34495e;
# #             border-left: 4px solid #3498db;
# #             padding-left: 10px;
# #             margin: 1.5rem 0;
# #         }
# #         .info-box {
# #             background-color: #e8f4f8;
# #             border-left: 5px solid #3498db;
# #             padding: 1rem;
# #             border-radius: 0.5rem;
# #         }
# #         .success-box {
# #             background-color: #d4edda;
# #             border-left: 5px solid #28a745;
# #             padding: 1rem;
# #             border-radius: 0.5rem;
# #         }
# #         .warning-box {
# #             background-color: #fff3cd;
# #             border-left: 5px solid #ffc107;
# #             padding: 1rem;
# #             border-radius: 0.5rem;
# #         }
# #         .sidebar-content {
# #             padding: 1.5rem 1rem;
# #         }
# #         .metric-card {
# #             background-color: white;
# #             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
# #             padding: 1.5rem;
# #             border-radius: 0.5rem;
# #             margin-bottom: 1rem;
# #             text-align: center;
# #         }
# #         .metric-value {
# #             font-size: 2rem;
# #             font-weight: bold;
# #             color: #3498db;
# #         }
# #         .metric-label {
# #             font-size: 1rem;
# #             color: #7f8c8d;
# #         }
# #         </style>
# #         """,
# #         unsafe_allow_html=True
# #     )

# # def replace_county_with_code(Insurance_data1, county_column='Region'):
# #     try:
# #         # Define county name to county code mapping
# #         county_mapping = {
# #             "Mombasa": 1, "Kwale": 2, "Kilifi": 3, "Tana River": 4, "Lamu": 5, "Taita Taveta": 6,
# #             "Garissa": 7, "Wajir": 8, "Mandera": 9, "Marsabit": 10, "Isiolo": 11, "Meru": 12,
# #             "Tharaka Nithi": 13, "Embu": 14, "Kitui": 15, "Machakos": 16, "Makueni": 17,
# #             "Nyandarua": 18, "Nyeri": 19, "Kirinyaga": 20, "Murang'a": 21, "Kiambu": 22,
# #             "Turkana": 23, "West Pokot": 24, "Samburu": 25, "Trans Nzoia": 26, "Eldoret": 27,
# #             "Elgeyo Marakwet": 28, "Nandi": 29, "Baringo": 30, "Laikipia": 31, "Nakuru": 32,
# #             "Narok": 33, "Kajiado": 34, "Kericho": 35, "Bomet": 36, "Kakamega": 37, "Vihiga": 38,
# #             "Bungoma": 39, "Busia": 40, "Siaya": 41, "Kisumu": 42, "Homa Bay": 43, "Migori": 44,
# #             "Kisii": 45, "Nyamira": 46, "Nairobi": 47
# #         }

# #         # Replace county names with county codes
# #         Insurance_data1['County'] = Insurance_data1[county_column].map(county_mapping)

# #         # Keep the original county column for display purposes
# #         # Insurance_data1 = Insurance_data1.drop(columns=[county_column])

# #         st.success("✅ County names mapped to codes successfully!")
# #         return Insurance_data1

# #     except Exception as e:
# #         st.error(f"Error replacing county names with codes: {e}")
# #         return Insurance_data1

# # def main():
# #     # Page config with a more professional title and icon
# #     st.set_page_config(
# #         page_title="Insurance Claims Resolution Predictor",
# #         page_icon="⏱️",
# #         layout="wide",
# #         initial_sidebar_state="expanded"
# #     )
    
# #     # Apply custom styling
# #     set_page_styling()
    
# #     # Custom header with improved styling
# #     st.markdown("<h1 class='main-header'>⏱️ Insurance Claim Resolution Time Predictor</h1>", unsafe_allow_html=True)
    
# #     # Improved sidebar with better organization
# #     with st.sidebar:
# #         st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        
# #         # App description
# #         st.markdown("### About This App")
# #         st.markdown("""
# #         This application helps insurance professionals predict how long it will take to resolve 
# #         customer claims based on various factors. You can:
        
# #         - Explore data patterns and distributions
# #         - Train and evaluate a predictive model
# #         - Get XAI (Explainable AI) insights
# #         - Make predictions for new claims
# #         """)
        
# #         st.divider()
        
# #         # File upload section with improved instructions
# #         st.markdown("### Data Input")
# #         uploaded_file = st.file_uploader(
# #             "📂 Upload your insurance claims data (CSV)",
# #             type=["csv"],
# #             help="The file should contain customer and claim information with a 'TimeToResolutionDays' column."
# #         )

# #         st.divider()
        
# #         # Navigation
# #         st.markdown("### Navigation")
# #         action = st.radio(
# #             "Select a section:",
# #             ('📊 Explore Data', '🧠 Train & Predict', '🔍 Explainable AI'),
# #             index=0
# #         )
        
# #         st.divider()
        
# #         # Add simple app stats or info at the bottom
# #         st.markdown("### App Information")
# #         st.markdown("""
# #         **Version:** 2.0
# #         **Last Updated:** April 2025
# #         """)
        
# #         st.markdown("</div>", unsafe_allow_html=True)

# #     # Load data
# #     @st.cache_data
# #     def load_data(file):
# #         return pd.read_csv(file)

# #     if uploaded_file is not None:
# #         df = load_data(uploaded_file)
# #         st.success("Successfully loaded uploaded data!")
# #     else:
# #         try:
# #             df = pd.read_csv("./data/Insurance_data.csv")
# #             df = replace_county_with_code(df, county_column='Region')
# #             st.info("Using default insurance dataset for demonstration.")
# #         except Exception as e:
# #             st.error(f"Error loading default data: {e}")
# #             st.warning("Please upload a CSV file with insurance claim data.")
# #             return

# #     st.divider()  # Adds a visual separator

# #     # Display the chosen section with improved titles and organization
# #     if '📊 Explore Data' in action:
# #         st.markdown("<h2 class='section-header'>📊 Data Exploration & Analysis</h2>", unsafe_allow_html=True)
        
# #         # Add description
# #         st.markdown("""
# #         Explore the insurance claims dataset to understand patterns, distributions, and relationships 
# #         between different variables. This analysis helps identify factors that influence claim resolution time.
# #         """)
        
# #         # Data overview 
# #         with st.expander("Dataset Overview", expanded=True):
# #             eda(df)
        
# #         # Distribution plots
# #         with st.expander("Time to Resolution Distribution", expanded=True):
# #             plot_time_to_resolution_distribution(df)
        
# #         # Correlation analysis
# #         with st.expander("Correlation Analysis", expanded=True):
# #             corelation_map(df)

# #     elif '🧠 Train & Predict' in action:
# #         st.markdown("<h2 class='section-header'>🧠 Model Training & Prediction</h2>", unsafe_allow_html=True)
        
# #         st.markdown("""
# #         Train a machine learning model to predict the time it takes to resolve insurance claims.
# #         The model uses XGBoost, which is well-suited for this type of regression problem.
# #         """)
        
# #         predict_customer_info(df)

# #     elif '🔍 Explainable AI' in action:
# #         st.markdown("<h2 class='section-header'>🔍 Explainable AI Insights</h2>", unsafe_allow_html=True)
        
# #         st.markdown("""
# #         Understand why the model makes specific predictions using modern explainable AI techniques.
# #         These insights help insurance professionals identify key factors affecting resolution time
# #         and make more informed decisions.
# #         """)
        
# #         explain_model(df)

# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import pandas as pd

# from src.explore_page import eda, plot_time_to_resolution_distribution, corelation_map
# from src.predict_page import predict_customer_info
# from src.xai_page import explain_model  # Import the XAI module we created

# # Set app-wide theme and styling
# def set_page_styling():
#     # Custom CSS to improve aesthetics
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-color: #f8f9fa;
#         }
#         .main-header {
#             font-size: 2.5rem;
#             color: #2c3e50;
#             text-align: center;
#             padding: 1rem 0;
#             margin-bottom: 2rem;
#             border-bottom: 2px solid #3498db;
#         }
#         .section-header {
#             font-size: 1.8rem;
#             color: #34495e;
#             border-left: 4px solid #3498db;
#             padding-left: 10px;
#             margin: 1.5rem 0;
#         }
#         .info-box {
#             background-color: #e8f4f8;
#             border-left: 5px solid #3498db;
#             padding: 1rem;
#             border-radius: 0.5rem;
#         }
#         .success-box {
#             background-color: #d4edda;
#             border-left: 5px solid #28a745;
#             padding: 1rem;
#             border-radius: 0.5rem;
#         }
#         .warning-box {
#             background-color: #fff3cd;
#             border-left: 5px solid #ffc107;
#             padding: 1rem;
#             border-radius: 0.5rem;
#         }
#         .sidebar-content {
#             padding: 1.5rem 1rem;
#         }
#         .metric-card {
#             background-color: white;
#             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             padding: 1.5rem;
#             border-radius: 0.5rem;
#             margin-bottom: 1rem;
#             text-align: center;
#         }
#         .metric-value {
#             font-size: 2rem;
#             font-weight: bold;
#             color: #3498db;
#         }
#         .metric-label {
#             font-size: 1rem;
#             color: #7f8c8d;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# def replace_county_with_code(Insurance_data1, county_column='Region'):
#     try:
#         # Define county name to county code mapping
#         county_mapping = {
#             "Mombasa": 1, "Kwale": 2, "Kilifi": 3, "Tana River": 4, "Lamu": 5, "Taita Taveta": 6,
#             "Garissa": 7, "Wajir": 8, "Mandera": 9, "Marsabit": 10, "Isiolo": 11, "Meru": 12,
#             "Tharaka Nithi": 13, "Embu": 14, "Kitui": 15, "Machakos": 16, "Makueni": 17,
#             "Nyandarua": 18, "Nyeri": 19, "Kirinyaga": 20, "Murang'a": 21, "Kiambu": 22,
#             "Turkana": 23, "West Pokot": 24, "Samburu": 25, "Trans Nzoia": 26, "Eldoret": 27,
#             "Elgeyo Marakwet": 28, "Nandi": 29, "Baringo": 30, "Laikipia": 31, "Nakuru": 32,
#             "Narok": 33, "Kajiado": 34, "Kericho": 35, "Bomet": 36, "Kakamega": 37, "Vihiga": 38,
#             "Bungoma": 39, "Busia": 40, "Siaya": 41, "Kisumu": 42, "Homa Bay": 43, "Migori": 44,
#             "Kisii": 45, "Nyamira": 46, "Nairobi": 47
#         }

#         # Replace county names with county codes
#         Insurance_data1['County'] = Insurance_data1[county_column].map(county_mapping)

#         # Keep the original county column for display purposes
#         # Insurance_data1 = Insurance_data1.drop(columns=[county_column])

#         st.success("✅ County names mapped to codes successfully!")
#         return Insurance_data1

#     except Exception as e:
#         st.error(f"Error replacing county names with codes: {e}")
#         return Insurance_data1

# def main():
#     # Page config with a more professional title and icon
#     st.set_page_config(
#         page_title="Insurance Claims Resolution Predictor",
#         page_icon="⏱️",
#         layout="wide",
#         initial_sidebar_state="expanded"
#     )
    
#     # Apply custom styling
#     set_page_styling()
    
#     # Custom header with improved styling
#     st.markdown("<h1 class='main-header'>⏱️ Insurance Claim Resolution Time Predictor</h1>", unsafe_allow_html=True)
    
#     # Improved sidebar with better organization
#     with st.sidebar:
#         st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        
#         # App description
#         st.markdown("### About This App")
#         st.markdown("""
#         This application helps insurance professionals predict how long it will take to resolve 
#         customer claims based on various factors. You can:
        
#         - Explore data patterns and distributions
#         - Train and evaluate a predictive model
#         - Get XAI (Explainable AI) insights
#         - Make predictions for new claims
#         """)
        
#         st.divider()
        
#         # File upload section with improved instructions
#         st.markdown("### Data Input")
#         uploaded_file = st.file_uploader(
#             "📂 Upload your insurance claims data (CSV)",
#             type=["csv"],
#             help="The file should contain customer and claim information with a 'TimeToResolutionDays' column."
#         )

#         st.divider()
        
#         # Navigation
#         st.markdown("### Navigation")
#         action = st.radio(
#             "Select a section:",
#             ('📊 Explore Data', '🧠 Train & Predict', '🔍 Explainable AI'),
#             index=0
#         )
        
#         st.divider()
        
#         # Add simple app stats or info at the bottom
#         st.markdown("### App Information")
#         st.markdown("""
#         **Version:** 2.0
#         **Last Updated:** April 2025
#         """)
        
#         st.markdown("</div>", unsafe_allow_html=True)

#     # Load data
#     @st.cache_data
#     def load_data(file):
#         return pd.read_csv(file)

#     if uploaded_file is not None:
#         df = load_data(uploaded_file)
#         st.success("Successfully loaded uploaded data!")
#     else:
#         try:
#             df = pd.read_csv("./data/Insurance_data.csv")
#             df = replace_county_with_code(df, county_column='Region')
#             st.info("Using default insurance dataset for demonstration.")
#         except Exception as e:
#             st.error(f"Error loading default data: {e}")
#             st.warning("Please upload a CSV file with insurance claim data.")
#             return

#     st.divider()  # Adds a visual separator

#     # Display the chosen section with improved titles and organization
#     if '📊 Explore Data' in action:
#         st.markdown("<h2 class='section-header'>📊 Data Exploration & Analysis</h2>", unsafe_allow_html=True)
        
#         # Add description
#         st.markdown("""
#         Explore the insurance claims dataset to understand patterns, distributions, and relationships 
#         between different variables. This analysis helps identify factors that influence claim resolution time.
#         """)
        
#         # Data overview 
#         with st.expander("Dataset Overview", expanded=True):
#             eda(df)
        
#         # Distribution plots
#         with st.expander("Time to Resolution Distribution", expanded=True):
#             plot_time_to_resolution_distribution(df)
        
#         # Correlation analysis
#         with st.expander("Correlation Analysis", expanded=True):
#             corelation_map(df)

#     elif '🧠 Train & Predict' in action:
#         st.markdown("<h2 class='section-header'>🧠 Model Training & Prediction</h2>", unsafe_allow_html=True)
        
#         st.markdown("""
#         Train a machine learning model to predict the time it takes to resolve insurance claims.
#         The model uses Random Forest, which is well-suited for this type of regression problem.
#         """)
        
#         predict_customer_info(df)

#     elif '🔍 Explainable AI' in action:
#         st.markdown("<h2 class='section-header'>🔍 Explainable AI Insights</h2>", unsafe_allow_html=True)
        
#         st.markdown("""
#         Understand why the model makes specific predictions using modern explainable AI techniques.
#         These insights help insurance professionals identify key factors affecting resolution time
#         and make more informed decisions.
#         """)
        
#         # Call our new explain_model function
#         explain_model(df, model)

# if __name__ == "__main__":
#     main()

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
