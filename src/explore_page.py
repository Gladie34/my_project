import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.express as px

def eda(df):
    # Display a concise overview card
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; 
    border-left: 5px solid #1E88E5; margin-bottom: 20px;'>
        <h4 style='margin-top: 0'>This exploration page offers:</h4>
        <ul>
            <li><strong>Dataset Overview</strong>: View key metrics and basic statistics</li>
            <li><strong>Resolution Time Analysis</strong>: Explore distribution of claim resolution times</li>
            <li><strong>Feature Relationships</strong>: Discover correlations between factors that influence resolution time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Key Dataset Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    # Check if TimeToResolutionDays exists before trying to use it
    if "TimeToResolutionDays" in df.columns:
        with col2:
            st.metric("Avg Resolution Time (Days)", f"{df['TimeToResolutionDays'].mean():.1f}")
    else:
        with col2:
            st.metric("Avg Resolution Time", "N/A")
            st.warning("Column 'TimeToResolutionDays' not found in dataset")
    
    with col3:
        st.metric("Features", df.shape[1])
    
    with col4:
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("Missing Values", f"{missing_percentage:.1f}%")
    
    # Show dataset preview
    st.markdown("### Raw Dataset Preview")
    try:
        if st.checkbox("Select specific columns to view"):
            selected_columns = st.multiselect("Choose columns:", df.columns.tolist(), 
                                             default=list(df.columns)[:min(5, len(df.columns))])
            st.dataframe(df[selected_columns].head(10), use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying dataset preview: {str(e)}")
    
    try:
        if st.checkbox("Show Dataset Information"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Data Types & Non-Null Counts")
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                st.code(info_str)
            with col2:
                st.markdown("#### Numerical Summary Stats")
                st.dataframe(df.describe(), use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying dataset information: {str(e)}")

def plot_time_to_resolution_distribution(data, column='TimeToResolutionDays', 
                                         title='Distribution of Time to Resolution (Days)'):
    st.markdown("### Claim Resolution Time Analysis")
    
    # Check if the column exists
    if column not in data.columns:
        st.warning(f"Column '{column}' not found in the dataset. Cannot create distribution plots.")
        return
    
    try:
        tab1, tab2, tab3 = st.tabs(["Distribution", "Box Plot", "Cumulative Distribution"])
        
        with tab1:
            fig = px.histogram(
                data, 
                x=column,
                nbins=30,
                marginal="box",
                title=title,
                labels={column: 'Time to Resolution (Days)', 'count': 'Number of Claims'},
                opacity=0.7,
                color_discrete_sequence=['#2ecc71']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.box(
                data,
                y=column,
                points="all",
                title="Box Plot of Time to Resolution",
                labels={column: 'Time to Resolution (Days)'},
                color_discrete_sequence=['#9b59b6']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            sorted_data = data[column].sort_values().reset_index(drop=True)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            fig = px.line(
                x=sorted_data, 
                y=cumulative,
                title="Cumulative Distribution of Resolution Time",
                labels={'x': 'Time to Resolution (Days)', 'y': 'Cumulative Probability'},
                color_discrete_sequence=['#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating distribution plots: {str(e)}")

def correlation_map(df):
    st.markdown("### Feature Relationships")
    
    try:
        # Get only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            st.warning("Not enough numeric columns for correlation analysis. Need at least 2 numeric columns.")
            return
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap of Numerical Features"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Find the most correlated features with time to resolution
        if "TimeToResolutionDays" in numeric_df.columns:
            st.markdown("#### Top Features Correlated with Resolution Time")
            corr_with_target = corr_matrix["TimeToResolutionDays"].drop("TimeToResolutionDays").sort_values(ascending=False)
            
            fig = px.bar(
                x=corr_with_target.values,
                y=corr_with_target.index,
                orientation="h",
                labels={"x": "Correlation Coefficient", "y": "Feature"},
                title="Features Correlation with Resolution Time",
                color=corr_with_target.values,
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating correlation map: {str(e)}")

def run_exploration(df):
    # Only use one header for the page - remove this line to prevent duplication
    # st.markdown("<h2 class='section-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    # Show dataset info
    eda(df)
    
    if "TimeToResolutionDays" in df.columns:
        with st.expander("Resolution Time Distribution", expanded=True):
            plot_time_to_resolution_distribution(df)
    else:
        st.warning("Cannot create time distribution plots: 'TimeToResolutionDays' column not found")
    
    with st.expander("Feature Correlations", expanded=True):
        correlation_map(df)