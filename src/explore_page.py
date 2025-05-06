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
            <li><strong>Demographics Analysis</strong>: Analyze demographics like gender and age</li>
            <li><strong>Feature Importance</strong>: Discover key factors that influence resolution time</li>
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
        if st.checkbox("Select specific columns to view", label_visibility="visible"):
            selected_columns = st.multiselect(
                "Choose columns:", 
                df.columns.tolist(), 
                default=list(df.columns)[:min(5, len(df.columns))],
                label_visibility="visible"
            )
            st.dataframe(df[selected_columns].head(10), use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying dataset preview: {str(e)}")
    
    try:
        if st.checkbox("Show Dataset Information", label_visibility="visible"):
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

def analyze_demographics(df):
    st.markdown("### Demographics Analysis")
    
    # Check if the necessary columns exist
    has_gender = "Gender" in df.columns
    has_age = "Age" in df.columns
    
    if not (has_gender or has_age):
        st.warning("Neither Gender nor Age columns found in the dataset. Cannot create demographics charts.")
        return
    
    col1, col2 = st.columns(2)
    
    # Gender Distribution Chart
    if has_gender:
        with col1:
            try:
                # Create gender counts
                gender_counts = df['Gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']
                
                # Create gender bar chart
                fig = px.bar(
                    gender_counts,
                    x='Gender',
                    y='Count',
                    title='Distribution by Gender',
                    color='Gender',
                    text='Count',
                    color_discrete_sequence=['#1E88E5', '#FFC107', '#4CAF50']
                )
                fig.update_layout(
                    xaxis_title="Gender",
                    yaxis_title="Number of Claims"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Gender by resolution time
                if "TimeToResolutionDays" in df.columns:
                    avg_by_gender = df.groupby('Gender')['TimeToResolutionDays'].mean().reset_index()
                    avg_by_gender.columns = ['Gender', 'Avg Resolution Time (Days)']
                    
                    fig2 = px.bar(
                        avg_by_gender,
                        x='Gender',
                        y='Avg Resolution Time (Days)',
                        title='Average Resolution Time by Gender',
                        color='Gender',
                        text_auto='.1f',
                        color_discrete_sequence=['#1E88E5', '#FFC107', '#4CAF50']
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating gender charts: {str(e)}")
    
    # Age Distribution Chart
    if has_age:
        with col2:
            try:
                # Bin ages into groups
                age_bins = [0, 18, 30, 45, 60, 75, 100]
                age_labels = ['<18', '18-30', '31-45', '46-60', '61-75', '75+']
                
                df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
                age_group_counts = df['AgeGroup'].value_counts().sort_index().reset_index()
                age_group_counts.columns = ['Age Group', 'Count']
                
                # Create age group bar chart
                fig = px.bar(
                    age_group_counts,
                    x='Age Group',
                    y='Count',
                    title='Distribution by Age Group',
                    color='Age Group',
                    text='Count',
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig.update_layout(
                    xaxis_title="Age Group",
                    yaxis_title="Number of Claims"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Age group by resolution time
                if "TimeToResolutionDays" in df.columns:
                    avg_by_age = df.groupby('AgeGroup')['TimeToResolutionDays'].mean().reset_index()
                    avg_by_age.columns = ['Age Group', 'Avg Resolution Time (Days)']
                    
                    fig2 = px.bar(
                        avg_by_age,
                        x='Age Group',
                        y='Avg Resolution Time (Days)',
                        title='Average Resolution Time by Age Group',
                        color='Age Group',
                        text_auto='.1f',
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating age charts: {str(e)}")

def feature_importance(df):
    st.markdown("### Feature Importance")
    
    try:
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            st.warning("Not enough numeric columns for feature importance analysis. Need at least 2 numeric columns.")
            return
        
        # Find the most correlated features with time to resolution
        if "TimeToResolutionDays" in numeric_df.columns:
            st.markdown("#### Top Features Correlated with Resolution Time")
            corr_matrix = numeric_df.corr()
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
            
            # Add explanation of correlation
            st.markdown("""
            **Interpretation of Feature Correlations:**
            - **Positive values (blue)**: As this feature increases, resolution time tends to increase
            - **Negative values (red)**: As this feature increases, resolution time tends to decrease
            - **Larger absolute values**: Stronger relationship with resolution time
            """)
    except Exception as e:
        st.error(f"Error creating feature importance chart: {str(e)}")

def run_exploration(df):
    # Show dataset info
    eda(df)
    
    # Demographics Analysis
    with st.expander("Demographics Analysis", expanded=True):
        analyze_demographics(df)
    
    # Resolution Time Analysis
    if "TimeToResolutionDays" in df.columns:
        with st.expander("Resolution Time Distribution", expanded=True):
            plot_time_to_resolution_distribution(df)
    else:
        st.warning("Cannot create time distribution plots: 'TimeToResolutionDays' column not found")
    
    # Feature Importance
    with st.expander("Feature Importance", expanded=True):
        feature_importance(df)