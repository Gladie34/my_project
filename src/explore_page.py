import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def eda(df):
    # Top metrics overview
    st.markdown("### 📈 Key Dataset Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Records</div>
            </div>
            """.format(len(df)),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">{:.1f}</div>
                <div class="metric-label">Avg Resolution Time (Days)</div>
            </div>
            """.format(df['TimeToResolutionDays'].mean()),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Features</div>
            </div>
            """.format(df.shape[1]),
            unsafe_allow_html=True
        )
    
    with col4:
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """.format(missing_percentage),
            unsafe_allow_html=True
        )

    # Raw dataset preview with better formatting
    st.markdown("### 📄 Raw Dataset Preview")
    
    # Add column filter
    if st.checkbox("Select specific columns to view"):
        selected_columns = st.multiselect("Choose columns:", df.columns.tolist(), default=df.columns.tolist()[:5])
        st.dataframe(df[selected_columns].head(10), use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)

    # Dataset info with better formatting
    if st.checkbox("Show Dataset Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data Types and Non-Null Counts")
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.code(info_str, language="")
        
        with col2:
            st.markdown("#### 📏 Numerical Summary Statistics")
            st.dataframe(df.describe(), use_container_width=True)

    # Missing values visualization with better formatting
    if st.checkbox("Show Missing Values Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Missing Values Count")
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ['Column', 'Missing Count']
            missing_values = missing_values[missing_values['Missing Count'] > 0]
            
            if len(missing_values) > 0:
                fig = px.bar(
                    missing_values, 
                    x='Column', 
                    y='Missing Count',
                    title='Missing Values by Column',
                    color='Missing Count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")
                
        with col2:
            st.markdown("#### Missing Values Detail")
            st.dataframe(df.isnull().sum().reset_index().rename(
                columns={'index': 'Column', 0: 'Missing Count'}
            ), use_container_width=True)

    # Enhanced distributions with Plotly for better interactivity
    st.markdown("### 👥 Customer Demographics Analysis")
    
    # Age distribution (using Plotly)
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig = px.histogram(
            df, 
            x='Age', 
            nbins=20, 
            title='Age Distribution',
            labels={'Age': 'Age (years)', 'count': 'Number of Customers'},
            opacity=0.8,
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Gender distribution
        gender_counts = df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender',
            title='Gender Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Income and Region distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Income distribution
        income_counts = df['Income'].value_counts().reset_index().head(10)
        income_counts.columns = ['Income', 'Count']
        
        fig = px.bar(
            income_counts,
            x='Income',
            y='Count',
            title='Income Distribution (Top 10)',
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Region distribution
        region_counts = df['Region'].value_counts().reset_index().head(10)
        region_counts.columns = ['Region', 'Count']
        
        fig = px.bar(
            region_counts,
            x='Region',
            y='Count',
            title='Region Distribution (Top 10)',
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def plot_time_to_resolution_distribution(data, column='TimeToResolutionDays', title='Distribution of Time to Resolution (Days)'):
    st.markdown("### ⏱️ Claim Resolution Time Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Distribution", "Box Plot", "Cumulative Distribution"])
    
    with tab1:
        # Modern histogram with KDE using Plotly
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
        
        fig.update_layout(
            xaxis_title='Time to Resolution (Days)',
            yaxis_title='Number of Claims',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Box plot with points
        fig = px.box(
            data,
            y=column,
            points="all",
            title="Box Plot of Time to Resolution",
            labels={column: 'Time to Resolution (Days)'},
            color_discrete_sequence=['#9b59b6']
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Cumulative distribution
        sorted_data = data[column].sort_values().reset_index(drop=True)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig = px.line(
            x=sorted_data, 
            y=cumulative,
            title="Cumulative Distribution of Resolution Time",
            labels={'x': 'Time to Resolution (Days)', 'y': 'Cumulative Probability'},
            color_discrete_sequence=['#e74c3c']
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add descriptive statistics
    st.markdown("#### Resolution Time Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average (Days)", f"{data[column].mean():.1f}")
    
    with col2:
        st.metric("Median (Days)", f"{data[column].median():.1f}")
    
    with col3:
        st.metric("Min (Days)", f"{data[column].min()}")
    
    with col4:
        st.metric("Max (Days)", f"{data[column].max()}")

def corelation_map(df):
    st.markdown("### 🔗 Correlation Analysis")
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Create correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Use Plotly for an interactive heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap of Numerical Features"
    )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature pair analysis
    st.markdown("#### Explore Relationships Between Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("Select X-axis feature:", numeric_df.columns)
    
    with col2:
        y_feature = st.selectbox("Select Y-axis feature:", numeric_df.columns, index=1)
    
    color_by = st.selectbox("Color points by:", [None] + df.columns.tolist())
    
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color=color_by if color_by else None,
        opacity=0.7,
        title=f"Relationship between {x_feature} and {y_feature}",
        labels={x_feature: x_feature, y_feature: y_feature},
        size_max=10,
        color_continuous_scale='Viridis' if color_by and df[color_by].dtype.kind in 'bifc' else None
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show correlation value
    if x_feature != y_feature:
        correlation = df[x_feature].corr(df[y_feature])
        st.info(f"Correlation coefficient between {x_feature} and {y_feature}: {correlation:.4f}")