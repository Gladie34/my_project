import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

def eda(df):
    st.subheader("📄 Raw Dataset Preview")
    st.write(df.head())

    # EDA Section
    st.subheader("📊 Exploratory Data Analysis")

    if st.checkbox("Show Dataset Info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    if st.checkbox("Show Missing Values"):
        st.write(df.isnull().sum())

    # Distribution of Age, Gender, Income, Region
    st.markdown("### Customer Distributions by Age, Gender, Income, and Region")
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Distribution of Age
    sns.histplot(df['Age'], bins=20, kde=True, color='g', ax=axs[0])
    axs[0].set_title('Distribution of Age')

    # Distribution of Gender
    gender_countplot = sns.countplot(x='Gender', data=df, palette='bright', saturation=0.95, ax=axs[1])
    for p in gender_countplot.patches:
        gender_countplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axs[1].set_title('Distribution of Gender')

    # Income distribution
    income_countplot = sns.countplot(x='Income', data=df, order=df['Income'].value_counts().head(7).index, ax=axs[2])
    for p in income_countplot.patches:
        income_countplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axs[2].set_title('Income Distribution')

    # Distribution of Region
    region_countplot = sns.countplot(x='Region', data=df, order=df['Region'].value_counts().head(7).index, ax=axs[3])
    for p in region_countplot.patches:
        region_countplot.annotate(format(p.get_height(), '.0f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axs[3].set_title('Distribution of Region')

    plt.tight_layout()
    st.pyplot(fig)



def plot_time_to_resolution_distribution(data, column='TimeToResolutionDays', title='Distribution of Time to Resolution (Days)'):
    # Create figure and axis for Streamlit
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort the x-axis categories
    order = sorted(data[column].unique())

    # Set style and palette
    sns.set_style("whitegrid")
    sns.countplot(x=column, data=data, palette='viridis', order=order, ax=ax)

    # Annotate bars with counts
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.0f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Set titles and labels
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Time to Resolution (Days)', fontsize=13)
    ax.set_ylabel('Number of Claims', fontsize=13)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)


def corelation_map(df):
    st.markdown("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])  # Select numeric columns only
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()