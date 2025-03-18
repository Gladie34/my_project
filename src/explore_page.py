import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def eda(df):
    st.subheader("📄 Raw Dataset Preview")
    st.write(df.head())

    # EDA Section
    st.subheader("📊 Exploratory Data Analysis")

    if st.checkbox("Show Dataset Info"):
        buffer = []
        df.info(buf=buffer)
        s = "\n".join(buffer)
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

    st.markdown("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])  # Select numeric columns only
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()