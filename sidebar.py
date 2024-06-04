import streamlit as st


def initialize():
    """Initialize the sidebar content"""
    st.sidebar.header("Table Of Content")
    st.sidebar.markdown(
        """
    1. [Introduction](#introduction)
    2. [Objectives](#objectives)
    3. [Definition](#definition)
    4. [How PCA Works](#how-pca-works)
    5. [PCA Demo with Iris Dataset](#pca-demo-with-iris-dataset)
    6. [What Is Kernel PCA?](#what-is-kernel-pca)
    7. [Conclusion](#conclusion)
    8. [Questions](#questions)
    """,
        unsafe_allow_html=True,
    )
