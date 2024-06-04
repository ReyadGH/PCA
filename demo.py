import streamlit as st
import plotly.express as px
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd


def iris_demo():
    """Display the PCA demo using the Iris dataset"""
    st.header("PCA Demo with Iris Dataset")

    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Standardize the data
    X_standardized = StandardScaler().fit_transform(X)

    # Add sliders for PCA components
    n_components = st.slider("Number of PCA Components", 1, 3, 2)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_standardized)

    # Create a DataFrame with the PCA results
    columns = [f"PC{i+1}" for i in range(n_components)]
    df = pd.DataFrame(data=principalComponents, columns=columns)
    df["target"] = y
    df["species"] = df["target"].apply(lambda x: target_names[x])

    # Plot the PCA result based on the number of components
    if n_components == 1:
        fig = px.scatter(
            df,
            x="PC1",
            y=[0] * len(df),
            color="species",
            labels={"PC1": "Principal Component 1"},
            title="PCA of Iris Dataset",
        )
    elif n_components == 2:
        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color="species",
            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
            title="PCA of Iris Dataset",
        )
    elif n_components == 3:
        fig = px.scatter_3d(
            df,
            x="PC1",
            y="PC2",
            z="PC3",
            color="species",
            labels={
                "PC1": "Principal Component 1",
                "PC2": "Principal Component 2",
                "PC3": "Principal Component 3",
            },
            title="PCA of Iris Dataset",
        )

    st.plotly_chart(fig)
