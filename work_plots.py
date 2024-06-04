import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def how_it_works():
    """Display the how it works section with Plotly graphs"""
    st.header("How PCA Works")
    st.markdown("""
    **How PCA Works**

    - PCA finds a sequence of linear combinations of features that have maximum variance and are uncorrelated.
    - PCA is an unsupervised learning method, meaning it doesn’t use class labels.

    **General Idea**:
    Principal components are the directions of maximum variance, minimizing the information loss when performing projection or compression onto these principal components. Imagine capturing the essence of a high-dimensional dataset in just a few strokes! 🎨
    """)

    # Display the code for generating example data (not normalized)
    st.code(
        """
    import numpy as np
    np.random.seed(42)
    mean = [10, 20]
    cov = [[5, 2], [2, 2]]  # covariance matrix
    x, y = np.random.multivariate_normal(mean, cov, 500).T
    """,
        language="python",
    )

    # Generate example data (not normalized)
    np.random.seed(42)
    mean = [10, 20]
    cov = [[5, 2], [2, 2]]  # covariance matrix
    x, y = np.random.multivariate_normal(mean, cov, 500).T

    # Create a scatter plot of the data
    fig = px.scatter(
        x=x, y=y, labels={"x": "X1", "y": "X2"}, title="Scatter Plot of Original Data"
    )
    st.plotly_chart(fig)

    st.markdown("""
    Let's break down the mathematical steps involved in PCA:

    1. **Standardization**: PCA is sensitive to the relative scaling of the original features, so standardize the features prior to PCA. Think of this as leveling the playing field for all features.
    """)
    st.latex(r"""
    X_{standardized} = \frac{X - \mu}{\sigma}
    """)

    # Display the code for standardizing the data
    st.code(
        """
    # Standardize the data
    x_standardized = (x - np.mean(x)) / np.std(x)
    y_standardized = (y - np.mean(y)) / np.std(y)
    """,
        language="python",
    )

    # Standardize the data
    x_standardized = (x - np.mean(x)) / np.std(x)
    y_standardized = (y - np.mean(y)) / np.std(y)

    fig = px.scatter(
        x=x_standardized,
        y=y_standardized,
        labels={"x": "X1 (standardized)", "y": "X2 (standardized)"},
        title="Scatter Plot of Standardized Data",
    )
    st.plotly_chart(fig)

    st.markdown("""
    2. **Compute Covariance Matrix**: Calculate the covariance matrix of the features. This matrix captures how the features vary together.
    """)
    st.latex(r"""
    \Sigma = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \mu)(X_i - \mu)^T
    """)

    # Display the code for computing the covariance matrix
    st.code(
        """
    # Compute the covariance matrix
    cov_matrix = np.cov(x_standardized, y_standardized)
    """,
        language="python",
    )

    # Compute the covariance matrix
    cov_matrix = np.cov(x_standardized, y_standardized)
    st.write("Covariance Matrix:")
    st.write(cov_matrix)

    st.markdown("""
    3. **Eigendecomposition**: Decompose the covariance matrix into eigenvectors and eigenvalues. Eigenvectors determine the directions of the principal components, while eigenvalues indicate their magnitude.
    """)
    st.latex(r"""
    \Sigma v = \lambda v
    """)

    # Display the code for eigendecomposition
    st.code(
        """
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    """,
        language="python",
    )

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Show eigenvectors and eigenvalues
    st.write("Eigenvectors:")
    st.write(eigenvectors)
    st.write("Eigenvalues:")
    st.write(eigenvalues)

    st.markdown("""
    4. **Choose Principal Components**: Select the top k eigenvectors based on the eigenvalues. These top k components capture the most variance.
    """)

    # Display the code for plotting eigenvectors
    st.code(
        """
    # Plot eigenvectors on top of standardized data
    fig = px.scatter(x=x_standardized, y=y_standardized, labels={'x': 'X1 (standardized)', 'y': 'X2 (standardized)'}, title="Scatter Plot with Eigenvectors")
    fig.add_trace(go.Scatter(x=[0, eigenvectors[0,0]], y=[0, eigenvectors[1,0]], mode='lines', name='First PC'))
    fig.add_trace(go.Scatter(x=[0, eigenvectors[0,1]], y=[0, eigenvectors[1,1]], mode='lines', name='Second PC'))
    """,
        language="python",
    )

    # Plot eigenvectors on top of standardized data
    fig = px.scatter(
        x=x_standardized,
        y=y_standardized,
        labels={"x": "X1 (standardized)", "y": "X2 (standardized)"},
        title="Scatter Plot with Eigenvectors",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, eigenvectors[0, 0]],
            y=[0, eigenvectors[1, 0]],
            mode="lines",
            name="First PC",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, eigenvectors[0, 1]],
            y=[0, eigenvectors[1, 1]],
            mode="lines",
            name="Second PC",
        )
    )
    st.plotly_chart(fig)

    st.markdown("""
    5. **Feature Transformation**: Transform the original feature space to the new subspace using the selected principal components. This is where the magic happens! 🪄
    """)
    st.latex(r"""
    Z = XW
    """)

    # Display the code for feature transformation
    st.code(
        """
    # Project the data onto the principal components
    pca_data = np.dot(np.array([x_standardized, y_standardized]).T, eigenvectors)
    """,
        language="python",
    )

    # Project the data onto the principal components
    pca_data = np.dot(np.array([x_standardized, y_standardized]).T, eigenvectors)

    # Plot the transformed data
    fig = px.scatter(
        x=pca_data[:, 0],
        y=pca_data[:, 1],
        labels={"x": "PC1", "y": "PC2"},
        title="Scatter Plot of Transformed Data (PCA)",
    )
    st.plotly_chart(fig)
