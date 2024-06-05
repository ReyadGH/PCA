import streamlit as st


def introduction():
    """Display the introduction section"""
    st.header("Introduction")
    st.markdown(
        "Welcome to our exploration of Principal Components Analysis (PCA)! 🎉 Let's dive into the fascinating world of data dimensionality reduction and uncover how PCA transforms complex datasets into simpler forms, helping us gain deeper insights."
    )


def objectives():
    """Display the objectives section"""
    st.header("Objectives")
    st.markdown("""
    By the end of this lesson, you will:
    - Understand what PCA is and its purpose.
    - Learn how PCA works and the mathematical basics.
    - Explore the concept of Kernel PCA.
    - See PCA in action with a hands-on demo.
    - Recognize the pros and cons of using PCA.
    - Be prepared to apply PCA to your own data projects!
    """)


def definition():
    """Display the definition section"""
    st.header("Definition")
    st.markdown("""
    **What is PCA?**
    
    PCA is a dimensionality reduction technique that transforms input features into their principal components. It converts a set of observations of possibly correlated features into a set of values of linearly uncorrelated features.

    **Goal**: Map the data from the original high-dimensional space to a lower-dimensional space that captures as much of the variation in the data as possible. It's like finding the hidden structure in a cluttered room!    
        
    """)
    st.image("image\pca.gif")

    st.markdown("""
    - **Maximum variance**
        
    - **Minimum Error**
                """)


def kernel_pca():
    """Display the Kernel PCA section"""
    st.header("What Is Kernel PCA (كبسة)?")
    st.markdown("""
    While traditional PCA is highly effective for linear data transformations, it may not capture the underlying structure of complex, nonlinear datasets. To handle this problem, we introduce kernel principal component analysis (KPCA). KPCA relies on the intuition that many datasets that are not linearly separable in their current dimension can be linearly separable by projecting them into a higher-dimensional space.

    In one sentence, KPCA takes our dataset, maps it into some higher dimension, and then performs PCA in that new dimensional space.

    To gain a better intuition of this process, let's examine the following example:
    """)

    st.subheader("PCA Example")
    st.markdown("""
    On the left side, we have our two-dimensional dataset. This dataset consists of two classes: red and blue. Blue classes are points on a donut-shaped cluster, while red points are in the circle that is in the center of that donut. It's clear that this dataset is not linearly separable, which means that no straight line can separate these two classes.
    
    """)

    st.image("image\kpca1.png")

    st.markdown("""
    
    If we apply PCA and reduce our data dimension from 2D to 1D, we'll get points on one axis in the image on the right. The goal of PCA is to simplify high-dimensional datasets while retaining essential information. But, as we can see in the right figure, points are now mixed, and we cannot separate or cluster them.
    """)

    st.subheader("KPCA Example")
    st.markdown("""
    Now, let's transform our dataset from 2D to 3D with a simple function:
    """)

    st.latex(r"""
    f(v) = f((x, y)) = (x, y, 2x^{2} + 2y^{2})
    """)

    st.markdown("""
    where \( v = (x, y) \) is a point in 2D space. Visually, this transformation looks like this:
    """)

    st.image("image\kpca2.png")

    st.markdown("""
    Following the logic o   f KPCA, we transformed our dataset to a higher dimension, from 2D to 3D. Now we can try to apply PCA and reduce the dimensionality from 3D to 1D. Moreover, let's transform our data first to 2D and after to 1D. The results are below:
    """)

    st.image("image\kpca3.png")

    st.markdown("""
    Unlike the initial attempt, the projection onto the first principal component now effectively separates the red and blue points. This outcome highlights the true potential of Kernel PCA over the standard PCA.
    """)

    st.subheader("Kernel Functions")
    st.markdown("""
    A kernel function is a function that we use to transform the original data into a higher-dimensional space where it becomes linearly separable. Generally, kernels are commonly employed in various machine learning models such as support vector machines (SVM), kernel ridge regression (KRR), kernelized k-means, and others. Some of the most popular kernel functions are:
    """)

    st.latex(r"""
    \text{Linear:} \quad K(u,v) = u \cdot v
    """)
    st.latex(r"""
    \text{Polynomial:} \quad K(u,v) = (\gamma u \cdot v + c)^d
    """)
    st.latex(r"""
    \text{GaussRBF:} \quad K(u,v) = \exp{(-\gamma \|u-v\|^2)}
    """)
    st.latex(r"""
    \text{Sigmoid:} \quad K(u,v) = \tanh{(\gamma u \cdot v + c)}
    """)

    st.markdown("""
    Usually, it's not easy to determine which kernel is better to use. One good option is to treat the kernel and its parameters as hyperparameters of the whole model and apply cross-validation to find them together with other hyperparameters.
    """)


# def demo():
#     """Display the demo section"""
#     st.header("Demo")
#     st.markdown(
#         "In this section, we'll see PCA in action! Stay tuned for an interactive demo where we'll transform a high-dimensional dataset into a lower-dimensional space, revealing hidden patterns and simplifying our analysis. 🚀"
#     )


def conclusion():
    """Display the conclusion section"""
    st.header("Conclusion")
    st.markdown("""
    PCA is a powerful technique for dimensionality reduction, especially useful for visualizing and processing high-dimensional data. It simplifies the complexity of data, making it easier to interpret and analyze. However, it has limitations regarding interpretability and sensitivity to feature scaling.

    Kernel PCA extends the capabilities of traditional PCA by allowing it to handle nonlinear data transformations, making it a valuable tool for more complex datasets.

    Thank you for joining this journey into PCA and Kernel PCA. Remember, the true power of PCA lies in its ability to uncover hidden structures within your data. Keep exploring and happy analyzing! 🌟
    """)


def questions():
    """Display the questions section"""
    st.header("Questions")
    st.markdown(
        "Feel free to ask any questions you have about PCA or Kernel PCA! Let's discuss and explore together. 🤔"
    )
