import streamlit as st
from sections import (
    introduction,
    objectives,
    definition,
    kernel_pca,
    conclusion,
    questions,
)
from demo import iris_demo
from work_plots import how_it_works

import sidebar

# Initialize the sidebar
sidebar.initialize()

col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
col2.image("image\pca.png", width=300, clamp=True)


# Title
st.title("Principal Components Analysis (PCA)")

# Display each section
introduction()
objectives()
definition()
how_it_works()
iris_demo()
kernel_pca()
conclusion()
questions()
