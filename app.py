import streamlit as st

st.set_page_config(page_title="H5 Kernel Lab", layout="wide")
st.title("H5 Kernel Lab")

st.markdown(
    "This application is a small toolkit for exploring convolutional kernels in "
    "Keras `.h5` models and their CSV exports. Use the pages in the sidebar to:\n\n"
    "- Inspect model layers and export convolution kernels to CSV\n"
    "- Analyze symmetry scores and score distributions from kernel CSV files\n"
    "- Recondition 3×3 kernels based on a condition-number threshold\n"
    "- Experiment with random or manually entered kernels using the same analysis pipeline\n\n"
    "Choose a page from the sidebar to get started."
)
