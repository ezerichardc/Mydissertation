import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Histopathology Cancer Classification Dashboard",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset Overview",
        "Model Evaluation",
        "Prediction",
        "Conclusions"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("🩺 Machine Learning-Based Histopathology Cancer Classification Dashboard")

    st.markdown("---")

    st.header("Project Overview")

    st.write("""
This dashboard presents the complete machine learning pipeline developed
for automated histopathology cancer classification.

The project compares multiple deep feature extraction techniques and
machine learning classifiers to identify the most effective model for
binary cancer classification.
""")

    st.header("Objectives")

    st.markdown("""
- Develop an automated cancer classification system.
- Compare multiple deep feature extractors.
- Compare multiple machine learning classifiers.
- Evaluate model performance using standard classification metrics.
- Provide an interactive prediction interface.
""")

    st.header("Models Developed")

    st.markdown("""
**Feature Extractors**

- VGG16
- ResNet50
- DenseNet201

**Classification Models**

- CNN
- CNN-SVM
- Random Forest
- Gradient Boosting
""")

    st.success("Dashboard successfully loaded.")