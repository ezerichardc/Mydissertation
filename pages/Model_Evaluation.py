import streamlit as st
import pandas as pd
import os
from PIL import Image

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Model Evaluation",
    layout="wide"
)

st.title("📈 Model Evaluation Dashboard")

st.markdown(
"""
This section presents the performance of all machine learning models
developed for histopathology cancer classification using three deep
feature extraction techniques.
"""
)

st.markdown("---")

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVAL = os.path.join(
    BASE_DIR,
    "deployment_assets",
    "Model_Evaluation"
)

summary_file = os.path.join(
    EVAL,
    "Model_Evaluation_Summary.csv"
)

# ==========================================================
# LOAD RESULTS
# ==========================================================

results = pd.read_csv(summary_file)

# ==========================================================
# BEST MODEL
# ==========================================================

best = results.sort_values(
    "Accuracy",
    ascending=False
).iloc[0]

st.header("🏆 Best Performing Model")

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{best['Accuracy']:.3f}"
)

c2.metric(
    "Precision",
    f"{best['Precision']:.3f}"
)

c3.metric(
    "Recall",
    f"{best['Recall (Sensitivity)']:.3f}"
)

c4.metric(
    "ROC-AUC",
    f"{best['ROC-AUC']:.3f}"
)

st.success(
f"""
Feature Extractor: **{best['Feature Extractor']}**

Classification Model: **{best['Classification Model']}**
"""
)

st.markdown("---")

# ==========================================================
# MODEL PERFORMANCE TABLE
# ==========================================================

st.header("Performance Comparison")

st.dataframe(
    results.sort_values(
        "Accuracy",
        ascending=False
    ),
    use_container_width=True
)

st.markdown("---")

# ==========================================================
# PERFORMANCE CHART
# ==========================================================

st.header("Accuracy Comparison")

chart = (
    results[
        ["Feature Extractor",
         "Classification Model",
         "Accuracy"]
    ]
)

chart["Model"] = (
    chart["Feature Extractor"]
    + " - "
    + chart["Classification Model"]
)

st.bar_chart(
    chart.set_index("Model")["Accuracy"]
)

st.markdown("---")

# ==========================================================
# MODEL SELECTOR
# ==========================================================

st.header("Model Visualisation")

model_names = []

for _,row in results.iterrows():

    model_names.append(

        f"{row['Feature Extractor']} - {row['Classification Model']}"

    )

selected = st.selectbox(

    "Select Model",

    model_names

)

feature = selected.split(" - ")[0]

classifier = selected.split(" - ")[1]

cm_file = os.path.join(

    EVAL,

    f"CM_{classifier}_{feature}.png"

)

roc_file = os.path.join(

    EVAL,

    f"ROC_{classifier}_{feature}.png"

)

col1,col2 = st.columns(2)

with col1:

    st.subheader("Confusion Matrix")

    if os.path.exists(cm_file):

        st.image(

            Image.open(cm_file),

            use_container_width=True

        )

    else:

        st.warning("Confusion matrix not found.")

with col2:

    st.subheader("ROC Curve")

    if os.path.exists(roc_file):

        st.image(

            Image.open(roc_file),

            use_container_width=True

        )

    else:

        st.warning("ROC curve not found.")

st.markdown("---")

# ==========================================================
# MODEL RANKING
# ==========================================================

st.header("Overall Model Ranking")

ranking = (

    results

    .sort_values(

        "Accuracy",

        ascending=False

    )

    .reset_index(drop=True)

)

ranking.index = ranking.index + 1

ranking.index.name = "Rank"

st.dataframe(

    ranking,

    use_container_width=True

)

st.markdown("---")

# ==========================================================
# CONCLUSION
# ==========================================================

st.header("Key Findings")

st.info(
"""
• Twelve machine learning models were evaluated.

• Three deep feature extraction architectures were investigated.

• Four classification algorithms were compared.

• Performance was assessed using Accuracy, Precision,
Recall, Specificity, F1-Score and ROC-AUC.

• The dashboard enables visual comparison of all developed models.
"""
)