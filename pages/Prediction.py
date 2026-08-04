import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

from tensorflow.keras.models import load_model

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import DenseNet201

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as dense_preprocess


# ===============================================================
# PAGE CONFIGURATION
# ===============================================================

st.set_page_config(
    page_title="Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Histopathology Cancer Prediction")

st.markdown("""
Upload a histopathology image and allow the trained machine learning model
to classify the tissue sample as **Benign** or **Malignant**.

This prediction is generated using the deep learning feature extractor and
classification model selected in the sidebar.
""")

st.divider()

# ===============================================================
# DATA PATHS
# ===============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    ROOT,
    "deployment_assets",
    "model"
)

# ===============================================================
# SIDEBAR
# ===============================================================

st.sidebar.header("⚙ Prediction Settings")

feature_model = st.sidebar.selectbox(
    "Deep Feature Extractor",
    (
        "VGG16",
        "ResNet50",
        "DenseNet201"
    )
)

# ===============================================================
# AVAILABLE CLASSIFIERS
# ===============================================================

available_classifiers = (
    "CNN",
    "CNN-SVM",
    "Random Forest",
    "Gradient Boosting"
)

classifier = st.sidebar.selectbox(
    "Classification Model",
    available_classifiers
)

uploaded_image = st.sidebar.file_uploader(
    "Upload Histopathology Image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)
# ===============================================================
# LOAD FEATURE EXTRACTOR
# ===============================================================

@st.cache_resource
def load_feature_extractor(name):

    if name == "VGG16":

        model = VGG16(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

        return model, vgg_preprocess, 512


    elif name == "ResNet50":

        model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

        return model, resnet_preprocess, 2048


    else:

        model = DenseNet201(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

        return model, dense_preprocess, 1920


# ===============================================================
# IMAGE PREPROCESSING
# ===============================================================

def preprocess_image(image, preprocess_function):

    image = image.resize((224,224))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_function(image)

    return image


# ===============================================================
# LOAD CLASSIFICATION MODEL
# ===============================================================

def load_classifier():

    if classifier == "CNN":

        filename = f"cnn_{feature_model.lower()}.keras"

        model = load_model(
            os.path.join(
                MODEL_PATH,
                filename
            )
        )

        return model


    elif classifier == "CNN-SVM":

        filename = f"cnn_svm_{feature_model.lower()}.pkl"


    elif classifier == "Random Forest":

        filename = f"rf_{feature_model.lower()}.pkl"


    else:

        filename = f"gb_{feature_model.lower()}.pkl"


    model = joblib.load(
        os.path.join(
            MODEL_PATH,
            filename
        )
    )

    return model


# ===============================================================
# PREDICTION
# ===============================================================

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    left, right = st.columns([1,1])

    # ----------------------------------------------------------
    # Uploaded image
    # ----------------------------------------------------------

    with left:

        st.subheader("🖼 Uploaded Histopathology Image")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"Filename: {uploaded_image.name}"
        )


    # ----------------------------------------------------------
    # Processing pipeline
    # ----------------------------------------------------------

    with right:

        st.subheader("⚙ Prediction Pipeline")

        st.info(f"""

**Step 1:** Image Uploaded

⬇

**Step 2:** Image resized to **224 × 224 pixels**

⬇

**Step 3:** Deep Features extracted using **{feature_model}**

⬇

**Step 4:** Feature Vector Generated

Dimension: **{load_feature_extractor(feature_model)[2]}**

⬇

**Step 5:** Classification performed using **{classifier}**

⬇

**Step 6:** Final Cancer Prediction Generated

""")

    extractor, preprocess_function, feature_dimension = load_feature_extractor(feature_model)

    with st.spinner("Analysing histopathology image..."):

        processed_image = preprocess_image(
            image,
            preprocess_function
        )

        extracted_features = extractor.predict(
            processed_image,
            verbose=0
        )

        model = load_classifier()

        if classifier == "CNN":

            malignant_probability = float(
                model.predict(
                    extracted_features,
                    verbose=0
                )[0][0]
            )

        else:

            malignant_probability = float(
                model.predict_proba(
                    extracted_features
                )[0][1]
            )

        benign_probability = 1 - malignant_probability

        prediction = (
            "Malignant"
            if malignant_probability >= 0.5
            else
            "Benign"
        )

        confidence = max(
            benign_probability,
            malignant_probability
        )
        # ===============================================================
# RESULTS
# ===============================================================

st.divider()

st.header("🧾 Prediction Results")

metric1, metric2, metric3 = st.columns(3)

with metric1:

    if prediction == "Malignant":
        st.error("### 🔴 Prediction")
        st.error("## MALIGNANT")
    else:
        st.success("### 🟢 Prediction")
        st.success("## BENIGN")

with metric2:

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

with metric3:

    if confidence >= 0.90:
        level = "Very High"
    elif confidence >= 0.70:
        level = "High"
    else:
        level = "Moderate"

    st.metric(
        "Confidence Level",
        level
    )

st.divider()

# ===============================================================
# PROBABILITY DISTRIBUTION
# ===============================================================

left, right = st.columns([1,1])

with left:

    st.subheader("📊 Probability Distribution")

    st.progress(float(malignant_probability))

    probability_df = pd.DataFrame({

        "Class":[
            "Benign",
            "Malignant"
        ],

        "Probability":[
            round(benign_probability*100,2),
            round(malignant_probability*100,2)
        ]

    })

    st.dataframe(
        probability_df,
        use_container_width=True,
        hide_index=True
    )

with right:

    st.subheader("🤖 Model Information")

    st.info(f"""
**Deep Feature Extractor**

{feature_model}

**Feature Dimension**

{feature_dimension}

**Classification Model**

{classifier}

**Decision Threshold**

50%

**Prediction Confidence**

{confidence*100:.2f}%
""")

st.divider()

# ===============================================================
# CLINICAL INTERPRETATION
# ===============================================================

st.header("🩺 Clinical Interpretation")

if prediction == "Malignant":

    st.error(f"""
### 🔴 Predicted Class: Malignant

The uploaded tissue image has been classified as **Malignant**.

### What does this mean?

A **malignant tumour** is generally considered **cancerous**.

Malignant cells usually:

• grow rapidly

• invade surrounding tissues

• may spread (metastasize) to other parts of the body

This model estimates that there is a **{malignant_probability*100:.2f}% probability**
that the uploaded histopathology image belongs to the malignant class.

### Confidence Interpretation

The prediction confidence is **{confidence*100:.2f}%**.

Higher confidence indicates that the model is more certain about its prediction.
""")

else:

    st.success(f"""
### 🟢 Predicted Class: Benign

The uploaded tissue image has been classified as **Benign**.

### What does this mean?

A **benign tumour** is generally **non-cancerous**.

Benign tissues usually:

• grow slowly

• remain localized

• do not spread to distant organs

This model estimates that there is a **{benign_probability*100:.2f}% probability**
that the uploaded histopathology image belongs to the benign class.

### Confidence Interpretation

The prediction confidence is **{confidence*100:.2f}%**.

Higher confidence indicates that the model is more certain about its prediction.
""")

st.divider()

# ===============================================================
# HOW THE MODEL MADE THE DECISION
# ===============================================================

st.header("🧠 How the Prediction Was Generated")

st.markdown(f"""
The uploaded histopathology image first passed through the **{feature_model}**
deep convolutional neural network, which extracted high-level visual
representations from the tissue image.

Instead of classifying the image directly, these deep feature vectors were
supplied to the selected **{classifier}** classifier, which produced the final
prediction by analysing the extracted feature representation.

This two-stage approach combines powerful deep feature extraction with
traditional machine learning classification.
""")

st.divider()

# ===============================================================
# DISCLAIMER
# ===============================================================

st.warning("""
## ⚠ Important Disclaimer

This dashboard was developed as part of an MSc dissertation for
research and educational purposes.

The predictions generated by this system **must not** be interpreted as a
medical diagnosis.

Although the models were trained using publicly available histopathology
datasets, real clinical diagnosis requires comprehensive pathological
assessment by qualified healthcare professionals.

Therefore, this system should only be used to demonstrate the application
of deep learning and machine learning techniques for automated
histopathology image classification.
""")

st.success("✅ Prediction completed successfully.")