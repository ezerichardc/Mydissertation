import streamlit as st
import pandas as pd
import os
import random
from PIL import Image

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(page_title="Dataset Overview", layout="wide")

st.title("📊 Dataset Overview")

st.markdown("---")

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASSETS = os.path.join(BASE_DIR, "deployment_assets")

PROCESSED = os.path.join(ASSETS, "Processed_Data")

SAMPLE_IMAGES = os.path.join(ASSETS, "Sample_Images")

train_csv = os.path.join(PROCESSED, "train.csv")
validation_csv = os.path.join(PROCESSED, "validation.csv")
test_csv = os.path.join(PROCESSED, "test.csv")

# ==========================================================
# LOAD DATA
# ==========================================================

train_df = pd.read_csv(train_csv)
validation_df = pd.read_csv(validation_csv)
test_df = pd.read_csv(test_csv)

# ==========================================================
# DATASET SUMMARY
# ==========================================================

st.header("Dataset Summary")

total_images = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

num_classes = train_df["label"].nunique()

col1, col2, col3 = st.columns(3)

col1.metric("Total Images", f"{total_images:,}")
col2.metric("Training Images", f"{len(train_df):,}")
col3.metric("Validation Images", f"{len(validation_df):,}")

col4, col5 = st.columns(2)

col4.metric("Testing Images", f"{len(test_df):,}")
col5.metric("Number of Classes", num_classes)

st.markdown("---")

# ==========================================================
# DATASET SPLIT
# ==========================================================

st.header("Dataset Split")

split_df = pd.DataFrame({

    "Dataset":[
        "Training",
        "Validation",
        "Testing"
    ],

    "Images":[
        len(train_df),
        len(validation_df),
        len(test_df)
    ]

})

st.dataframe(split_df, use_container_width=True)

st.bar_chart(
    split_df.set_index("Dataset")
)

st.markdown("---")

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

st.header("Training Class Distribution")

class_distribution = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

distribution_df = pd.DataFrame({

    "Class":class_distribution.index.astype(str),

    "Images":class_distribution.values

})

st.dataframe(distribution_df, use_container_width=True)

st.bar_chart(
    distribution_df.set_index("Class")
)

st.markdown("---")

# ==========================================================
# SAMPLE HISTOPATHOLOGY IMAGES
# ==========================================================

st.header("Sample Histopathology Images")

image_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
)

image_files = [

    f for f in os.listdir(SAMPLE_IMAGES)

    if f.lower().endswith(image_extensions)

]

samples = random.sample(
    image_files,
    min(6, len(image_files))
)

columns = st.columns(3)

for i, file in enumerate(samples):

    image = Image.open(
    os.path.join(SAMPLE_IMAGES, file)
    )

    columns[i % 3].image(
        image,
        caption=file,
        use_container_width=True
    )

st.markdown("---")

# ==========================================================
# FEATURE EXTRACTION SUMMARY
# ==========================================================

st.header("Deep Feature Extraction Models")

feature_df = pd.DataFrame({

    "Feature Extractor":[
        "VGG16",
        "ResNet50",
        "DenseNet201"
    ],

    "Feature Dimension":[
        512,
        2048,
        1920
    ]

})

st.dataframe(
    feature_df,
    use_container_width=True
)

st.success("Dataset overview loaded successfully.")