import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
import os

# ------------------ Setup ------------------
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        color: #1f2937;
    }
    h1, h2, h3, .stMarkdown, .stSlider, .stImage, .stFileUploader, .stInfo {
        color: #1f2937 !important;
    }
    .recommendation-img {
        border-radius: 8px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .recommendation-caption {
        text-align: center;
        margin-top: 8px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: left;'>🛍️ Fashion Visual Recommender</h1>", unsafe_allow_html=True)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

model = load_model()

# ------------------ Load Precomputed Features ------------------
@st.cache_data
def load_features():
    with open("resnet_features.pkl", "rb") as f:
        all_features, all_image_paths = pickle.load(f)
    return all_features, all_image_paths

all_features, all_image_paths = load_features()

# ------------------ Utility Functions ------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img, verbose=0)
    return features.flatten() / np.linalg.norm(features.flatten())

def get_top_n_similar(query_feature, all_features, all_image_paths, top_n=4):
    similarities = [1 - cosine(query_feature, feat) for feat in all_features]
    similarity_scores = list(enumerate(similarities))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarity_scores[:top_n]]
    return [all_image_paths[idx] for idx in top_indices]

# ------------------ Sidebar Inputs ------------------
with st.sidebar:
    st.header("Upload Product Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    num_recommendations = st.slider("Number of recommendations", 1, 8, 4)

    if uploaded_file:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.markdown("##### Preview:")
        st.image(input_image, width=150)

# ------------------ Main Area ------------------
if uploaded_file:
    temp_path = "temp_uploaded_image.jpg"
    input_image.save(temp_path, format="JPEG")
    preprocessed_img = preprocess_image(temp_path)
    input_features = extract_features(model, preprocessed_img)
    similar_image_paths = get_top_n_similar(input_features, all_features, all_image_paths, top_n=num_recommendations)
    os.remove(temp_path)

    st.subheader("How this Recommender System Works")
    st.markdown("""
    This AI system uses a pretrained **ResNet50** deep learning model to extract visual features from fashion images.
    It then compares the uploaded image against a database of fashion items using **cosine similarity** between image embeddings.
    The most visually similar items are recommended in real-time.
    """)

    st.subheader("Recommended for You")
    cols_per_row = 4
    rows = (num_recommendations + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx
            if img_idx < len(similar_image_paths):
                with cols[col_idx]:
                    img = Image.open(similar_image_paths[img_idx]).resize((220, 220))
                    st.image(img, use_container_width=False, width=220)
                    st.markdown(f"<div class='recommendation-caption'>Recommendation {img_idx + 1}</div>", unsafe_allow_html=True)
else:
    st.info("👈 Use the sidebar to upload a fashion image and get started.")
