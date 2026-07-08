import os
import sys

# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
import streamlit as st
from PIL import Image
import tempfile
import os

from inference.search import search_image

st.set_page_config(
    page_title="Product Matching Framework",
    layout="wide"
)

st.title("🛍️ Product Matching Framework")
st.write("Upload a product image to retrieve visually similar products.")

uploaded_file = st.file_uploader(
    "Choose a product image",
    type=["jpg", "jpeg", "png", "JPG"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("Query Image")
        st.image(image, use_container_width=True)

    # Save uploaded image temporarily
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    image.save(temp_path)

    with st.spinner("Searching similar products..."):

        results = search_image(temp_path, top_k=5)

    with col2:

        st.subheader("Top 5 Similar Products")

        cols = st.columns(5)

        for i, result in enumerate(results):

            with cols[i]:

                st.image(
                    result["image_path"],
                    use_container_width=True
                )

                st.caption(
                    f"Distance : {result['distance']:.4f}"
                )