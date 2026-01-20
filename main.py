import streamlit as st
import os
import numpy as np
import pickle
from numpy.linalg import norm
import tensorflow
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

# App title and description
st.set_page_config(page_title="Fashion Recommendation System", layout="wide", initial_sidebar_state="expanded")
st.title("🛍️ Fashion Recommendation System")
st.markdown("""
Welcome to the **Fashion Recommendation System**!  
Upload an image of a fashion item, and this app will recommend similar items based on our advanced machine learning model.  
Enjoy discovering your next favorite style! 😊
""")

# ------------------------------
# DUMMY EMBEDDINGS (for demo deployment)
import pickle
import numpy as np
import os

if not os.path.exists('embeddings.pkl') or not os.path.exists('filenames.pkl'):
    n_images = 15
    embedding_size = 2048
    feature_list = np.random.rand(n_images, embedding_size)
    filenames = [
    'images/1541.jpg','images/1542.jpg','images/1543.jpg','images/1548.jpg','images/1549.jpg',
    'images/2111.jpg','images/2112.jpg','images/2114.jpg','images/2115.jpg','images/2116.jpg',
    'images/22759.jpg','images/22766.jpg','images/22767.jpg','images/22768.jpg','images/22769.jpg'
    ]

    pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
    pickle.dump(filenames, open('filenames.pkl', 'wb'))
# ------------------------------


# Load pre-trained model and data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# Functions for feature extraction, recommendations, and file handling
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        print(e)
        return 0

# Sidebar for navigation
st.sidebar.header("Navigation")
st.sidebar.markdown("""
- **About**: Learn how the system works.
- **Contact**: Get in touch for queries or feedback.
""")
st.sidebar.markdown("### Contact")
st.sidebar.info("""
For any queries or feedback, feel free to reach out:  
📧 **[anubhavpatwal2929@gmail.com](mailto:anubhavpatwal2929@gmail.com)**  
""")

# File upload section
st.markdown("### Upload Your Fashion Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_img = Image.open(uploaded_file)
        st.image(display_img, caption="Uploaded Image", use_column_width=True)

        # Feature extraction and recommendation
        st.markdown("### Recommendations")
        with st.spinner("Analyzing your image and fetching recommendations..."):
            features = extract_features(os.path.join("uploads", uploaded_file.name), model)
            indices = recommend(features, feature_list)
            st.success("Here are your recommendations:")

        # Display recommendations in a grid format
        st.markdown("#### Similar Items:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[0][i]], use_column_width=True)
    else:
        st.error("Error uploading your image. Please try again.")

# Footer
st.markdown("---")
st.markdown("""
**Developed by Anubhav Patwal (https://github.com/dai-29)**  
Powered by Python, TensorFlow, and Streamlit.
""")



