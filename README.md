# Fashion Recommendation System

The Fashion Recommendation System is a machine learning-based application that suggests similar fashion items based on an uploaded image. It leverages advanced image processing techniques and a user-friendly interface to enhance the shopping experience and assist users in discovering fashion choices tailored to their preferences.

---

## Features

- **Image Upload**: Users can upload an image of a fashion item (e.g., a shirt, dress, or accessory).
- **Feature Extraction**: Utilizes a pre-trained ResNet50 model for extracting features from the uploaded image.
- **Similarity Search**: Implements K-Nearest Neighbors (KNN) to find and recommend visually similar items from a pre-existing dataset.
- **User-Friendly Interface**: Built using Streamlit for an intuitive and interactive user experience.
- **Real-Time Recommendations**: Delivers top recommendations within seconds.

---

## Tech Stack

- **Python**: Core programming language.
- **TensorFlow/Keras**: For leveraging the ResNet50 model and image processing.
- **NumPy**: For numerical computations and feature normalization.
- **scikit-learn**: To implement the KNN algorithm for similarity search.
- **Streamlit**: For building the web-based application interface.
- **Pillow**: For image processing.
- **Pickle**: For saving and loading preprocessed data (e.g., embeddings and filenames).

---

## How It Works

1. **Image Upload**:
   - Users upload an image of a fashion item through the interface.

2. **Feature Extraction**:
   - The system processes the image using the ResNet50 model to extract features.
   - The extracted features are normalized for better comparison.

3. **Recommendation**:
   - The extracted features are compared with precomputed features (stored embeddings).
   - KNN is used to find the most similar items in the database.
   - The top 5 similar items are displayed as recommendations.

---

## Installation and Usage

### Prerequisites
- Python 3.8 or above
- Required libraries: TensorFlow, scikit-learn, Streamlit, Pillow, NumPy

## Screenshot
![Fashion Recommendation System Screenshot](assets/Screenshot (145).png)

