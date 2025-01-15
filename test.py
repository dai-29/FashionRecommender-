import pickle

import cv2
import numpy as np
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Process the input image
img = image.load_img('1541.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_image_array = np.expand_dims(img_array, axis=0)
preprocessed_image = preprocess_input(expanded_image_array)
result = model.predict(preprocessed_image).flatten()
normalized_result = result / norm(result)

# Perform Nearest Neighbor search
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices [0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img, (206,206)))
    cv2.waitKey(0)