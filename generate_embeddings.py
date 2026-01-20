import os
import pickle
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from tqdm import tqdm

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = model = GlobalMaxPooling2D()(model.output)  # Global Max Pooling
from tensorflow.keras.models import Model
model = Model(inputs=model.input, outputs=model)

# Folder me images ka path
img_folder = 'images'

# Filenames
filenames = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.jpg','.jpeg','.png'))]

# Extract features
feature_list = []
for file in tqdm(filenames):
    img = image.load_img(file, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)
    result = model.predict(preprocessed).flatten()
    normalized = result / norm(result)
    feature_list.append(normalized)

# Save embeddings
pickle.dump(np.array(feature_list), open('embeddings.pkl','wb'))
pickle.dump(filenames, open('filenames.pkl','wb'))

print("✅ Real embeddings generated and saved!")
