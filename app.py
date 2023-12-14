import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
from tensorflow.keras.models import load_model  # Add this import for the TensorFlow model

# title
st.title('PulmoScan Ai')

# header
st.header('please upload an x-ray scan of a chest')

# Load your TensorFlow model here
model = load_model('model2_updated.h5')


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('p2.jpg')


def classify(image, model, class_mapping):
    """
    This function takes an image, a model, and a dictionary mapping class indices to class names,
    and returns the predicted class and confidence score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_mapping (dict): A dictionary mapping class indices to class names.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # greyscale
    gray_image = np.mean(normalized_image_array, axis=-1, keepdims=True)

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 1), dtype=np.float32)
    data[0] = gray_image

    # make prediction
    prediction = model.predict(data)

    # Use np.argmax to get the index with the maximum value
    index = np.argmax(prediction)

    # Get class name and confidence score
    predicted_class_name = class_mapping.get(index, "Unknown")
    confidence_score = prediction[0][index]

    return predicted_class_name, confidence_score


file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Define class mapping directly in the code
    class_mapping = {0: 'PNEUMONIA', 1: 'NORMAL', 2: 'COVID'}

    # classify image
    predicted_class_name, conf_score = classify(image, model, class_mapping)

    # write classification
    st.write("## {}".format(predicted_class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
