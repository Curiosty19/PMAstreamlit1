import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np


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


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1


    #greyscale 
    gray_image = np.mean(normalized_image_array, axis= -1, keepdims=True)

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 1), dtype=np.float32)
    data[0] = gray_image

    # make prediction
    prediction = model.predict(data)

    # Load class mapping from the text file
    with open(' ', 'r') as file:
        class_mapping = [line.strip().split() for line in file]

    # Convert indices to integers and create a dictionary
    class_mapping = {int(index): class_name for index, class_name in class_mapping}

    # Use np.argmax to get the index with the maximum value
    index = np.argmax(prediction)

    # Get class name and confidence score
    class_name = class_mapping.get(index, "Unknown")
    confidence_score = prediction[0][index]


    return class_name, confidence_score