import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('p2.jpg')


#title 
st.title('PulmoScan Ai')

#header 
st.header('please upload an x-ray scan of a chest')

#upload
file = st.file_uploader('', type = ['jpeg', 'jpg', 'png'])

#classifier
model_path = 'model2_updated.h5'
model = load_model(model_path)


#Class Names
with open('labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

#Images
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))