import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def negative(img):
    return 255 - img

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    return c * np.log(1 + img)

def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

# Streamlit app
st.title("Intensity Transformations")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Original Image", use_column_width=True)

    transformation = st.selectbox("Choose a Transformation", 
                                  ("Negative", "Log Transform", "Gamma Correction"))

    if transformation == "Negative":
        result = negative(img)
    elif transformation == "Log Transform":
        result = log_transform(img).astype(np.uint8)
    elif transformation == "Gamma Correction":
        gamma = st.slider("Gamma", 0.1, 5.0, 1.0)
        result = gamma_correction(img, gamma)

    st.image(result, caption="Transformed Image", use_column_width=True)
