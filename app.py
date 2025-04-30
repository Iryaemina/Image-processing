import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Title
st.title("Image Intensity Transformations")

# Transformation Functions
def negative(img):
    return 255 - img

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img.astype(np.float32))
    return np.clip(log_img, 0, 255).astype(np.uint8)

def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def plot_histogram(image, title="Histogram"):
    plt.figure()
    if len(image.shape) == 2:  # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    else:
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    st.pyplot(plt.gcf())
    plt.close()

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Grayscale Option
    grayscale = st.checkbox("Convert to Grayscale")
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    st.image(img, caption="Original Image", use_column_width=True)

    # Transformation Selection
    transformation = st.selectbox("Choose a Transformation", 
                                  ("Negative", "Log Transform", "Gamma Correction"))

    if transformation == "Negative":
        result = negative(img)
    elif transformation == "Log Transform":
        result = log_transform(img)
    elif transformation == "Gamma Correction":
        gamma = st.slider("Gamma", 0.1, 5.0, 1.0)
        result = gamma_correction(img, gamma)

    st.image(result, caption="Transformed Image", use_column_width=True)

    # Histogram Comparison
    st.subheader("Histogram Comparison")
    plot_histogram(img, "Original Image Histogram")
    plot_histogram(result, "Transformed Image Histogram")

    # Download Button
    result_pil = Image.fromarray(result)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("Download Transformed Image", data=byte_im, file_name="transformed.png", mime="image/png")
