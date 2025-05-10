import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page configuration
st.set_page_config(page_title="Image Intensity Transformation", layout="wide")

st.sidebar.image("logo.png", use_container_width=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Transform Image", "About", "How It Works"])

# ----------------------------- Page: Home -----------------------------
if page == "Home":
    st.title("Welcome to the Image Intensity Transformation App")
    st.markdown("""
    This web app allows you to apply various **intensity transformations** on images for visualization and learning purposes.

    ### 🔧 Features:
    - Negative transformation  
    - Logarithmic transformation  
    - Gamma correction  
    - Histogram visualization  
    - Image download  

    Navigate using the sidebar to get started!
    """)

# ------------------------ Page: Transform Image -----------------------
elif page == "Transform Image":
    st.title("Transform Your Image")

    # -- Transformation functions --
    def negative(img):
        return 255 - img

    def log_transform(img):
        img = img.astype(np.float32)
        # grayscale
        if len(img.shape) == 2:
            c = 255.0 / np.log(1 + np.max(img))
            log_img = c * np.log(1 + img)
            return np.uint8(np.clip(log_img, 0, 255))
        # color
        result = np.zeros_like(img, dtype=np.uint8)
        for i in range(3):
            channel = img[:, :, i]
            c = 255.0 / np.log(1 + np.max(channel))
            log_channel = c * np.log(1 + channel)
            result[:, :, i] = np.uint8(np.clip(log_channel, 0, 255))
        return result

    def gamma_correction(img, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def plot_histogram(image, title="Histogram"):
        plt.figure()
        if len(image.shape) == 2:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(hist, color='black')
        else:
            for i, col in enumerate(('r', 'g', 'b')):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
        plt.title(title)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        st.pyplot(plt.gcf())
        plt.close()

    # -- UI: file uploader --
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Read and convert
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Optional grayscale
        grayscale = st.checkbox("Convert to Grayscale")
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        st.image(img, caption="Original Image", use_container_width=True)

        # Choose transformation
        transformation = st.selectbox("Choose a Transformation", 
                                      ("Negative", "Log Transform", "Gamma Correction"))
        if transformation == "Negative":
            result = negative(img)
        elif transformation == "Log Transform":
            result = log_transform(img)
        else:  # Gamma Correction
            gamma = st.slider("Gamma", 0.1, 5.0, 1.0)
            result = gamma_correction(img, gamma)

        # Display result
        st.image(result, caption="Transformed Image", use_container_width=True)

        # Histograms
        st.subheader("Histogram Comparison")
        with st.expander("Original Histogram"):
            plot_histogram(img, "Original Image Histogram")
        with st.expander("Transformed Histogram"):
            plot_histogram(result, "Transformed Image Histogram")

        # Download button
        result_pil = Image.fromarray(result)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download Transformed Image", data=byte_im,
                           file_name="transformed.png", mime="image/png")

# -------------------------- Page: About -----------------------------
elif page == "About":
    st.title("About This App")
    st.markdown("""
    This app is a simple demonstration of **image intensity transformation techniques** using:

    - 🐍 Python  
    - 📷 OpenCV  
    - 📊 Matplotlib  
    - 🌐 Streamlit  

    Designed for learners and researchers in image processing.

    **Version:** 1.0  
    **Owner:**B P Logesh    
    """)

# --------------------- Page: How It Works -------------------------
elif page == "How It Works":
    st.title("How It Works")
    st.markdown("""
    ### 🔄 Intensity Transformations Explained

    **🖤 Negative Transformation:**  
    Inverts pixel values:  
    $$s = 255 - r$$

    **📈 Logarithmic Transformation:**  
    Enhances dark regions:  
    $$s = c \cdot \log(1 + r)$$

    **🌗 Gamma Correction:**  
    Adjusts brightness/darkness:  
    $$s = c \cdot r^{\gamma}$$

    - γ < 1 brightens  
    - γ > 1 darkens
    """)
