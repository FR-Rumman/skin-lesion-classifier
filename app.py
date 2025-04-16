import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import gdown  # 🔁 Added for downloading model from Google Drive

# 🧠 Grad-CAM Utility Functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=plt.cm.jet):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)
    heatmap = np.array(heatmap)

    colormap_heatmap = colormap(heatmap)
    colormap_heatmap = Image.fromarray((colormap_heatmap[:, :, :3] * 255).astype(np.uint8))

    overlayed = Image.blend(image, colormap_heatmap, alpha)
    return overlayed

# 🩺 Must be the FIRST Streamlit command!
st.set_page_config(page_title="Skin Lesion Classifier", layout="centered", page_icon="🩺")

# ✅ Download and load model from Google Drive
@st.cache_resource
def load_model():
    model_path = "densenet201_model.keras"
    if not os.path.exists(model_path):
        file_id = "1byM6sy9ZF4lTnuKq9uN6_gGTmSu4eCya"  # Your actual file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# 🔤 Class names
class_names = ["Chickenpox", "Measles", "Monkeypox", "Normal"]

# 🖼️ App UI
st.title("Skin Lesion Classifier")
st.write("Choose an option to provide a skin lesion image:")

# 🚦 Let user choose input method
input_method = st.radio("Select input method:", ["Upload Image", "Take a Picture"])

# 📁 File uploader
uploaded_file = None
camera_image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

elif input_method == "Take a Picture":
    st.markdown("""
    <style>
        .camera-grid-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-template-rows: repeat(4, 1fr);
            grid-gap: 2px;
            z-index: 10;
        }
        .camera-grid-overlay div {
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="camera-grid-overlay"></div>', unsafe_allow_html=True)
    camera_image = st.camera_input("📸 Take a photo using your webcam")

# ✅ Use either uploaded file or camera image
image_source = uploaded_file or camera_image

if image_source is not None:
    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="🖼️ Selected Image", use_container_width=True)

    image_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("🔍 Classifying..."):
        prediction = model.predict(img_array)

        if prediction.shape[1] > 1 and np.max(prediction) > 1:
            prediction = tf.nn.softmax(prediction).numpy()

        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        st.success(f"Prediction: **{class_names[predicted_class]}** ({confidence:.2%} confidence)")

        # 📊 Probability scores
        st.subheader("📊 Prediction Probabilities")
        pred_df = pd.DataFrame({
            "Class": class_names,
            "Probability": prediction[0]
        }).sort_values(by="Probability", ascending=False)

        st.bar_chart(pred_df.set_index("Class"))

        # 🧠 Grad-CAM Visualization
        st.subheader("🧪 Grad-CAM Heatmap")
        last_conv_layer_name = "conv5_block32_concat"

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=predicted_class)
        cam_image = overlay_heatmap(heatmap, image_resized)

        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.image(image_resized, caption="🖼️ Original Image", use_container_width=True)
        with col2:
            st.image(cam_image, caption="🔍 Grad-CAM Heatmap", use_container_width=True)

        with st.expander("🔬 View Raw Prediction Values"):
            st.dataframe(pred_df)
