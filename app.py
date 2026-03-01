import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.generator import Generator
from models.cnn import TrafficCNN
import os
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Traffic Sign AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #1e1e2f, #2c3e50);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 30px;
}

/* PASSPORT SIZE IMAGE */
[data-testid="stImage"] img {
    width: 220px !important;
    height: 280px !important;
    object-fit: contain;
    border-radius: 14px;
    border: 2px solid rgba(255,255,255,0.3);
    box-shadow: 0 6px 18px rgba(0,0,0,0.5);
    display: block;
    margin-left: auto;
    margin-right: auto;
}

/* Label */
.section-label {
    text-align: center;
    margin-top: 10px;
    font-size: 18px;
    color: #dddddd;
}

/* Result Card */
.result-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.6);
    width: 55%;
    margin: 30px auto 10px auto;
    text-align: center;
}

/* Prediction */
.prediction {
    font-size: 26px;
    font-weight: bold;
    color: #00ffcc;
}

/* Confidence */
.confidence {
    font-size: 18px;
    margin-top: 8px;
    color: #eeeeee;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5

class_names = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4"
}

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    generator = Generator().to(device)
    generator.load_state_dict(torch.load("generator.pth", map_location=device))
    generator.eval()

    classifier = TrafficCNN(NUM_CLASSES).to(device)
    classifier.load_state_dict(torch.load("classifier.pth", map_location=device))
    classifier.eval()

    return generator, classifier

generator, classifier = load_models()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- TITLE ----------------
st.markdown(
    '<div class="title">🚦 Traffic Sign Enhancement & Classification</div>',
    unsafe_allow_html=True
)

image_path = "enhanced.png"

if os.path.exists(image_path):

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Low resolution
    low_res = torch.nn.functional.interpolate(img_tensor, scale_factor=0.5)

    # Enhance
    with torch.no_grad():
        enhanced = generator(low_res)

    # Classify
    with torch.no_grad():
        outputs = classifier(enhanced)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_id = predicted.item()
    confidence_score = confidence.item() * 100

    enhanced_img = (enhanced.squeeze().cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()

    # ---------------- CENTERED SIDE-BY-SIDE IMAGES ----------------
    space1, col1, col2, space2 = st.columns([1, 1, 1, 1])

    with col1:
        st.image(image)
        st.markdown(
            '<div class="section-label">Original Image</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.image(enhanced_img)
        st.markdown(
            '<div class="section-label">Enhanced Image</div>',
            unsafe_allow_html=True
        )

    # ---------------- RESULT ----------------
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="prediction">Prediction: {class_names[class_id]}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="confidence">Confidence: {confidence_score:.2f}%</div>',
        unsafe_allow_html=True
    )
    st.progress(confidence_score / 100)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Run visualize_srgan.py first to generate enhanced.png")