"""
app.py
------
Handwritten Character Recognition — Streamlit front-end.

Features
  • Draw a character on an interactive canvas, or upload an image.
  • Live CNN prediction with a confidence bar chart of the top classes.
  • Switch between the MNIST (digits) and EMNIST (characters) models.
  • Shows the exact 28x28 tensor the model actually sees.

Run:  streamlit run app.py
"""
import os
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

from utils.preprocess import preprocess_image
from models.labels import get_labels

# --------------------------------------------------------------------------- #
#  Page config & theme
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide",
)

# Custom CSS — ink-on-paper identity, restrained accent
st.markdown(
    """
    <style>
      :root {
        --ink:        #1a1a2e;
        --paper:      #f6f4ee;
        --accent:     #e94560;
        --accent-2:   #16213e;
        --muted:      #6b6b7b;
      }
      .stApp { background: var(--paper); }
      .hcr-title {
        font-family: 'Georgia', serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.5px;
        margin-bottom: 0;
      }
      .hcr-sub {
        font-family: 'Georgia', serif;
        font-style: italic;
        color: var(--muted);
        margin-top: 0.2rem;
        font-size: 1.05rem;
      }
      .hcr-rule {
        height: 3px;
        background: linear-gradient(90deg, var(--accent), transparent);
        border: none; margin: 0.4rem 0 1.4rem 0;
      }
      .pred-box {
        background: var(--accent-2);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        color: #fff;
      }
      .pred-char {
        font-family: 'Georgia', serif;
        font-size: 5.5rem;
        line-height: 1;
        font-weight: 700;
        color: #fff;
      }
      .pred-conf { font-size: 1.1rem; color: #ffd6dd; margin-top: .4rem; }
      .stem { color: var(--muted); font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  Model loading (cached)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "MNIST — Digits (0-9)": os.path.join(HERE, "models", "mnist_cnn.keras"),
    "EMNIST — Characters (A-Z, 0-9)": os.path.join(HERE, "models", "emnist_cnn.keras"),
}
DATASET_KEY = {
    "MNIST — Digits (0-9)": "mnist",
    "EMNIST — Characters (A-Z, 0-9)": "emnist",
}


@st.cache_resource(show_spinner="Loading model…")
def get_model(path):
    return load_model(path)


def predict(model, tensor, labels):
    probs = model.predict(tensor, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = [(labels[int(i)], float(probs[int(i)])) for i in order[:5]]
    return labels[int(order[0])], float(probs[order[0]]), top


# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
st.markdown('<p class="hcr-title">✍️ Handwritten Character Recognition</p>',
            unsafe_allow_html=True)
st.markdown('<p class="hcr-sub">Draw a character — a convolutional neural '
            'network reads it in real time.</p>', unsafe_allow_html=True)
st.markdown('<hr class="hcr-rule">', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Model", list(MODEL_PATHS.keys()))
    input_mode = st.radio("Input method", ["Draw", "Upload image"])
    stroke_width = st.slider("Brush width", 8, 35, 18)
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Your strokes are captured at 280×280.\n"
        "2. The image is cropped, scaled to 20px, padded to 28×28 and "
        "centered by mass — the MNIST recipe.\n"
        "3. The CNN outputs a probability for each class.",
    )
    st.markdown('<p class="stem">Tip: draw large and centered for best '
                'results.</p>', unsafe_allow_html=True)

model_path = MODEL_PATHS[model_choice]
labels = get_labels(DATASET_KEY[model_choice])

if not os.path.exists(model_path):
    st.warning(
        f"The model file `{os.path.basename(model_path)}` was not found. "
        "Train it first with `python train.py --dataset "
        f"{DATASET_KEY[model_choice]}`. Falling back to the MNIST model if "
        "available."
    )
    model_path = MODEL_PATHS["MNIST — Digits (0-9)"]
    labels = get_labels("mnist")

model = get_model(model_path) if os.path.exists(model_path) else None

# --------------------------------------------------------------------------- #
#  Main two-column layout
# --------------------------------------------------------------------------- #
col_input, col_result = st.columns([1, 1], gap="large")

tensor = None

with col_input:
    st.subheader("Input")
    if input_mode == "Draw":
        canvas = st_canvas(
            fill_color="#000000",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280, height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas.image_data is not None and canvas.image_data[..., :3].sum() > 0:
            pil = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
            # Canvas already has white ink on black -> no inversion needed
            tensor = preprocess_image(pil, invert=False, center=True)
    else:
        up = st.file_uploader("Upload a character image",
                              type=["png", "jpg", "jpeg", "bmp"])
        if up is not None:
            pil = Image.open(up).convert("RGB")
            st.image(pil, caption="Uploaded image", width=180)
            # Uploads are usually dark ink on light paper -> invert
            tensor = preprocess_image(pil, invert=True, center=True)

with col_result:
    st.subheader("Prediction")
    if tensor is not None and model is not None:
        char, conf, top = predict(model, tensor, labels)
        st.markdown(
            f'<div class="pred-box"><div class="pred-char">{char}</div>'
            f'<div class="pred-conf">{conf*100:.1f}% confidence</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("##### Top predictions")
        st.bar_chart(
            {lbl: p for lbl, p in top},
            horizontal=True, color="#e94560",
        )
        with st.expander("What the model sees (28×28)"):
            disp = (tensor.reshape(28, 28) * 255).astype("uint8")
            st.image(Image.fromarray(disp).resize((140, 140), Image.NEAREST),
                     caption="Normalised input tensor")
    else:
        st.info("Draw a character or upload an image to see the prediction.")

st.markdown('<hr class="hcr-rule">', unsafe_allow_html=True)
st.markdown('<p class="stem">Built with TensorFlow + Streamlit · CNN trained '
            'on MNIST / EMNIST · Educational project.</p>',
            unsafe_allow_html=True)
