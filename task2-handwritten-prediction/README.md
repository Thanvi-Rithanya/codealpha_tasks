# ✍️ Handwritten Character Recognition

A deep-learning project that identifies handwritten digits and characters using a
Convolutional Neural Network (CNN), with an interactive Streamlit web interface
where you can draw a character and get a live prediction.

---

## 1. Features

- **Draw or upload** a character and get an instant prediction.
- **CNN classifier** trained on MNIST (digits 0–9) and EMNIST (47 character classes).
- **Confidence chart** showing the top-5 candidate classes.
- **Tensor preview** — see the exact 28×28 image the network receives.
- **Extension path** to full word/sentence recognition via a CRNN + CTC model
  (`models/crnn_model.py`).

---

## 2. Project structure

```
hcr/
├── app.py                  # Streamlit front-end
├── train.py                # Train on real MNIST / EMNIST
├── train_synthetic.py      # Bootstrap trainer (ships a working demo model)
├── requirements.txt
├── README.md
├── models/
│   ├── cnn_model.py        # CNN architecture
│   ├── crnn_model.py       # CRNN extension (word recognition)
│   ├── labels.py           # class-index -> character mappings
│   ├── mnist_cnn.keras     # pre-trained demo model (included)
│   └── emnist_cnn.keras    # created when you run train.py --dataset emnist
├── utils/
│   └── preprocess.py       # shared image preprocessing
└── assets/                 # training curves saved here
```

---

## 3. Environment setup

Requires **Python 3.10–3.12**.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> On Apple Silicon, replace `tensorflow` with `tensorflow-macos` and
> `tensorflow-metal` in `requirements.txt`. On a machine without a GPU,
> `tensorflow-cpu` works fine for this project.

---

## 4. Train the models (recommended)

The repository ships with a small demo model so the app runs immediately, but
for real accuracy train on the actual datasets:

```bash
# Digits — reaches ~99.3% test accuracy in ~15 epochs
python train.py --dataset mnist --epochs 15

# Characters — EMNIST balanced (47 classes), ~88% test accuracy
python train.py --dataset emnist --epochs 25
```

Each run saves `models/<dataset>_cnn.keras` and a training-curve plot in
`assets/`.

---

## 5. Run the app

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

### User guide
1. Pick a model in the sidebar (**MNIST** for digits, **EMNIST** for letters).
2. Choose **Draw** or **Upload image**.
3. In Draw mode, write a single character large and centered on the black pad.
4. The prediction, confidence, and top-5 chart update instantly.
5. Expand **"What the model sees"** to inspect the normalised 28×28 input.

**Tips for best accuracy:** draw thick, centered strokes; for uploads use a
clear single character — dark ink on a light background works best.

---

## 6. How it works (pipeline)

```
Canvas / upload  →  grayscale  →  crop to ink  →  resize longest side to 20px
   →  pad to 28×28  →  center by mass  →  scale to [0,1]  →  CNN  →  softmax
```

The CNN is a compact VGG-style network: two convolution blocks (Conv-BN-Conv-BN-
Pool-Dropout) followed by a dense classifier head with dropout and L2
regularisation. See `models/cnn_model.py`.

---

## 7. Extending to words and sentences

Single-character CNNs cannot read whole words because a word is a *variable-length
sequence*. `models/crnn_model.py` provides a CRNN skeleton: a CNN encoder feeds a
bidirectional LSTM, trained with **CTC loss** so the network learns to read a
word strip without per-character segmentation. Train it on the **IAM Handwriting**
or **Synthetic Words** datasets to recognise full words.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit_drawable_canvas` import error | `pip install streamlit-drawable-canvas==0.9.3` |
| Model file not found | run the matching `train.py` command, or use the bundled `mnist_cnn.keras` |
| Predictions look wrong on uploads | ensure the image is a single character; the app inverts uploads assuming dark ink on light paper |
| EMNIST download fails | `pip install emnist`; the first run downloads ~500 MB |
