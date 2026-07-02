"""
train_synthetic.py
------------------
Bootstrap trainer used ONLY to ship a working demo model when the real
MNIST/EMNIST download is unavailable. It renders digits 0-9 in many system
fonts with random rotation/shift/noise and trains the same CNN on them.

On your own machine you should prefer `train.py`, which uses the real MNIST /
EMNIST data and produces materially higher real-world accuracy.
"""
import os, sys, glob, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cnn_model import build_cnn, compile_model

FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/freefont",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/google-fonts",
]

def collect_fonts():
    fonts = []
    for d in FONT_DIRS:
        fonts += glob.glob(os.path.join(d, "*.ttf"))
    random.shuffle(fonts)
    return fonts[:40]

def render_digit(d, font_path, size=28):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    fsize = random.randint(18, 24)
    try:
        font = ImageFont.truetype(font_path, fsize)
    except Exception:
        font = ImageFont.load_default()
    s = str(d)
    bbox = draw.textbbox((0, 0), s, font=font)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (size - w)//2 - bbox[0] + random.randint(-2, 2)
    y = (size - h)//2 - bbox[1] + random.randint(-2, 2)
    draw.text((x, y), s, fill=255, font=font)
    img = img.rotate(random.uniform(-15, 15), resample=Image.BILINEAR)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.8)))
    arr = np.asarray(img, dtype=np.float32)/255.0
    if random.random() < 0.3:
        arr += np.random.normal(0, 0.05, arr.shape)
    return np.clip(arr, 0, 1)

def build_dataset(n_per_class=600):
    fonts = collect_fonts()
    X, y = [], []
    for d in range(10):
        for _ in range(n_per_class):
            X.append(render_digit(d, random.choice(fonts)))
            y.append(d)
    X = np.array(X).reshape(-1, 28, 28, 1)
    y = np.array(y, dtype="int64")
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def main():
    print("Rendering synthetic digit dataset...")
    X, y = build_dataset(700)
    split = int(0.9*len(X))
    x_tr, x_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    print("Train", x_tr.shape, "Test", x_te.shape)

    model = compile_model(build_cnn(num_classes=10))
    model.fit(x_tr, y_tr, validation_data=(x_te, y_te),
              epochs=12, batch_size=128, verbose=2)
    acc = model.evaluate(x_te, y_te, verbose=0)[1]
    print(f"Synthetic test accuracy: {acc:.4f}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "models", "mnist_cnn.keras")
    model.save(out)
    print("Saved ->", out)

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    main()
