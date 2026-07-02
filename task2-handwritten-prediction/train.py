"""
train.py
--------
Train the CNN on MNIST (digits) or EMNIST (characters) and save the model to
models/<dataset>_cnn.keras together with a training-history plot.

Usage
-----
    python train.py --dataset mnist  --epochs 15
    python train.py --dataset emnist --epochs 25

EMNIST requires the `emnist` pip package (pip install emnist). MNIST ships with
Keras. Both download automatically the first time you run them.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import build_cnn, compile_model


def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    return x_tr, y_tr, x_te, y_te, 10


def load_emnist():
    # pip install emnist
    from emnist import extract_training_samples, extract_test_samples
    x_tr, y_tr = extract_training_samples("balanced")
    x_te, y_te = extract_test_samples("balanced")
    return x_tr, y_tr, x_te, y_te, 47


def prepare(x, y):
    x = x.astype("float32") / 255.0
    x = x.reshape(-1, 28, 28, 1)
    return x, y.astype("int64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "emnist"], default="mnist")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    if args.dataset == "mnist":
        x_tr, y_tr, x_te, y_te, n_cls = load_mnist()
    else:
        x_tr, y_tr, x_te, y_te, n_cls = load_emnist()

    x_tr, y_tr = prepare(x_tr, y_tr)
    x_te, y_te = prepare(x_te, y_te)

    print(f"Train: {x_tr.shape}  Test: {x_te.shape}  Classes: {n_cls}")

    model = compile_model(build_cnn(num_classes=n_cls))
    model.summary()

    # Light augmentation — helps real-world canvas input generalise
    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1,
    )
    datagen.fit(x_tr)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-5),
    ]

    history = model.fit(
        datagen.flow(x_tr, y_tr, batch_size=args.batch),
        validation_data=(x_te, y_te),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(out_dir, "models", f"{args.dataset}_cnn.keras")
    model.save(model_path)
    print(f"Saved model -> {model_path}")

    # Plot training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "assets", f"{args.dataset}_history.png")
    fig.savefig(plot_path, dpi=120)
    print(f"Saved training curves -> {plot_path}")


if __name__ == "__main__":
    main()
