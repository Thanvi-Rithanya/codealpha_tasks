"""
crnn_model.py  (EXTENSION — word / sequence recognition)
--------------------------------------------------------
Single-character CNNs cannot read a whole word, because a word is a *sequence*
of glyphs of variable length. The standard solution is a CRNN: a CNN extracts a
feature sequence from the image strip, a bidirectional LSTM models the
left-to-right context, and a CTC (Connectionist Temporal Classification) loss
aligns the predicted character sequence to the image without needing per-pixel
labels.

This file is a ready-to-train skeleton you can use to extend the project from
single characters to full words (e.g. on the IAM Handwriting dataset).
"""
from tensorflow.keras import layers, models
import tensorflow as tf


def build_crnn(img_height=32, img_width=128, num_classes=80):
    """CNN + BiLSTM feature encoder producing per-timestep class logits.

    num_classes = size of your character vocabulary + 1 (for the CTC blank).
    """
    inp = layers.Input(shape=(img_height, img_width, 1), name="image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 1))(x)          # keep width resolution

    # Collapse height -> a sequence along the width axis
    new_h = img_height // 8
    new_w = img_width // 4
    x = layers.Reshape((new_w, new_h * 128))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    logits = layers.Dense(num_classes, activation="softmax", name="logits")(x)

    return models.Model(inp, logits, name="CRNN")


class CTCLayer(layers.Layer):
    """Adds the CTC loss during training. Wrap labels + logits with this."""
    def call(self, y_true, y_pred):
        bs = tf.cast(tf.shape(y_true)[0], "int64")
        in_len = tf.cast(tf.shape(y_pred)[1], "int64") * tf.ones((bs, 1), "int64")
        lbl_len = tf.cast(tf.shape(y_true)[1], "int64") * tf.ones((bs, 1), "int64")
        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, in_len, lbl_len)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred


# Decoding helper (greedy):
def ctc_decode(pred, idx_to_char):
    """pred: (batch, timesteps, num_classes). Returns list of strings."""
    input_len = tf.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len,
                                          greedy=True)[0][0]
    out = []
    for seq in results.numpy():
        out.append("".join(idx_to_char.get(int(i), "") for i in seq if i != -1))
    return out


if __name__ == "__main__":
    build_crnn().summary()
