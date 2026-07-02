"""
cnn_model.py
------------
Defines the Convolutional Neural Network used for Handwritten Character
Recognition. The same architecture works for MNIST (10 digit classes) and
EMNIST (47 balanced classes) — only `num_classes` changes.

The network is a compact VGG-style stack: two convolution blocks for feature
extraction followed by a dense classifier head with dropout for regularisation.
"""

from tensorflow.keras import layers, models, regularizers


def build_cnn(input_shape=(28, 28, 1), num_classes=10, dropout=0.4):
    """Build and return an uncompiled CNN model.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input image. MNIST/EMNIST are 28x28 grayscale.
    num_classes : int
        Number of output classes (10 for MNIST digits, 47 for EMNIST balanced).
    dropout : float
        Dropout rate applied in the classifier head.
    """
    model = models.Sequential(name="HCR_CNN")

    # ---- Convolution block 1 ----
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Convolution block 2 ----
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ---- Classifier head ----
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def compile_model(model, learning_rate=1e-3):
    """Compile the model with Adam + sparse categorical cross-entropy."""
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = compile_model(build_cnn(num_classes=10))
    m.summary()
