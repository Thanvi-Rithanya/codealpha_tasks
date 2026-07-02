"""
labels.py
---------
Class-index -> human-readable character mappings for each dataset.

MNIST  : 10 classes, digits 0-9.
EMNIST : the 'balanced' split has 47 classes (digits + upper + a subset of
         lower-case letters whose shapes differ from their upper-case form).
         The official mapping is reproduced below.
"""

# MNIST digit labels (index == digit)
MNIST_LABELS = {i: str(i) for i in range(10)}

# EMNIST 'balanced' split, 47 classes.
# Order follows the official emnist-balanced-mapping.txt file.
_EMNIST_CHARS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abdefghnqrt"
)
EMNIST_LABELS = {i: ch for i, ch in enumerate(_EMNIST_CHARS)}


def get_labels(dataset: str) -> dict:
    dataset = dataset.lower()
    if dataset == "mnist":
        return MNIST_LABELS
    if dataset == "emnist":
        return EMNIST_LABELS
    raise ValueError(f"Unknown dataset: {dataset}")
