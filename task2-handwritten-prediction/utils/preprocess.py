"""
preprocess.py
-------------
Image preprocessing utilities. These are used both at training time and at
inference time so the canvas/uploaded image is normalised exactly the way the
training data was. Keeping preprocessing in one place is the single most
important thing for getting good real-world accuracy.
"""

import numpy as np
from PIL import Image, ImageOps


def _center_by_mass(img28: np.ndarray) -> np.ndarray:
    """Shift the digit so its center of mass sits in the middle of the frame.
    MNIST images are centered this way, so matching it improves accuracy."""
    from scipy import ndimage
    cy, cx = ndimage.center_of_mass(img28)
    if np.isnan(cy) or np.isnan(cx):
        return img28
    rows, cols = img28.shape
    shifty = np.round(rows / 2.0 - cy).astype(int)
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    return ndimage.shift(img28, shift=[shifty, shiftx], mode="constant")


def preprocess_image(pil_img: Image.Image, invert=True, center=True) -> np.ndarray:
    """Convert an arbitrary PIL image into a (1, 28, 28, 1) float32 array
    scaled to [0, 1], matching the MNIST/EMNIST convention of a white digit
    on a black background.

    Parameters
    ----------
    invert : bool
        Set True when the source has dark ink on a light background
        (scanned paper, most uploads). The drawable canvas already produces a
        white stroke on a dark canvas, so it passes invert=False.
    center : bool
        Re-center the glyph by its center of mass (recommended).
    """
    img = pil_img.convert("L")  # grayscale

    if invert:
        img = ImageOps.invert(img)

    # Crop to the bounding box of the ink so scale is consistent
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Resize the longest side to 20px and pad to 28x28 (MNIST recipe)
    w, h = img.size
    if w == 0 or h == 0:
        arr = np.zeros((28, 28), dtype=np.float32)
        return arr.reshape(1, 28, 28, 1)
    scale = 20.0 / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    img = img.resize(new_size, Image.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    upper_left = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    canvas.paste(img, upper_left)

    arr = np.asarray(canvas, dtype=np.float32) / 255.0

    if center:
        try:
            arr = _center_by_mass(arr)
        except Exception:
            pass  # scipy missing or degenerate image — skip centering

    return arr.reshape(1, 28, 28, 1)


def emnist_orient(arr: np.ndarray) -> np.ndarray:
    """EMNIST images are stored transposed & flipped relative to how a human
    writes them. Apply this when feeding raw EMNIST tensors. Not needed for
    canvas input."""
    return np.transpose(arr, (0, 2, 1, 3))[:, :, ::-1, :]
