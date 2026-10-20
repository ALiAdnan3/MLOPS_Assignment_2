"""Image preprocessing shared by training (tf.data) and serving (Flask/API).

Keeping ONE implementation guarantees the model sees identically prepared
pixels at training time and at prediction time.

Why PIL instead of tf.io.decode_image: the dataset contains WEBP, GIF, CMYK
and palette images. TensorFlow cannot decode WEBP and handles CMYK/palette
poorly; PIL handles all of them and converts everything to 3-channel RGB.

Model-specific scaling (e.g. /255 for the CNN, [-1, 1] for MobileNetV2) is
NOT done here - it is a layer inside each model, so it is saved with the
model and can never be forgotten at serving time.
"""
import io

import numpy as np
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = {0: "Real", 1: "AI-generated"}


def load_image(data: bytes, img_size: int = IMG_SIZE) -> np.ndarray:
    """bytes -> uint8 array (img_size, img_size, 3).

    Raises PIL.UnidentifiedImageError / OSError if the bytes are not a decodable image.
    """
    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)                      # first frame of animated GIF/WEBP
        if im.mode == "P":
            im = im.convert("RGBA")     # palette with transparency -> RGBA first
        im = im.convert("RGB")          # RGBA / P / L / CMYK -> RGB
        # Plain resize ("squash") to a square. Aspect ratio is NOT preserved;
        # see report 2.1 for why we accept this instead of padding/cropping.
        im = im.resize((img_size, img_size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_image_path(path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    with open(path, "rb") as f:
        return load_image(f.read(), img_size)
