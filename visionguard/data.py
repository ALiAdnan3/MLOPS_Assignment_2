"""Task 2.1 - tf.data input pipeline.

split CSV -> (path, label) -> decode+resize (PIL, shared with serving)
          -> cache -> [shuffle -> augment]  (train only) -> batch -> prefetch
"""
import pandas as pd
import tensorflow as tf

from visionguard.preprocess import IMG_SIZE, load_image_path

AUTOTUNE = tf.data.AUTOTUNE


def augmentation_layers(strength: str = "light") -> tf.keras.Sequential:
    """Random augmentation applied to TRAINING batches only.

    Kept mild on purpose: heavy blur/noise/colour changes would destroy the
    subtle texture and colour artefacts that distinguish AI images.
    """
    if strength == "none":
        return tf.keras.Sequential([], name="no_augmentation")
    layers = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),        # +-18 degrees
        tf.keras.layers.RandomZoom(0.10),
    ]
    if strength == "strong":
        layers += [
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ]
    return tf.keras.Sequential(layers, name=f"augment_{strength}")


def _decode(path, label, img_size):
    img = tf.numpy_function(lambda p: load_image_path(p.decode(), img_size), [path], tf.uint8)
    img.set_shape((img_size, img_size, 3))
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)


def make_dataset(split_csv: str, training: bool, batch_size: int = 32,
                 img_size: int = IMG_SIZE, augment: str = "light", seed: int = 42) -> tf.data.Dataset:
    df = pd.read_csv(split_csv)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    ds = ds.map(lambda p, y: _decode(p, y, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()                                   # decode each file only once
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training and augment != "none":
        aug = augmentation_layers(augment)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
