"""Task 2.2 (baseline CNN) and Task 2.3 (transfer learning) model builders.

Both models take raw RGB pixels in [0, 255] of shape (IMG_SIZE, IMG_SIZE, 3)
and do their own scaling as the first layer, so the saved .keras file is
self-contained and serving only needs visionguard.preprocess.load_image.
"""
import tensorflow as tf
from tensorflow.keras import layers

from visionguard.preprocess import IMG_SIZE

BACKBONE_NAME = "backbone"


def metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def compile_model(model, lr):
    # Binary cross-entropy: the correct loss for one sigmoid output (log-likelihood of a Bernoulli label).
    # Adam: adaptive per-parameter step sizes, robust default that needs little LR tuning.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=metrics())
    return model


def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    # momentum 0.9 (default 0.99): with only ~22 batches/epoch the moving mean/variance used at
    # validation time must converge quickly, otherwise val predictions collapse to one class.
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_baseline_cnn(img_size=IMG_SIZE, dropout=0.4):
    inp = layers.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inp)
    for i, f in enumerate([32, 64, 128, 256], start=1):
        x = conv_block(x, f, f"block{i}")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    out = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return tf.keras.Model(inp, out, name="baseline_cnn")


BACKBONES = {
    # name: (constructor, scaling expected by that network's ImageNet weights, last-block prefix)
    "mobilenetv2": (tf.keras.applications.MobileNetV2, dict(scale=1 / 127.5, offset=-1.0), "block_16"),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, None, "block7a"),  # has built-in rescaling
}


def build_transfer_model(backbone="mobilenetv2", img_size=IMG_SIZE, dropout=0.3):
    ctor, scaling, _ = BACKBONES[backbone]
    base = ctor(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.name = BACKBONE_NAME  # noqa - stable name so we can find it later
    base.trainable = False      # stage 1: frozen feature extractor

    inp = layers.Input((img_size, img_size, 3), name="image")
    # MobileNetV2 was trained on pixels in [-1, 1]; this layer is exactly
    # keras.applications.mobilenet_v2.preprocess_input, but saved inside the model.
    x = layers.Rescaling(**scaling, name="preprocess")(inp) if scaling else inp
    x = base(x, training=False)  # keep BatchNorm in inference mode even when fine-tuning
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    return tf.keras.Model(inp, out, name=f"{backbone}_transfer")


def unfreeze_last(model, backbone="mobilenetv2", n_layers=None):
    """Stage 2: unfreeze the last block (default) or the last n_layers of the backbone.

    BatchNormalization layers stay frozen: with small batches their statistics
    would be destroyed by fine-tuning.
    """
    base = model.get_layer(BACKBONE_NAME)
    prefix = BACKBONES[backbone][2]
    base.trainable = True
    if n_layers is None:
        start = next(i for i, l in enumerate(base.layers) if l.name.startswith(prefix))
    else:
        start = len(base.layers) - n_layers
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return [l.name for l in base.layers if l.trainable and l.weights]
