"""Task 1.1 - print exact library versions and whether TensorFlow sees a GPU."""
import platform
import sys
from importlib.metadata import version

import tensorflow as tf

print("Python      :", sys.version.split()[0], "|", platform.platform())
for pkg in ["tensorflow", "keras", "numpy", "pandas", "pillow", "scikit-learn", "matplotlib",
            "mlflow", "dagshub", "dvc", "keras-tuner", "flask"]:
    print(f"{pkg:12s}:", version(pkg))
gpus = tf.config.list_physical_devices("GPU")
print("GPUs visible:", gpus if gpus else "none -> training runs on CPU")
print("Built with CUDA:", tf.test.is_built_with_cuda())
