"""Task 2.7 - Keras Tuner.

  python -m src.tuning --part toy     # tiny Dense net on Fashion-MNIST: learning rate + units, RandomSearch
  python -m src.tuning --part apply   # VisionGuard MobileNetV2: fine-tune LR, dropout, # unfrozen layers (10 trials)

Results -> reports/tuner_<part>.csv
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras_tuner as kt  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

SEED = 42


def results_table(tuner, keys, metric):
    rows = []
    for t in tuner.oracle.trials.values():
        if t.score is None:
            continue
        rows.append({"trial": t.trial_id, **{k: t.hyperparameters.get(k) for k in keys}, metric: round(t.score, 4)})
    df = pd.DataFrame(rows).sort_values(metric, ascending=False)
    return df


# ------------------------------------------------------------------ toy
def toy():
    (x, y), (xv, yv) = tf.keras.datasets.fashion_mnist.load_data()
    x, y, xv, yv = x[:15000] / 255.0, y[:15000], xv[:3000] / 255.0, yv[:3000]

    def build(hp):
        m = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28)), tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")])
        m.compile(optimizer=tf.keras.optimizers.Adam(hp.Float("learning_rate", 1e-4, 1e-1, sampling="log")),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return m

    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=10, seed=SEED,
                            directory="tuner_runs", project_name="toy_fashion_mnist", overwrite=True)
    tuner.search_space_summary()
    tuner.search(x, y, validation_data=(xv, yv), epochs=4, batch_size=128, verbose=0)
    df = results_table(tuner, ["learning_rate", "units"], "val_accuracy")
    best = tuner.get_best_hyperparameters(1)[0]
    print(df.to_string(index=False))
    print("BEST:", best.values)
    df.to_csv("reports/tuner_toy.csv", index=False)


# ------------------------------------------------------------------ apply
class VisionGuardHyperModel(kt.HyperModel):
    """Same two-stage recipe as src/train.py: train head on frozen backbone, then fine-tune the last N layers."""

    def build(self, hp):
        from visionguard.models import build_transfer_model, compile_model
        model = build_transfer_model("mobilenetv2", dropout=hp.Float("dropout", 0.1, 0.5, step=0.1))
        return compile_model(model, 1e-3)

    def fit(self, hp, model, train_ds, val_ds, callbacks=None, **_):
        from visionguard.models import compile_model, unfreeze_last
        model.fit(train_ds, validation_data=val_ds, epochs=4, verbose=0)               # stage 1: head only
        n = hp.Choice("unfreeze_layers", [0, 10, 20, 40, 60])
        if n > 0:
            unfreeze_last(model, "mobilenetv2", n_layers=n)
        compile_model(model, hp.Float("fine_tune_lr", 1e-5, 1e-3, sampling="log"))
        return model.fit(train_ds, validation_data=val_ds, epochs=6, verbose=0,       # stage 2: fine-tune
                         callbacks=callbacks)


def apply():
    from visionguard.data import make_dataset
    tf.keras.utils.set_random_seed(SEED)
    train_ds = make_dataset("data/splits/train.csv", True, 32, augment="light")
    val_ds = make_dataset("data/splits/val.csv", False, 32)
    tuner = kt.RandomSearch(VisionGuardHyperModel(), objective=kt.Objective("val_auc", "max"), max_trials=10,
                            seed=SEED, directory="tuner_runs", project_name="visionguard_mnv2", overwrite=True)
    tuner.search_space_summary()
    tuner.search(train_ds, val_ds)
    df = results_table(tuner, ["fine_tune_lr", "dropout", "unfreeze_layers"], "val_auc")
    print(df.to_string(index=False))
    print("BEST:", tuner.get_best_hyperparameters(1)[0].values)
    df.to_csv("reports/tuner_visionguard.csv", index=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["toy", "apply"], required=True)
    toy() if p.parse_args().part == "toy" else apply()
