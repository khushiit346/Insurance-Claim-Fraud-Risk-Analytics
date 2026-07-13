"""
modeling.py
-----------
Preprocessing + two independent anomaly detectors.

Why two models instead of one:
There is no ground-truth fraud label anywhere in this dataset (the closest
proxy, CLAIM_STATUS, turned out to have zero correlation with the original
autoencoder's flags -- see README for the full discussion). With no labels
to compute precision/recall against, a single model's flags are unverifiable.
Two structurally different unsupervised models (a neural autoencoder and a
classical Isolation Forest) that independently agree on the same claims is
a much stronger signal than either one alone, and is a standard technique
for validating unsupervised fraud models in practice.

Reproducibility fix: every source of randomness (numpy, TensorFlow, sklearn's
train/validation split, Isolation Forest) is seeded.
"""

import numpy as np
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import layers, models

from .data_pipeline import CAT_FEATURES, NUM_FEATURES

RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED):
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUM_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CAT_FEATURES,
            ),
        ]
    )


def build_autoencoder(input_dim: int) -> models.Model:
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation="relu")(input_layer)
    encoded = layers.Dense(16, activation="relu")(encoded)
    bottleneck = layers.Dense(8, activation="relu")(encoded)
    decoded = layers.Dense(16, activation="relu")(bottleneck)
    decoded = layers.Dense(32, activation="relu")(decoded)
    output_layer = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(X: np.ndarray, train_frac: float = 0.8, epochs: int = 40):
    """
    Fit the autoencoder on a random subset of claims rather than the full
    dataset it will later be scored against. This is a lightweight guard
    against the model simply memorizing every record (anomalies included)
    -- reconstruction error is a more honest anomaly signal when the model
    has not directly trained on every single row it is scoring.
    """
    set_global_seed()
    n = X.shape[0]
    rng = np.random.default_rng(RANDOM_SEED)
    train_idx = rng.choice(n, size=int(n * train_frac), replace=False)
    X_train = X[train_idx]

    model = build_autoencoder(X.shape[1])
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )

    reconstructions = model.predict(X, verbose=0)
    reconstruction_error = np.mean(np.power(X - reconstructions, 2), axis=1)
    return model, reconstruction_error


def train_isolation_forest(X: np.ndarray, contamination: float = 0.05):
    """
    Classical ensemble anomaly detector, used as an independent cross-check
    against the autoencoder. `contamination` should match the percentile
    threshold used for the autoencoder so the two flag comparable claim
    counts.
    """
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=RANDOM_SEED,
    )
    iso.fit(X)
    # Higher score = more anomalous (sklearn's raw score is inverted)
    anomaly_score = -iso.score_samples(X)
    return iso, anomaly_score
