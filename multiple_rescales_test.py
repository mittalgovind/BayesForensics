import os
import numpy as np
import pandas as pd
import tensorflow as tf
from helpers import tf_helpers as tfh
from bayesian.sfp import SFP
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm


def variation_ratio(X):
    probs = np.array(X)
    n_passes = probs.shape[0]

    means = np.array([np.sum(c) / n_passes for c in probs.T])

    return 1 - means[np.argmax(means)]


def predictive_entropy(X):
    probs = np.array(X)
    n_passes = probs.shape[0]

    means = np.array([np.sum(c) / n_passes for c in probs.T])

    return -np.sum(
        np.multiply(
            means, np.divide(np.log(np.clip(means, 1e-12, None)), np.log(n_passes))
        )
    )


def mutual_information(X):
    probs = np.array(X)
    n_passes = probs.shape[0]

    pred_ent = predictive_entropy(X)
    exp_value = np.divide(
        np.sum(
            np.multiply(
                probs, np.divide(np.log(np.clip(probs, 1e-12, None)), np.log(n_passes))
            )
        ),
        n_passes,
    )

    return pred_ent + exp_value


TARGET = "data/rgb/clic512"

np.random.seed(0)
files = np.random.choice(
    [TARGET + "/" + f for f in os.listdir(TARGET) if f[-4:] == ".png"], 250
)

imgs = np.array([np.asarray(Image.open(f)) for f in files])

scales = (0.25, 1.0)
n_classes = 31
patch_size = 128
methods = ["nearest", "bilinear", "bicubic", "lanczos3"]
classes = np.linspace(*scales, num=n_classes)
num_runs = 50

model = SFP(
    c_filters=(32, 32, 32, 32),
    d_filters=(32, 16, n_classes),
    kernel=5,
    activation="leaky_relu",
    trainable_residual=True,
    drop=0.1,
    append_rgb=False,
)
loaded = False


df = pd.DataFrame(
    columns=[
        "file",
        "true_sf",
        "method",
        "predicted_sf",
        "variation_ratio",
        "predictive_entropy",
        "mutual_information",
    ]
)
for sf in classes:
    for method in methods:
        print(f"SF: {sf}, Method: {method}")
        rescaled = np.asarray(
            tf.image.resize(
                imgs, [int(sf * patch_size), int(sf * patch_size)], method=method
            )
        )

        if not loaded:
            preds = model(
                tf.expand_dims(rescaled[0].astype(float), axis=0), training=False
            ).numpy()
            model.load_weights("model_weights/bayesian_scaleFactor_multAlgo.h5")
            loaded = True

        for i in range(len(rescaled)):
            preds = []
            for j in range(num_runs):
                preds.append(
                    softmax(
                        model(
                            tf.expand_dims(rescaled[j].astype(float), axis=0),
                            training=False,
                        ).numpy()[0]
                    )
                )

            df = df.append(
                {
                    "file": files[i],
                    "true_sf": sf,
                    "method": method,
                    "predicted_sf": classes[
                        np.argmax(
                            np.array(
                                [np.sum(c) / len(preds) for c in np.array(preds).T]
                            )
                        )
                    ],
                    "variation_ratio": variation_ratio(preds),
                    "predictive_entropy": predictive_entropy(preds),
                    "mutual_information": mutual_information(preds),
                },
                ignore_index=True,
            )

df.to_csv("test_clic_multAlgo_0.25-1.0.csv", index=False)
