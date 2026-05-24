import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def average_precision_macro(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average="macro")


def f1_macro_by_group(y_true, y_pred, group_indexes):
    return f1_macro(y_true[:, group_indexes], y_pred[:, group_indexes])


def average_precision_macro_by_group(y_true, y_pred, group_indexes):
    return average_precision_macro(y_true[:, group_indexes], y_pred[:, group_indexes])


def cosine_similarity_mean_sd(y_true, y_pred):
    """Per-sample cosine similarity between true and predicted label vectors.

    Returns (mean, sd) over all samples.  Samples whose true or predicted
    vector is all-zero receive a similarity of 0.0 (cosine is undefined there).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scores = np.array(
        [
            float(_cosine_similarity(t.reshape(1, -1), p.reshape(1, -1))[0, 0])
            if np.any(t) and np.any(p)
            else 0.0
            for t, p in zip(y_true, y_pred)
        ]
    )
    return float(scores.mean()), float(scores.std())


def cosine_similarity_mean_sd_by_group(y_true, y_pred, group_indexes):
    return cosine_similarity_mean_sd(y_true[:, group_indexes], y_pred[:, group_indexes])


def map_mean_sd(y_true, y_pred):
    """Per-label average precision, returned as (mean, sd) across labels.

    Labels with no positive samples are excluded from the computation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    score = average_precision_score(y_true, y_pred, average="macro")
    return score, 0.0


def map_mean_sd_by_group(y_true, y_pred, group_indexes):
    return map_mean_sd(y_true[:, group_indexes], y_pred[:, group_indexes])
