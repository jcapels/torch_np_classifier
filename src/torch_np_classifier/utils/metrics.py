import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def average_precision_macro(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average="macro")


def f1_macro_by_group(y_true, y_pred, group_indexes):
    return f1_macro(y_true[:, group_indexes], y_pred[:, group_indexes])


def average_precision_macro_by_group(y_true, y_pred, group_indexes):
    return average_precision_macro(y_true[:, group_indexes], y_pred[:, group_indexes])
