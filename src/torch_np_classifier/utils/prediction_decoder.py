"""Decode multilabel probability arrays into named NP Classifier labels."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

PATHWAY_SLICE    = slice(None, 7)
SUPERCLASS_SLICE = slice(7, 77)
CLASS_SLICE      = slice(77, None)

_LEVEL_SLICES: Dict[str, slice] = {
    "pathway":    PATHWAY_SLICE,
    "superclass": SUPERCLASS_SLICE,
    "class":      CLASS_SLICE,
}


def decode_predictions(
    probs: Union[np.ndarray, List[float]],
    label_names: List[str],
    threshold: float = 0.5,
) -> Union[Dict[str, List[str]], List[Dict[str, List[str]]]]:
    """Convert multilabel probabilities to named pathway / superclass / class labels.

    Parameters
    ----------
    probs:
        Array of shape ``(num_labels,)`` for a single molecule or
        ``(N, num_labels)`` for a batch.
    label_names:
        Ordered list of ``num_labels`` label names matching the columns of
        ``probs`` (i.e. the label columns from the CSV, starting at index 2).
    threshold:
        Probability cutoff for a positive prediction (default ``0.5``).

    Returns
    -------
    For a single molecule: a ``dict`` with keys ``"pathway"``, ``"superclass"``,
    and ``"class"``, each containing a list of predicted label names.
    For a batch: a list of such dicts, one per molecule.
    """
    arr = np.asarray(probs, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]

    results: List[Dict[str, List[str]]] = []
    for row in arr:
        entry: Dict[str, List[str]] = {}
        for level, sl in _LEVEL_SLICES.items():
            level_probs = row[sl]
            level_names = label_names[sl]
            entry[level] = [
                name for name, p in zip(level_names, level_probs) if p >= threshold
            ]
        results.append(entry)

    return results[0] if single else results
