"""NPClassifierDataset — a minimal torch Dataset for NP Classifier data."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class NPClassifierDataset(Dataset):
    """PyTorch Dataset that holds pre-computed fingerprints and optional labels.

    Parameters
    ----------
    features:
        Float array of shape ``(N, feature_dim)``.
    labels:
        Optional float array of shape ``(N, num_categories)``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.rand(10, 6144).astype(np.float32)
    >>> y = np.random.randint(0, 2, (10, 730)).astype(np.float32)
    >>> ds = NPClassifierDataset(X, y)
    >>> len(ds)
    10
    >>> feat, lab = ds[0]
    >>> feat.shape
    torch.Size([6144])
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        if labels is not None:
            self.labels: Optional[torch.Tensor] = torch.as_tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]
