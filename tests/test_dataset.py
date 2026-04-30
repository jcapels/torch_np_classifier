"""Tests for NPClassifierDataset."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from torch_np_classifier.data.dataset import NPClassifierDataset


@pytest.fixture
def features():
    np.random.seed(0)
    return np.random.rand(20, 6144).astype(np.float32)


@pytest.fixture
def labels():
    np.random.seed(1)
    return np.random.randint(0, 2, (20, 730)).astype(np.float32)


class TestNPClassifierDataset:
    def test_len(self, features, labels):
        ds = NPClassifierDataset(features, labels)
        assert len(ds) == 20

    def test_getitem_with_labels(self, features, labels):
        ds = NPClassifierDataset(features, labels)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        feat, lab = item
        assert feat.shape == torch.Size([6144])
        assert lab.shape == torch.Size([730])

    def test_getitem_no_labels(self, features):
        ds = NPClassifierDataset(features)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == torch.Size([6144])

    def test_tensor_dtype(self, features, labels):
        ds = NPClassifierDataset(features, labels)
        feat, lab = ds[0]
        assert feat.dtype == torch.float32
        assert lab.dtype == torch.float32

    def test_dataloader_batch_shape(self, features, labels):
        ds = NPClassifierDataset(features, labels)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch_feat, batch_lab = next(iter(loader))
        assert batch_feat.shape == torch.Size([8, 6144])
        assert batch_lab.shape == torch.Size([8, 730])

    def test_len_no_labels(self, features):
        ds = NPClassifierDataset(features)
        assert len(ds) == 20
