"""Tests for EmbeddingCollector callback."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from torch_np_classifier.callbacks.embedding_collector import EmbeddingCollector
from torch_np_classifier.models.lightning_module import NPClassifierLightning

NUM_CATEGORIES = 10
EMBEDDING_DIM = 1536
BATCH_SIZE = 4


def _make_fake_outputs(batch_size: int = BATCH_SIZE) -> dict:
    return {
        "embeddings": torch.randn(batch_size, EMBEDDING_DIM),
        "predictions": torch.rand(batch_size, NUM_CATEGORIES),
    }


class TestEmbeddingCollectorEmpty:
    def test_embeddings_empty_raises(self):
        collector = EmbeddingCollector()
        with pytest.raises(RuntimeError):
            _ = collector.embeddings

    def test_predictions_empty_raises(self):
        collector = EmbeddingCollector()
        with pytest.raises(RuntimeError):
            _ = collector.predictions


class TestEmbeddingCollectorCollect:
    def test_collect_embeddings(self):
        collector = EmbeddingCollector()
        pl_module = NPClassifierLightning(num_categories=NUM_CATEGORIES)
        trainer = MagicMock()

        # Simulate predict lifecycle
        collector.on_predict_start(trainer, pl_module)

        outputs1 = _make_fake_outputs(batch_size=BATCH_SIZE)
        outputs2 = _make_fake_outputs(batch_size=BATCH_SIZE)
        collector.on_predict_batch_end(trainer, pl_module, outputs1, batch=None, batch_idx=0)
        collector.on_predict_batch_end(trainer, pl_module, outputs2, batch=None, batch_idx=1)

        collector.on_predict_end(trainer, pl_module)

        emb = collector.embeddings
        pred = collector.predictions

        assert emb.shape == (BATCH_SIZE * 2, EMBEDDING_DIM)
        assert pred.shape == (BATCH_SIZE * 2, NUM_CATEGORIES)
        assert isinstance(emb, np.ndarray)
        assert isinstance(pred, np.ndarray)

    def test_return_embedding_toggled(self):
        collector = EmbeddingCollector()
        pl_module = NPClassifierLightning(num_categories=NUM_CATEGORIES)
        trainer = MagicMock()

        assert pl_module.return_embedding is False

        collector.on_predict_start(trainer, pl_module)
        assert pl_module.return_embedding is True

        collector.on_predict_end(trainer, pl_module)
        assert pl_module.return_embedding is False

    def test_internal_lists_cleared_on_predict_start(self):
        collector = EmbeddingCollector()
        pl_module = NPClassifierLightning(num_categories=NUM_CATEGORIES)
        trainer = MagicMock()

        # First run
        collector.on_predict_start(trainer, pl_module)
        collector.on_predict_batch_end(
            trainer, pl_module, _make_fake_outputs(), batch=None, batch_idx=0
        )
        collector.on_predict_end(trainer, pl_module)
        assert collector.embeddings.shape[0] == BATCH_SIZE

        # Second run — internal state must be reset
        collector.on_predict_start(trainer, pl_module)
        collector.on_predict_batch_end(
            trainer, pl_module, _make_fake_outputs(batch_size=2), batch=None, batch_idx=0
        )
        collector.on_predict_end(trainer, pl_module)
        # Should only contain results from the second run
        assert collector.embeddings.shape[0] == 2
