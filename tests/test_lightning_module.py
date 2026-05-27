"""Tests for NPClassifierLightning."""

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_np_classifier.models.lightning_module import NPClassifierLightning

BATCH_SIZE = 4
INPUT_DIM = 6144
NUM_CATEGORIES = 10


@pytest.fixture
def batch():
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    y = torch.randint(0, 2, (BATCH_SIZE, NUM_CATEGORIES)).float()
    return x, y


@pytest.fixture
def module():
    return NPClassifierLightning(num_categories=NUM_CATEGORIES)


@pytest.fixture
def module_original():
    return NPClassifierLightning(num_categories=NUM_CATEGORIES, original=True)


class TestTrainingStep:
    def test_training_step_loss(self, module, batch):
        module.train()
        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0


class TestValidationStep:
    def test_validation_step_loss(self, module, batch):
        module.eval()
        with torch.no_grad():
            loss = module.validation_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0.0


class TestPredictStep:
    def test_predict_step_shape(self, module, batch):
        module.eval()
        with torch.no_grad():
            out = module.predict_step(batch, batch_idx=0)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (BATCH_SIZE, NUM_CATEGORIES)

    def test_predict_step_with_embedding(self, module, batch):
        module.eval()
        module.return_embedding = True
        with torch.no_grad():
            out = module.predict_step(batch, batch_idx=0)
        assert isinstance(out, dict)
        assert "predictions" in out
        assert "embeddings" in out
        assert out["predictions"].shape == (BATCH_SIZE, NUM_CATEGORIES)
        assert out["embeddings"].shape == (BATCH_SIZE, 1152)
        module.return_embedding = False

    def test_predict_step_with_embedding_original(self, module_original, batch):
        module_original.eval()
        module_original.return_embedding = True
        with torch.no_grad():
            out = module_original.predict_step(batch, batch_idx=0)
        assert isinstance(out, dict)
        assert out["predictions"].shape == (BATCH_SIZE, NUM_CATEGORIES)
        assert out["embeddings"].shape == (BATCH_SIZE, 1152)
        module_original.return_embedding = False

    def test_predict_step_bare_tensor(self, module):
        """predict_step should also accept a plain tensor (no labels)."""
        module.eval()
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        with torch.no_grad():
            out = module.predict_step(x, batch_idx=0)
        assert out.shape == (BATCH_SIZE, NUM_CATEGORIES)


class TestConfigureOptimizers:
    def test_configure_optimizers_no_scheduler(self):
        module = NPClassifierLightning(num_categories=NUM_CATEGORIES, scheduler=False)
        result = module.configure_optimizers()
        assert isinstance(result, Adam)

    def test_configure_optimizers_with_scheduler(self):
        module = NPClassifierLightning(num_categories=NUM_CATEGORIES, scheduler=True)
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(result["optimizer"], Adam)
        sched_cfg = result["lr_scheduler"]
        assert isinstance(sched_cfg["scheduler"], ReduceLROnPlateau)
        assert sched_cfg["monitor"] == "val_loss"
