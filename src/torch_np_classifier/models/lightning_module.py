"""NPClassifierLightning — a self-contained Lightning module for NP Classifier.

No plants-sm or deepmol dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import lightning
import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_np_classifier.models.np_classifier import NPClassifierDNN


class NPClassifierLightning(lightning.LightningModule):
    """Lightning wrapper around :class:`NPClassifierDNN`.

    Parameters
    ----------
    num_categories:
        Number of output labels (multi-label classification).
    lr:
        Learning rate for the Adam optimiser.
    scheduler:
        If ``True``, attach a ``ReduceLROnPlateau`` scheduler that monitors
        ``val_loss`` with ``patience=3``.
    original:
        If ``True``, use the "original" DNN architecture (3072→2304→1152→out).
        Otherwise use the default architecture (2048→3072→1536→1536→out).

    Attributes
    ----------
    return_embedding:
        When ``True``, ``predict_step`` returns a dict with both
        ``"predictions"`` and ``"embeddings"`` keys.  Toggled by
        :class:`~torch_np_classifier.callbacks.EmbeddingCollector`.

    Examples
    --------
    >>> module = NPClassifierLightning(num_categories=730)
    >>> x = torch.randn(4, 6144)
    >>> out = module.model([x])
    >>> out.shape
    torch.Size([4, 730])
    """

    def __init__(
        self,
        num_categories: int = 730,
        lr: float = 1e-5,
        scheduler: bool = False,
        original: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = NPClassifierDNN(num_categories=num_categories, original=original)
        self.return_embedding: bool = False

    # ------------------------------------------------------------------
    # Forward / steps
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model([x])

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model([x])
        loss = BCELoss()(logits, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model([x])
        loss = BCELoss()(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        logits = self.model([x])
        loss = BCELoss()(logits, y)
        self.log("test_loss", loss)
        return {"logits": logits, "labels": y}

    def predict_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        if self.return_embedding:
            logits, embedding = self.model([x], return_embedding=True)
            return {"predictions": logits, "embeddings": embedding}

        return self.model([x])

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(
        self,
    ) -> Union[Adam, Dict[str, Any]]:
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler:
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        return optimizer
