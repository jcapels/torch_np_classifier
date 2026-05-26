"""EmbeddingCollector — Lightning Callback that gathers embeddings during predict."""

from __future__ import annotations

from typing import Any, Dict, List

import lightning
import numpy as np
import torch


class EmbeddingCollector(lightning.Callback):
    """Collect molecular embeddings produced during a predict run.

    The callback sets ``pl_module.return_embedding = True`` before prediction
    starts, so that :meth:`NPClassifierLightning.predict_step` returns a dict
    with both ``"predictions"`` and ``"embeddings"`` keys.

    After prediction, access results through the :attr:`embeddings` and
    :attr:`predictions` properties.

    Examples
    --------
    >>> collector = EmbeddingCollector()
    >>> trainer = lightning.Trainer(callbacks=[collector])
    >>> trainer.predict(module, datamodule=dm)
    >>> emb = collector.embeddings   # shape (N, embedding_dim)
    >>> pred = collector.predictions  # shape (N, num_categories)
    """

    def __init__(self) -> None:
        super().__init__()
        self._embeddings: List[np.ndarray] = []
        self._predictions: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_predict_start(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        pl_module.return_embedding = True
        self._embeddings = []
        self._predictions = []

    def on_predict_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        pl_module.return_embedding = False

    def on_predict_batch_end(
        self,
        trainer: lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._embeddings.append(outputs["embeddings"].cpu().numpy())
        self._predictions.append(outputs["predictions"].cpu().numpy())

    # ------------------------------------------------------------------
    # Result properties
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> np.ndarray:
        """Concatenated embeddings from all predict batches.

        Raises
        ------
        RuntimeError
            If no embeddings have been collected yet (i.e. predict has not
            been called, or the first batch has not finished).
        """
        if not self._embeddings:
            raise RuntimeError("No embeddings collected. Run trainer.predict() first.")
        return np.concatenate(self._embeddings, axis=0)

    @property
    def predictions(self) -> np.ndarray:
        """Concatenated predictions from all predict batches.

        Raises
        ------
        RuntimeError
            If no predictions have been collected yet.
        """
        if not self._predictions:
            raise RuntimeError("No predictions collected. Run trainer.predict() first.")
        return np.concatenate(self._predictions, axis=0)
