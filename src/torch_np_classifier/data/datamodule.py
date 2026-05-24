"""NPClassifierDataModule — Lightning DataModule for NP Classifier."""

from __future__ import annotations

from typing import List, Optional

import lightning
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from torch_np_classifier.data.dataset import NPClassifierDataset
from torch_np_classifier.featurization.np_classifier_fp import NPClassifierFeaturizer


class NPClassifierDataModule(lightning.LightningDataModule):
    """LightningDataModule that reads CSV files and featurizes SMILES on the fly.

    Parameters
    ----------
    train_csv, val_csv, test_csv, predict_csv:
        Paths to CSV files for each split.  All are optional — only the splits
        whose paths are provided will be loaded.
    smiles_col:
        Name of the column containing SMILES strings.
    label_cols:
        Explicit list of column names to use as label columns.  Takes
        precedence over ``label_slice``.
    label_slice:
        ``slice`` applied to ``df.columns`` to select label columns when
        ``label_cols`` is not given.  E.g. ``slice(1, None)`` selects all
        columns after the first.
    batch_size:
        Batch size for all DataLoaders.
    num_workers:
        Number of worker processes for DataLoaders.
    featurizer:
        An ``NPClassifierFeaturizer`` instance.  Defaults to
        ``NPClassifierFeaturizer()`` (radius=2, 6144-dim output).

    Examples
    --------
    >>> dm = NPClassifierDataModule(
    ...     train_csv="train.csv",
    ...     val_csv="val.csv",
    ...     smiles_col="SMILES",
    ...     label_slice=slice(1, None),
    ... )
    >>> dm.setup("fit")
    >>> loader = dm.train_dataloader()
    """

    def __init__(
        self,
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        predict_csv: Optional[str] = None,
        smiles_col: str = "SMILES",
        label_cols: Optional[List[str]] = None,
        label_slice: Optional[slice] = None,
        batch_size: int = 128,
        num_workers: int = 4,
        featurizer: Optional[NPClassifierFeaturizer] = None,
    ) -> None:
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.predict_csv = predict_csv
        self.smiles_col = smiles_col
        self.label_cols = label_cols
        self.label_slice = label_slice
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.featurizer = (
            featurizer if featurizer is not None else NPClassifierFeaturizer()
        )

        self._train_dataset: Optional[NPClassifierDataset] = None
        self._val_dataset: Optional[NPClassifierDataset] = None
        self._test_dataset: Optional[NPClassifierDataset] = None
        self._predict_dataset: Optional[NPClassifierDataset] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(self, csv_path: str) -> NPClassifierDataset:
        """Read a CSV, featurize SMILES and extract labels."""
        df = pd.read_csv(csv_path)

        smiles_list = df[self.smiles_col].tolist()
        features: np.ndarray = self.featurizer.transform(smiles_list)

        labels: Optional[np.ndarray] = None
        if self.label_cols is not None:
            labels = df[self.label_cols].values.astype(np.float32)
        elif self.label_slice is not None:
            label_columns = df.columns[self.label_slice].tolist()
            labels = df[label_columns].values.astype(np.float32)

        return NPClassifierDataset(features, labels)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", None):
            if self.train_csv is not None and self._train_dataset is None:
                self._train_dataset = self._load_dataset(self.train_csv)
            if self.val_csv is not None and self._val_dataset is None:
                self._val_dataset = self._load_dataset(self.val_csv)

        if stage in ("test", None):
            if self.test_csv is not None and self._test_dataset is None:
                self._test_dataset = self._load_dataset(self.test_csv)

        if stage in ("predict", None):
            if self.predict_csv is not None and self._predict_dataset is None:
                self._predict_dataset = self._load_dataset(self.predict_csv)

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, (
            "train_dataset is None — call setup('fit') first or provide train_csv."
        )
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, (
            "val_dataset is None — call setup('fit') first or provide val_csv."
        )
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None, (
            "test_dataset is None — call setup('test') first or provide test_csv."
        )
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self._predict_dataset is not None, (
            "predict_dataset is None — call setup('predict') first or provide predict_csv."
        )
        return DataLoader(
            self._predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
