"""NPClassifierPipeline — unified fit / predict / explain API.

High-level wrapper that trains three hierarchical NP-Classifier models
(pathway, superclass, class), wraps them in NPClassifierEnsemble for
ontology-based voting, and exposes a scikit-learn-style interface for
prediction, embedding extraction, and SHAP-based explainability.

Typical workflow
----------------
>>> from torch_np_classifier import NPClassifierPipeline
>>>
>>> # Train from CSVs
>>> pipeline = NPClassifierPipeline(max_epochs=50)
>>> pipeline.fit("train.csv", val_smiles_or_csv="val.csv")
>>>
>>> # Or load from saved checkpoints
>>> pipeline = NPClassifierPipeline.from_checkpoints(
...     pathway_ckpt="pathway.ckpt",
...     superclass_ckpt="superclass.ckpt",
...     class_ckpt="class.ckpt",
... )
>>>
>>> # Full voted prediction
>>> result = pipeline.predict("Cn1cnc2c1c(=O)n(c(=O)n2C)C")
>>> result["pathway"]
['Alkaloids']
>>>
>>> # Level-specific shortcuts
>>> pipeline.predict_class("Cn1cnc2c1c(=O)n(c(=O)n2C)C")
['Purine alkaloids']
>>>
>>> # Embeddings from the class model
>>> emb = pipeline.predict_embeddings(["smi1", "smi2"], level="class")
>>> emb.shape
(2, 1536)
>>>
>>> # SHAP explanation (background = ~100 training SMILES)
>>> fig  = pipeline.explain("Cn1cnc2c1c(=O)n(c(=O)n2C)C", background_smiles)
>>> fig2 = pipeline.explain_bits("Cn1cnc2c1c(=O)n(c(=O)n2C)C",
...                              level="class", background=background_smiles)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import lightning
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    import matplotlib.figure
    from torch_np_classifier.explainability.ensemble_shap import (
        NPClassifierEnsembleSHAP,
    )

from torch_np_classifier.callbacks.embedding_collector import EmbeddingCollector
from torch_np_classifier.data.dataset import NPClassifierDataset
from torch_np_classifier.featurization.np_classifier_fp import NPClassifierFeaturizer
from torch_np_classifier.models.ensemble import NPClassifierEnsemble
from torch_np_classifier.models.lightning_module import NPClassifierLightning

_DATA_DIR = Path(__file__).parent.parent / "data"
_LABEL_PKL = _DATA_DIR / "label_names.pkl"

_N_PATHWAY = 7
_N_SUPERCLASS = 70

_LEVEL_SLICES: Dict[str, slice] = {
    "pathway": slice(None, _N_PATHWAY),
    "superclass": slice(_N_PATHWAY, _N_PATHWAY + _N_SUPERCLASS),
    "class": slice(_N_PATHWAY + _N_SUPERCLASS, None),
}


class NPClassifierPipeline:
    """Unified train / predict / explain interface for the NP Classifier.

    Trains three hierarchical models (pathway / superclass / class) and wraps
    them in :class:`~torch_np_classifier.models.ensemble.NPClassifierEnsemble`
    for ontology-based voting predictions.

    Parameters
    ----------
    lr:
        Learning rate for the Adam optimiser.
    scheduler:
        Attach a ``ReduceLROnPlateau`` scheduler that monitors ``val_loss``
        (ignored when no validation set is provided at fit time).
    original:
        Use the original DNN architecture (3072→2304→1152→out) instead of the
        default (2048→3072→1536→1536→out).
    max_epochs:
        Maximum training epochs per level.
    batch_size:
        Mini-batch size for all DataLoaders.
    num_workers:
        DataLoader worker processes.
    pathway_threshold:
        Probability threshold for pathway predictions (default 0.5).
    superclass_threshold:
        Probability threshold for superclass predictions (default 0.3).
    class_threshold:
        Probability threshold for class predictions (default 0.1).
    featurizer:
        ``NPClassifierFeaturizer`` instance.  Defaults to
        ``NPClassifierFeaturizer()`` (radius=2, 6144-dim Morgan fingerprints).
    """

    def __init__(
        self,
        lr: float = 1e-5,
        scheduler: bool = True,
        original: bool = False,
        max_epochs: int = 150,
        batch_size: int = 128,
        num_workers: int = 4,
        pathway_threshold: float = 0.5,
        superclass_threshold: float = 0.3,
        class_threshold: float = 0.1,
        featurizer: Optional[NPClassifierFeaturizer] = None,
    ) -> None:
        self.lr = lr
        self.scheduler = scheduler
        self.original = original
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pathway_threshold = pathway_threshold
        self.superclass_threshold = superclass_threshold
        self.class_threshold = class_threshold
        self.featurizer = (
            featurizer if featurizer is not None else NPClassifierFeaturizer()
        )

        self._ensemble: Optional[NPClassifierEnsemble] = None
        self._label_names: Optional[List[str]] = None  # full 730-name list

    # ── construction helpers ─────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(
        cls,
        pathway_ckpt: Union[str, Path],
        superclass_ckpt: Union[str, Path],
        class_ckpt: Union[str, Path],
        label_names_pkl: Optional[Union[str, Path]] = None,
        label_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "NPClassifierPipeline":
        """Build a pipeline from three saved ``.ckpt`` files.

        Parameters
        ----------
        pathway_ckpt, superclass_ckpt, class_ckpt:
            Paths to Lightning checkpoint files produced by :meth:`fit` or the
            hierarchical training scripts.
        label_names_pkl:
            Path to a label-names pickle (730 names in canonical order).
            Falls back to the bundled copy when ``None`` and *label_names* is
            not provided.
        label_names:
            Explicit flat list of 730 label names
            (7 pathway + 70 superclass + 653 class, in that order).
        **kwargs:
            Forwarded to :class:`NPClassifierPipeline` (e.g. ``lr``,
            ``max_epochs``, ``pathway_threshold``).
        """
        pipeline = cls(**kwargs)

        if label_names is not None:
            names = list(label_names)
            pipeline._ensemble = NPClassifierEnsemble.from_checkpoints(
                pathway_ckpt=pathway_ckpt,
                superclass_ckpt=superclass_ckpt,
                class_ckpt=class_ckpt,
                pathway_labels=names[:_N_PATHWAY],
                superclass_labels=names[_N_PATHWAY : _N_PATHWAY + _N_SUPERCLASS],
                class_labels=names[_N_PATHWAY + _N_SUPERCLASS :],
                pathway_threshold=pipeline.pathway_threshold,
                superclass_threshold=pipeline.superclass_threshold,
                class_threshold=pipeline.class_threshold,
                featurizer=pipeline.featurizer,
            )
            pipeline._label_names = names
        else:
            pipeline._ensemble = NPClassifierEnsemble.from_checkpoints(
                pathway_ckpt=pathway_ckpt,
                superclass_ckpt=superclass_ckpt,
                class_ckpt=class_ckpt,
                label_names_pkl=label_names_pkl,
                pathway_threshold=pipeline.pathway_threshold,
                superclass_threshold=pipeline.superclass_threshold,
                class_threshold=pipeline.class_threshold,
                featurizer=pipeline.featurizer,
            )
            pkl_path = label_names_pkl or _LABEL_PKL
            with open(pkl_path, "rb") as f:
                pipeline._label_names = pickle.load(f)

        return pipeline

    # ── training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        smiles_or_csv: Union[str, Path, List[str]],
        labels: Optional[np.ndarray] = None,
        val_smiles_or_csv: Optional[Union[str, Path, List[str]]] = None,
        val_labels: Optional[np.ndarray] = None,
        smiles_col: str = "SMILES",
        label_start: int = 2,
        label_names: Optional[List[str]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        patience: int = 5,
        early_stopping: bool = True,
        devices: int = 1,
    ) -> "NPClassifierPipeline":
        """Train all three hierarchical models and build the ensemble.

        Parameters
        ----------
        smiles_or_csv:
            Either a path to a CSV file or a list of SMILES strings.
        labels:
            Float32 array of shape ``(N, 730)`` — required when
            *smiles_or_csv* is a SMILES list, ignored for CSV input.
        val_smiles_or_csv:
            Validation set as a CSV path or SMILES list.  Enables early
            stopping and ``ReduceLROnPlateau`` scheduling when provided.
        val_labels:
            Validation labels array — required when *val_smiles_or_csv* is a
            SMILES list.
        smiles_col:
            Name of the SMILES column (CSV input only).
        label_start:
            Number of leading CSV columns to skip before the label columns
            (default ``2`` for the ``key`` and ``SMILES`` columns).
        label_names:
            Explicit list of 730 label names in canonical order.  Extracted
            from the CSV header when ``None``; falls back to the bundled
            ``label_names.pkl`` for array input.
        checkpoint_dir:
            Directory to save per-level best checkpoints.  A sub-directory
            per level (``pathway/``, ``superclass/``, ``class/``) is created
            inside.  When ``None``, no checkpoints are written.
        patience:
            Early-stopping patience in epochs.
        early_stopping:
            Whether to apply early stopping.  Has no effect when no validation
            set is provided.
        devices:
            Number of devices (GPUs) passed to ``lightning.Trainer``.

        Returns
        -------
        self
        """
        train_features, train_labels, names = self._load_data(
            smiles_or_csv, labels, smiles_col, label_start, label_names
        )
        self._label_names = names

        val_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if val_smiles_or_csv is not None:
            val_feat, val_lbl, _ = self._load_data(
                val_smiles_or_csv,
                val_labels,
                smiles_col,
                label_start,
                label_names=names,
            )
            val_tuple = (val_feat, val_lbl)

        trained_models: Dict[str, NPClassifierLightning] = {}

        for level_name, level_sl in _LEVEL_SLICES.items():
            tr_lbl = train_labels[:, level_sl]
            num_cats = tr_lbl.shape[1]

            train_loader = DataLoader(
                NPClassifierDataset(train_features, tr_lbl),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            val_loader = None
            if val_tuple is not None:
                vl_lbl = val_tuple[1][:, level_sl]
                val_loader = DataLoader(
                    NPClassifierDataset(val_tuple[0], vl_lbl),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )

            use_scheduler = self.scheduler and val_loader is not None
            model = NPClassifierLightning(
                num_categories=num_cats,
                lr=self.lr,
                scheduler=use_scheduler,
                original=self.original,
            )

            callbacks = []
            if val_loader is not None and early_stopping:
                callbacks.append(
                    EarlyStopping(monitor="val_loss", patience=patience, mode="min")
                )

            ckpt_cb: Optional[ModelCheckpoint] = None
            if checkpoint_dir is not None:
                level_ckpt_dir = os.path.join(str(checkpoint_dir), level_name)
                filename = (
                    f"{level_name}-{{epoch:02d}}-{{val_loss:.4f}}"
                    if val_loader is not None
                    else f"{level_name}-{{epoch:02d}}"
                )
                ckpt_cb = ModelCheckpoint(
                    dirpath=level_ckpt_dir,
                    filename=filename,
                    monitor="val_loss" if val_loader is not None else None,
                    save_top_k=1,
                    mode="min",
                )
                callbacks.append(ckpt_cb)

            trainer = lightning.Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks or None,
                log_every_n_steps=10,
                devices=devices,
                enable_progress_bar=True,
            )

            if val_loader is not None:
                trainer.fit(model, train_loader, val_loader)
            else:
                trainer.fit(model, train_loader)

            # Reload best checkpoint when available
            if ckpt_cb is not None and ckpt_cb.best_model_path:
                model = NPClassifierLightning.load_from_checkpoint(
                    ckpt_cb.best_model_path
                )

            model.eval()
            trained_models[level_name] = model

        self._ensemble = NPClassifierEnsemble(
            pathway_model=trained_models["pathway"],
            superclass_model=trained_models["superclass"],
            class_model=trained_models["class"],
            pathway_labels=self._label_names[:_N_PATHWAY],
            superclass_labels=self._label_names[
                _N_PATHWAY : _N_PATHWAY + _N_SUPERCLASS
            ],
            class_labels=self._label_names[_N_PATHWAY + _N_SUPERCLASS :],
            pathway_threshold=self.pathway_threshold,
            superclass_threshold=self.superclass_threshold,
            class_threshold=self.class_threshold,
            featurizer=self.featurizer,
        )
        return self

    # ── internal data loader ──────────────────────────────────────────────────

    def _load_data(
        self,
        smiles_or_csv: Union[str, Path, List[str]],
        labels: Optional[np.ndarray],
        smiles_col: str,
        label_start: int,
        label_names: Optional[List[str]],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return ``(features, labels_730, label_names)``."""
        if isinstance(smiles_or_csv, (str, Path)):
            df = pd.read_csv(smiles_or_csv)
            smiles_list = df[smiles_col].tolist()
            cols = df.columns[label_start:].tolist()
            lbl = df[cols].values.astype(np.float32)
            names = label_names if label_names is not None else cols
        else:
            smiles_list = list(smiles_or_csv)
            if labels is None:
                raise ValueError(
                    "labels must be provided when smiles_or_csv is a list of SMILES."
                )
            lbl = np.asarray(labels, dtype=np.float32)
            if label_names is not None:
                names = list(label_names)
            else:
                with open(_LABEL_PKL, "rb") as f:
                    names = pickle.load(f)

        features = self.featurizer.transform(smiles_list)
        return features, lbl, names

    # ── prediction ───────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._ensemble is None:
            raise RuntimeError(
                "Pipeline is not fitted yet. "
                "Call fit() or use NPClassifierPipeline.from_checkpoints()."
            )

    def predict(
        self,
        smiles: Union[str, Sequence[str]],
        check_glycoside: bool = True,
    ) -> Union[Dict, List[Dict]]:
        """Run ensemble voting prediction.

        Parameters
        ----------
        smiles:
            Single SMILES string or a list of SMILES strings.
        check_glycoside:
            Whether to run the glycoside SMARTS check.

        Returns
        -------
        For a single SMILES: dict with keys ``"pathway"``, ``"superclass"``,
        ``"class"``, ``"isglycoside"``.
        For a list: a list of such dicts.
        """
        self._check_fitted()
        return self._ensemble.predict(smiles, check_glycoside=check_glycoside)

    def predict_pathway(
        self,
        smiles: Union[str, Sequence[str]],
        check_glycoside: bool = False,
    ) -> Union[List[str], List[List[str]]]:
        """Return pathway-level voted predictions.

        Parameters
        ----------
        smiles:
            Single SMILES or list of SMILES.
        check_glycoside:
            Passed to the ensemble prediction.

        Returns
        -------
        For a single SMILES: list of pathway label strings.
        For a list: list of lists.
        """
        self._check_fitted()
        single = isinstance(smiles, str)
        results = self._ensemble.predict(smiles, check_glycoside=check_glycoside)
        if single:
            return results["pathway"]
        return [r["pathway"] for r in results]

    def predict_superclass(
        self,
        smiles: Union[str, Sequence[str]],
        check_glycoside: bool = False,
    ) -> Union[List[str], List[List[str]]]:
        """Return superclass-level voted predictions.

        Parameters
        ----------
        smiles:
            Single SMILES or list of SMILES.
        check_glycoside:
            Passed to the ensemble prediction.

        Returns
        -------
        For a single SMILES: list of superclass label strings.
        For a list: list of lists.
        """
        self._check_fitted()
        single = isinstance(smiles, str)
        results = self._ensemble.predict(smiles, check_glycoside=check_glycoside)
        if single:
            return results["superclass"]
        return [r["superclass"] for r in results]

    def predict_class(
        self,
        smiles: Union[str, Sequence[str]],
        check_glycoside: bool = False,
    ) -> Union[List[str], List[List[str]]]:
        """Return class-level voted predictions.

        Parameters
        ----------
        smiles:
            Single SMILES or list of SMILES.
        check_glycoside:
            Passed to the ensemble prediction.

        Returns
        -------
        For a single SMILES: list of class label strings.
        For a list: list of lists.
        """
        self._check_fitted()
        single = isinstance(smiles, str)
        results = self._ensemble.predict(smiles, check_glycoside=check_glycoside)
        if single:
            return results["class"]
        return [r["class"] for r in results]

    def predict_embeddings(
        self,
        smiles: Union[str, Sequence[str]],
        level: str = "class",
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract penultimate-layer embeddings.

        Uses :class:`~torch_np_classifier.callbacks.EmbeddingCollector` to
        capture the activations just before the output layer of the chosen
        hierarchical model.

        Parameters
        ----------
        smiles:
            Single SMILES string or a list of SMILES strings.
        level:
            Which model to extract embeddings from: ``"pathway"``,
            ``"superclass"``, or ``"class"``.
        batch_size:
            Batch size for the predict DataLoader.

        Returns
        -------
        Float32 array of shape ``(N, embedding_dim)`` — ``embedding_dim``
        is 1536 for the default architecture (``original=False``).
        """
        self._check_fitted()
        if level not in _LEVEL_SLICES:
            raise ValueError(
                f"level must be one of {list(_LEVEL_SLICES)}, got {level!r}"
            )

        smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
        model = getattr(self._ensemble, f"{level}_model")
        features = self.featurizer.transform(smiles_list)
        dataset = NPClassifierDataset(features)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        collector = EmbeddingCollector()
        trainer = lightning.Trainer(
            enable_progress_bar=False,
            logger=False,
            callbacks=[collector],
        )
        trainer.predict(model, loader)
        return collector.embeddings

    # ── explainability ────────────────────────────────────────────────────────

    def _resolve_background(
        self,
        background: Union[Sequence[str], np.ndarray],
    ) -> np.ndarray:
        """Featurize a SMILES list or pass through a pre-computed array."""
        if isinstance(background, np.ndarray):
            return background.astype(np.float32)
        return self.featurizer.transform(list(background))

    def _make_shap(
        self,
        background: Union[Sequence[str], np.ndarray],
    ) -> NPClassifierEnsembleSHAP:
        from torch_np_classifier.explainability.ensemble_shap import (
            NPClassifierEnsembleSHAP,
        )

        self._check_fitted()
        bg_features = self._resolve_background(background)
        return NPClassifierEnsembleSHAP(self._ensemble, bg_features)

    def explain(
        self,
        smiles: str,
        background: Union[Sequence[str], np.ndarray],
        k: int = 6,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> matplotlib.figure.Figure:
        """Full three-level SHAP explanation figure.

        Runs ensemble voting, computes per-level SHAP values with
        ``shap.GradientExplainer``, and produces a combined figure with one
        panel per predicted level — molecule with highlighted bits, a SHAP bar
        chart, and a Morgan-environment fragment grid.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        background:
            Reference distribution for SHAP — a list of SMILES strings (will
            be featurized internally) or a pre-computed ``(N, 6144)`` float32
            array.  A random sample of 50–200 training molecules is recommended.
        k:
            Top-*k* bits to highlight per level.
        figsize:
            Override figure size ``(width, height)`` in inches.

        Returns
        -------
        matplotlib Figure.
        """
        return self._make_shap(background).explanation_figure(
            smiles, self.featurizer, k=k, figsize=figsize
        )

    def explain_bits(
        self,
        smiles: str,
        level: str,
        background: Union[Sequence[str], np.ndarray],
        k: int = 6,
    ) -> matplotlib.figure.Figure:
        """SHAP explanation figure for one hierarchy level.

        Produces one explanation panel per predicted label at *level*,
        each showing the molecule with the most important fingerprint bits
        highlighted, a SHAP bar chart, and a Morgan-environment fragment grid.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        level:
            ``"pathway"``, ``"superclass"``, or ``"class"``.
        background:
            Reference distribution for SHAP (SMILES list or feature array).
        k:
            Top-*k* bits to highlight.

        Returns
        -------
        matplotlib Figure.
        """
        return self._make_shap(background).draw_level(
            smiles, level, self.featurizer, k=k
        )
