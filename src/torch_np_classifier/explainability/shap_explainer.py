"""SHAP-based explainability for NPClassifier predictions.

Uses ``shap.GradientExplainer`` (expected gradients) — preferred over
DeepExplainer for models with BatchNorm layers.

Typical workflow
----------------
1. Sample ~100 fingerprints from the training set as background.
2. Build an ``NPClassifierSHAP`` with the trained model and the background,
   restricting to the class indices you care about.
3. Call ``shap_values`` on one or more fingerprints.
4. Call ``draw_explanation`` with the per-class SHAP array to highlight the
   structural features that drove the prediction.

Example
-------
>>> bg  = train_features[np.random.choice(len(train_features), 100)]
>>> exp = NPClassifierSHAP(model, bg, class_indices=[0, 5, 42])
>>> sv  = exp.shap_values(feat)            # (1, 6144, 3)
>>> svg = exp.draw_explanation(smiles, sv[0, :, 0], featurizer)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from torch_np_classifier.featurization.np_classifier_fp import (
    NPClassifierFeaturizer,
    fp_index_to_radius_bit,
    _COLORS,
)


class _ClassSubsetWrapper(nn.Module):
    """Expose only *class_indices* outputs of the underlying DNN.

    Limiting the output count makes GradientExplainer compute gradients only
    for the requested classes, which is significantly faster and saves memory.
    """

    def __init__(self, dnn: nn.Module, class_indices: Optional[List[int]]) -> None:
        super().__init__()
        self.dnn = dnn
        self.class_indices = class_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dnn([x])
        if self.class_indices is not None:
            out = out[:, self.class_indices]
        return out


class NPClassifierSHAP:
    """SHAP explainer for NP Classifier predictions.

    Parameters
    ----------
    model:
        Trained ``NPClassifierLightning`` module.
    background_features:
        Float32 array of shape ``(n_background, 6144)``.  A random sample of
        50–200 training fingerprints works well — smaller is faster, larger
        gives more stable estimates.
    class_indices:
        Output class indices to explain.  Pass only the classes you are
        interested in (e.g. the predicted-positive classes) rather than all
        730, which is very slow.  ``None`` explains every output.

    Attributes
    ----------
    class_indices : list[int] | None
        The class indices this explainer was built for.
    explainer : shap.GradientExplainer
        The underlying SHAP explainer (useful for advanced use).
    """

    def __init__(
        self,
        model: "NPClassifierLightning",  # noqa: F821
        background_features: np.ndarray,
        class_indices: Optional[List[int]] = None,
    ) -> None:
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required: pip install shap") from exc

        self.class_indices = class_indices

        self._wrapper = _ClassSubsetWrapper(model.model, class_indices)
        self._wrapper.eval()

        background = torch.tensor(np.asarray(background_features), dtype=torch.float32)
        self.explainer = shap.GradientExplainer(self._wrapper, background)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def shap_values(self, features: np.ndarray) -> np.ndarray:
        """Compute SHAP values for *features*.

        Parameters
        ----------
        features:
            Float32 array of shape ``(n_samples, 6144)``.

        Returns
        -------
        np.ndarray of shape ``(n_samples, 6144, n_classes_explained)``.

        - Positive values → the feature pushes that class probability **up**.
        - Negative values → the feature pushes it **down**.
        - Axis 2 aligns with ``self.class_indices`` (or all classes if None).
        """
        x = torch.tensor(np.asarray(features), dtype=torch.float32)
        raw = self.explainer.shap_values(x)
        # GradientExplainer returns list[array(n,f)] for multi-output,
        # or a single array(n,f) for single-output models/subsets.
        raw = np.asarray(raw) if not isinstance(raw, list) else np.stack(raw, axis=-1)
        # Normalize to (n_samples, n_features, n_classes)
        if raw.ndim == 2:
            return raw[:, :, np.newaxis]
        return raw

    def explain_smiles(
        self,
        smiles: str,
        featurizer: NPClassifierFeaturizer,
    ) -> np.ndarray:
        """Featurize *smiles* and return SHAP values in one call.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        featurizer:
            The ``NPClassifierFeaturizer`` used during training.

        Returns
        -------
        np.ndarray of shape ``(1, 6144, n_classes_explained)``.
        """
        features = featurizer.transform([smiles])
        return self.shap_values(features)

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def top_feature_indices(
        self,
        shap_values_1d: np.ndarray,
        k: int = 10,
        positive_only: bool = True,
    ) -> List[int]:
        """Return the top-*k* feature indices ranked by SHAP contribution.

        Parameters
        ----------
        shap_values_1d:
            SHAP values for one sample and one class: shape ``(6144,)``.
        k:
            Number of features to return.
        positive_only:
            If ``True`` (default), only consider features with **positive**
            SHAP values — those that push the predicted probability up.
            Set to ``False`` to rank by absolute value instead.

        Returns
        -------
        List of up to *k* feature indices, most important first.
        """
        scores = np.asarray(shap_values_1d, dtype=float).copy()
        if positive_only:
            scores = np.where(scores > 0, scores, 0.0)
        else:
            scores = np.abs(scores)
        order = np.argsort(scores)[::-1]
        nonzero = order[scores[order] > 0]
        return nonzero[:k].tolist()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw_explanation(
        self,
        smiles: str,
        shap_values_1d: np.ndarray,
        featurizer: NPClassifierFeaturizer,
        k: int = 10,
        positive_only: bool = True,
        size: Tuple[int, int] = (400, 300),
        as_svg: bool = True,
        per_feature_colors: bool = True,
    ) -> Union[str, "PIL.Image.Image"]:  # noqa: F821
        """Highlight the top-*k* SHAP features on *smiles*.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        shap_values_1d:
            SHAP values for one sample and one class: shape ``(6144,)``.
            Typically ``shap_vals[sample_idx, :, class_pos]`` where
            ``class_pos`` is the position of the class in ``class_indices``.
        featurizer:
            The ``NPClassifierFeaturizer`` used during training.
        k:
            Number of top features to highlight.
        positive_only:
            Only highlight features with positive SHAP values.
        size, as_svg, per_feature_colors:
            Forwarded to :meth:`~NPClassifierFeaturizer.draw_bits`.

        Returns
        -------
        SVG string or PIL ``Image``.
        """
        top_indices = self.top_feature_indices(
            shap_values_1d, k=k, positive_only=positive_only
        )
        return featurizer.draw_bits(
            smiles,
            top_indices,
            size=size,
            as_svg=as_svg,
            per_feature_colors=per_feature_colors,
        )

    def draw_bit_fragments(
        self,
        smiles: str,
        shap_values_1d: np.ndarray,
        featurizer: NPClassifierFeaturizer,
        k: int = 6,
        positive_only: bool = True,
        fragment_size: Tuple[int, int] = (150, 150),
        cols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "matplotlib.figure.Figure":  # noqa: F821
        """Return a matplotlib figure showing the top-*k* Morgan bit fragments.

        Each cell displays the isolated Morgan environment (substructure) for one
        of the top-*k* features ranked by SHAP value, annotated with its rank,
        Morgan radius, and SHAP contribution.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        shap_values_1d:
            SHAP values for one sample and one class: shape ``(6144,)``.
        featurizer:
            The ``NPClassifierFeaturizer`` used during training.
        k:
            Number of top features to draw.
        positive_only:
            Only include features with positive SHAP values.
        fragment_size:
            ``(width, height)`` in pixels for each individual fragment image.
        cols:
            Number of columns in the fragment grid.
        figsize:
            Override the automatic figure size ``(width, height)`` in inches.

        Returns
        -------
        matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required: pip install matplotlib") from exc

        top_indices = self.top_feature_indices(
            shap_values_1d, k=k, positive_only=positive_only
        )
        scores = np.asarray(shap_values_1d, dtype=float)

        fragments = []
        for rank, fp_idx in enumerate(top_indices):
            img = featurizer.draw_bit_fragment(
                smiles, fp_idx, size=fragment_size, as_svg=False
            )
            if img is not None:
                env_r, bit = fp_index_to_radius_bit(fp_idx, featurizer.radius)
                fragments.append(
                    dict(
                        rank=rank + 1,
                        fp_idx=fp_idx,
                        shap=scores[fp_idx],
                        env_radius=env_r,
                        morgan_bit=bit,
                        img=img,
                    )
                )

        n = len(fragments)
        if n == 0:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(
                0.5,
                0.5,
                "No active bits found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.axis("off")
            return fig

        rows = (n + cols - 1) // cols
        if figsize is None:
            cell_w = fragment_size[0] / 72
            cell_h = fragment_size[1] / 72 + 0.5
            figsize = (max(cell_w * cols, 4), max(cell_h * rows, 2))

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(rows, cols)

        for i, frag in enumerate(fragments):
            row, col = divmod(i, cols)
            ax = axes[row, col]
            ax.imshow(np.array(frag["img"]))
            ax.axis("off")
            ax.set_title(
                f"#{frag['rank']}  r={frag['env_radius']}  SHAP={frag['shap']:.4f}",
                fontsize=8,
            )

        for i in range(n, rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].axis("off")

        fig.tight_layout()
        return fig

    def explanation_figure(
        self,
        smiles: str,
        shap_values_1d: np.ndarray,
        featurizer: NPClassifierFeaturizer,
        class_name: str = "",
        k: int = 6,
        positive_only: bool = True,
        mol_size: Tuple[int, int] = (400, 300),
        fragment_size: Tuple[int, int] = (150, 150),
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "matplotlib.figure.Figure":  # noqa: F821
        """Create a complete matplotlib explanation figure.

        The figure has two rows:

        - **Top row** — (left) full molecule with the top-*k* bits highlighted
          in distinct colours; (right) horizontal bar chart of SHAP values.
        - **Bottom row** — grid of isolated Morgan-environment fragment images
          for the top-*k* bits, each labelled with rank, radius, and SHAP value.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        shap_values_1d:
            SHAP values for one sample and one class: shape ``(6144,)``.
        featurizer:
            The ``NPClassifierFeaturizer`` used during training.
        class_name:
            Optional label for the predicted class (used as figure title).
        k:
            Number of top features to explain.
        positive_only:
            Only include features with positive SHAP values.
        mol_size:
            ``(width, height)`` in pixels for the full-molecule image.
        fragment_size:
            ``(width, height)`` in pixels for each fragment image.
        figsize:
            Override the automatic figure size ``(width, height)`` in inches.

        Returns
        -------
        matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError as exc:
            raise ImportError("matplotlib is required: pip install matplotlib") from exc

        top_indices = self.top_feature_indices(
            shap_values_1d, k=k, positive_only=positive_only
        )
        scores = np.asarray(shap_values_1d, dtype=float)

        # Gather fragment images
        fragments = []
        for rank, fp_idx in enumerate(top_indices):
            img = featurizer.draw_bit_fragment(
                smiles, fp_idx, size=fragment_size, as_svg=False
            )
            if img is not None:
                env_r, bit = fp_index_to_radius_bit(fp_idx, featurizer.radius)
                fragments.append(
                    dict(
                        rank=rank + 1,
                        fp_idx=fp_idx,
                        shap=scores[fp_idx],
                        env_radius=env_r,
                        morgan_bit=bit,
                        img=img,
                    )
                )

        n_frags = len(fragments)
        frag_cols = min(max(n_frags, 1), k)
        frag_rows = max(1, (n_frags + frag_cols - 1) // frag_cols)

        if figsize is None:
            figsize = (12, 4 + frag_rows * 2.4)

        fig = plt.figure(figsize=figsize)
        title = "SHAP Explanation" + (f": {class_name}" if class_name else "")
        fig.suptitle(title, fontsize=13, fontweight="bold")

        gs = gridspec.GridSpec(
            2,
            2,
            height_ratios=[3, frag_rows * 1.6],
            hspace=0.5,
            wspace=0.3,
            top=0.92,
        )

        # --- Full molecule (top-left) ---
        ax_mol = fig.add_subplot(gs[0, 0])
        mol_img = self.draw_explanation(
            smiles,
            shap_values_1d,
            featurizer,
            k=k,
            positive_only=positive_only,
            size=mol_size,
            as_svg=False,
            per_feature_colors=True,
        )
        ax_mol.imshow(np.array(mol_img))
        ax_mol.axis("off")
        ax_mol.set_title("Top bits highlighted on molecule", fontsize=9)

        # --- SHAP bar chart (top-right) ---
        ax_bar = fig.add_subplot(gs[0, 1])
        if top_indices:
            bar_labels = [
                f"r{fp_index_to_radius_bit(i, featurizer.radius)[0]}"
                f"·b{fp_index_to_radius_bit(i, featurizer.radius)[1]}"
                for i in top_indices
            ]
            bar_vals = [scores[i] for i in top_indices]
            colors_rgb = [_COLORS[j % len(_COLORS)] for j in range(len(top_indices))]
            y_pos = list(range(len(top_indices)))[::-1]
            ax_bar.barh(y_pos, bar_vals, color=colors_rgb)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(bar_labels, fontsize=8)
            ax_bar.set_xlabel("SHAP value", fontsize=8)
            ax_bar.set_title("Feature importance (SHAP)", fontsize=9)
            ax_bar.tick_params(axis="both", labelsize=7)
        else:
            ax_bar.text(
                0.5,
                0.5,
                "No active bits",
                ha="center",
                va="center",
                transform=ax_bar.transAxes,
            )
            ax_bar.axis("off")

        # --- Fragment grid (bottom, spanning both columns) ---
        if fragments:
            gs_frags = gridspec.GridSpecFromSubplotSpec(
                frag_rows,
                frag_cols,
                subplot_spec=gs[1, :],
                hspace=0.6,
                wspace=0.1,
            )
            for i, frag in enumerate(fragments):
                row, col = divmod(i, frag_cols)
                ax_f = fig.add_subplot(gs_frags[row, col])
                ax_f.imshow(np.array(frag["img"]))
                ax_f.axis("off")
                ax_f.set_title(
                    f"#{frag['rank']}  r={frag['env_radius']}\nSHAP={frag['shap']:.4f}",
                    fontsize=7,
                )
            for i in range(n_frags, frag_rows * frag_cols):
                row, col = divmod(i, frag_cols)
                ax_f = fig.add_subplot(gs_frags[row, col])
                ax_f.axis("off")

        return fig
