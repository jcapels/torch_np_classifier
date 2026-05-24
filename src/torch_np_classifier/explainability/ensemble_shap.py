"""SHAP-based explainability for NPClassifierEnsemble.

Wraps three per-level ``NPClassifierSHAP`` explainers (one for pathway, one
for superclass, one for class) and provides drawing helpers that mirror the
single-model API but cover the full three-level hierarchy in one call.

Typical workflow
----------------
>>> bg   = featurizer.transform(background_smiles)   # (N, 6144)
>>> eshap = NPClassifierEnsembleSHAP(ensemble, bg)
>>>
>>> # Full combined figure (one panel per level, top voted label)
>>> fig  = eshap.explanation_figure("O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", featurizer)
>>>
>>> # Fragment grids + bar chart for one level only
>>> fig2 = eshap.draw_level("O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "class", featurizer)
"""

from __future__ import annotations

import io
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from torch_np_classifier.featurization.np_classifier_fp import NPClassifierFeaturizer
from torch_np_classifier.models.ensemble import NPClassifierEnsemble
from torch_np_classifier.explainability.shap_explainer import NPClassifierSHAP

# Level names in canonical display order
_LEVELS = ("pathway", "superclass", "class")


def _level_attrs(ensemble: NPClassifierEnsemble):
    """Return (model, label_list) for each level in canonical order."""
    return {
        "pathway": (ensemble.pathway_model, ensemble._pathway_labels),
        "superclass": (ensemble.superclass_model, ensemble._superclass_labels),
        "class": (ensemble.class_model, ensemble._class_labels),
    }


class NPClassifierEnsembleSHAP:
    """SHAP explainability wrapper for :class:`NPClassifierEnsemble`.

    For a given molecule, first runs the ensemble voting prediction to obtain
    predicted labels per level, then builds a per-level
    :class:`~torch_np_classifier.explainability.shap_explainer.NPClassifierSHAP`
    restricted to those predicted class indices only (for speed), and finally
    offers drawing helpers at the level of individual levels or the full
    three-level hierarchy.

    Parameters
    ----------
    ensemble:
        A ready-to-use :class:`NPClassifierEnsemble`.
    background_features:
        Float32 array of shape ``(n_background, 6144)`` — a sample of ~100
        training fingerprints used as the SHAP reference distribution.
    """

    def __init__(
        self,
        ensemble: NPClassifierEnsemble,
        background_features: np.ndarray,
    ) -> None:
        self.ensemble = ensemble
        self._bg = np.asarray(background_features, dtype=np.float32)

    # ── core SHAP computation ────────────────────────────────────────────────

    def _build_level_explainer(
        self,
        level: str,
        features: np.ndarray,
        predicted_labels: List[str],
    ) -> Optional[Dict]:
        """Build a per-level SHAP explainer and compute values.

        Returns ``None`` if *predicted_labels* is empty or none of them are
        found in the model's label list.

        Returns a dict::

            {
                "labels":    [str, ...],          predicted label names
                "indices":   [int, ...],          model output positions
                "shap":      np.ndarray,          shape (1, 6144, n_predicted)
                "explainer": NPClassifierSHAP,
            }
        """
        if not predicted_labels:
            return None

        attrs = _level_attrs(self.ensemble)
        model, label_list = attrs[level]

        indices = [
            label_list.index(lbl) for lbl in predicted_labels if lbl in label_list
        ]
        if not indices:
            warnings.warn(
                f"None of the predicted {level} labels {predicted_labels!r} "
                "found in the model's label list — skipping SHAP for this level.",
                UserWarning,
                stacklevel=3,
            )
            return None

        explainer = NPClassifierSHAP(model, self._bg, class_indices=indices)
        sv = explainer.shap_values(features)  # (1, 6144, n_predicted)

        return {
            "labels": predicted_labels,
            "indices": indices,
            "shap": sv,
            "explainer": explainer,
        }

    def explain_smiles(
        self,
        smiles: str,
        featurizer: NPClassifierFeaturizer,
    ) -> Dict:
        """Run ensemble voting + compute per-level SHAP values.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        featurizer:
            The ``NPClassifierFeaturizer`` used during training.

        Returns
        -------
        dict with keys:

        - ``"prediction"`` — voted result dict (pathway / superclass / class /
          isglycoside) from :meth:`~NPClassifierEnsemble.predict`.
        - ``"features"``   — (1, 6144) float32 array.
        - ``"pathway"``    — per-level SHAP data dict (or ``None``).
        - ``"superclass"`` — per-level SHAP data dict (or ``None``).
        - ``"class"``      — per-level SHAP data dict (or ``None``).

        Each per-level dict has keys ``"labels"``, ``"indices"``, ``"shap"``,
        ``"explainer"`` (see :meth:`_build_level_explainer`).
        """
        features = featurizer.transform([smiles])
        prediction = self.ensemble.predict(smiles, check_glycoside=True)

        result: Dict = {"prediction": prediction, "features": features}
        for level in _LEVELS:
            result[level] = self._build_level_explainer(
                level, features, prediction[level]
            )
        return result

    # ── drawing: one level ───────────────────────────────────────────────────

    def draw_level(
        self,
        smiles: str,
        level: str,
        featurizer: NPClassifierFeaturizer,
        k: int = 6,
    ) -> "matplotlib.figure.Figure":  # noqa: F821
        """Create explanation figures for every predicted label at *level*.

        One :meth:`~NPClassifierSHAP.explanation_figure` panel is produced per
        predicted label (molecule + SHAP bar chart + bit fragments).  When the
        level has no predictions the figure carries an informational message.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        level:
            ``"pathway"``, ``"superclass"``, or ``"class"``.
        featurizer:
            The ``NPClassifierFeaturizer`` used during training.
        k:
            Top-*k* bits to highlight per label.

        Returns
        -------
        matplotlib Figure.
        """
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib and Pillow are required: pip install matplotlib Pillow"
            ) from exc

        exp = self.explain_smiles(smiles, featurizer)
        level_data = exp[level]

        if level_data is None:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(
                0.5,
                0.5,
                f"No {level} predictions for this molecule.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.axis("off")
            return fig

        panels = []
        for pos, label in enumerate(level_data["labels"]):
            sv_1d = level_data["shap"][0, :, pos]
            fig_panel = level_data["explainer"].explanation_figure(
                smiles,
                sv_1d,
                featurizer,
                class_name=f"{level}: {label}",
                k=k,
            )
            panels.append(self._fig_to_array(fig_panel))
            plt.close(fig_panel)

        return self._stack_panels(panels, title=f"Level: {level.upper()}")

    # ── drawing: full ensemble explanation ───────────────────────────────────

    def explanation_figure(
        self,
        smiles: str,
        featurizer: NPClassifierFeaturizer,
        k: int = 6,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> "matplotlib.figure.Figure":  # noqa: F821
        """Combined three-level explanation figure.

        Produces one panel per level (pathway / superclass / class), each
        showing the **top voted label** at that level — the molecule with
        highlighted bits, the SHAP bar chart, and the bit-fragment grid.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        featurizer:
            The ``NPClassifierFeaturizer`` used during training.
        k:
            Top-*k* bits to highlight per level.
        figsize:
            Override ``(width, height)`` in inches.

        Returns
        -------
        matplotlib Figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib and Pillow are required: pip install matplotlib Pillow"
            ) from exc

        exp = self.explain_smiles(smiles, featurizer)
        pred = exp["prediction"]

        panels = []  # (level_name, label_name, image_array)
        for level in _LEVELS:
            level_data = exp[level]
            if level_data is None or not level_data["labels"]:
                continue
            # Use only the top voted label at this level
            label = level_data["labels"][0]
            sv_1d = level_data["shap"][0, :, 0]
            fig_panel = level_data["explainer"].explanation_figure(
                smiles,
                sv_1d,
                featurizer,
                class_name=f"{level}: {label}",
                k=k,
            )
            panels.append((level, label, self._fig_to_array(fig_panel)))
            plt.close(fig_panel)

        if not panels:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(
                0.5,
                0.5,
                "No predictions found for any level.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.axis("off")
            return fig

        n = len(panels)
        if figsize is None:
            figsize = (14, 7.5 * n)

        fig, axes = plt.subplots(n, 1, figsize=figsize)
        if n == 1:
            axes = [axes]

        for ax, (level, label, img) in zip(axes, panels):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(
                f"{level.upper()} — {label}",
                fontsize=11,
                fontweight="bold",
                pad=3,
            )

        glyco_tag = "  [glycoside]" if pred.get("isglycoside") else ""
        path_str = ", ".join(pred["pathway"]) or "—"
        sup_str = ", ".join(pred["superclass"]) or "—"
        cls_str = ", ".join(pred["class"]) or "—"
        suptitle = (
            f"Ensemble SHAP Explanation{glyco_tag}\n"
            f"Pathway: {path_str}   |   "
            f"Superclass: {sup_str}   |   "
            f"Class: {cls_str}"
        )
        fig.suptitle(suptitle, fontsize=9, y=1.01)
        fig.tight_layout()
        return fig

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_array(fig: "matplotlib.figure.Figure") -> np.ndarray:  # noqa: F821
        """Render a matplotlib Figure to an RGBA numpy array."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        from PIL import Image

        return np.array(Image.open(buf))

    @staticmethod
    def _stack_panels(
        panels: List[np.ndarray],
        title: str = "",
    ) -> "matplotlib.figure.Figure":  # noqa: F821
        """Vertically stack image panels into a single matplotlib Figure."""
        import matplotlib.pyplot as plt

        n = len(panels)
        heights = [p.shape[0] for p in panels]
        total_h = sum(heights)
        fig_h = max(total_h / 100, 4)  # approx inches at 100 dpi
        fig_w = max(panels[0].shape[1] / 100, 8)

        fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h))
        if n == 1:
            axes = [axes]

        for ax, img in zip(axes, panels):
            ax.imshow(img)
            ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig
