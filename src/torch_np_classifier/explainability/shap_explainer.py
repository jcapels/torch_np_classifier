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

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from torch_np_classifier.featurization.np_classifier_fp import NPClassifierFeaturizer


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
