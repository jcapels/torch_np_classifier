"""Tests for NPClassifierSHAP."""

import numpy as np
import pytest
import torch

from torch_np_classifier import NPClassifierFeaturizer, NPClassifierLightning
from torch_np_classifier.explainability.shap_explainer import (
    NPClassifierSHAP,
    _ClassSubsetWrapper,
)

shap = pytest.importorskip("shap")

# Small constants so tests run quickly
INPUT_DIM = 6144
NUM_CATS = 10  # small model for speed
N_BACKGROUND = 8
N_SAMPLES = 3
CLASS_IDX = [0, 2, 5]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    m = NPClassifierLightning(num_categories=NUM_CATS)
    m.eval()
    return m


@pytest.fixture(scope="module")
def background():
    rng = np.random.default_rng(0)
    return rng.random((N_BACKGROUND, INPUT_DIM)).astype(np.float32)


@pytest.fixture(scope="module")
def sample_features():
    rng = np.random.default_rng(1)
    return rng.random((N_SAMPLES, INPUT_DIM)).astype(np.float32)


@pytest.fixture(scope="module")
def explainer_subset(small_model, background):
    return NPClassifierSHAP(small_model, background, class_indices=CLASS_IDX)


@pytest.fixture(scope="module")
def explainer_all(small_model, background):
    return NPClassifierSHAP(small_model, background, class_indices=None)


# ---------------------------------------------------------------------------
# _ClassSubsetWrapper
# ---------------------------------------------------------------------------


class TestClassSubsetWrapper:
    def test_full_output(self, small_model):
        wrapper = _ClassSubsetWrapper(small_model.model, class_indices=None)
        x = torch.randn(2, INPUT_DIM)
        out = wrapper(x)
        assert out.shape == (2, NUM_CATS)

    def test_subset_output(self, small_model):
        wrapper = _ClassSubsetWrapper(small_model.model, class_indices=CLASS_IDX)
        x = torch.randn(2, INPUT_DIM)
        out = wrapper(x)
        assert out.shape == (2, len(CLASS_IDX))

    def test_subset_values_match_full(self, small_model):
        full = _ClassSubsetWrapper(small_model.model, class_indices=None)
        subset = _ClassSubsetWrapper(small_model.model, class_indices=CLASS_IDX)
        x = torch.randn(2, INPUT_DIM)
        with torch.no_grad():
            full_out = full(x)
            subset_out = subset(x)
        for pos, cls in enumerate(CLASS_IDX):
            np.testing.assert_allclose(
                subset_out[:, pos].numpy(),
                full_out[:, cls].numpy(),
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# NPClassifierSHAP construction
# ---------------------------------------------------------------------------


class TestNPClassifierSHAPInit:
    def test_stores_class_indices(self, explainer_subset):
        assert explainer_subset.class_indices == CLASS_IDX

    def test_none_class_indices(self, explainer_all):
        assert explainer_all.class_indices is None

    def test_has_explainer_attribute(self, explainer_subset):
        assert hasattr(explainer_subset, "explainer")


# ---------------------------------------------------------------------------
# shap_values
# ---------------------------------------------------------------------------


class TestShapValues:
    def test_shape_subset(self, explainer_subset, sample_features):
        sv = explainer_subset.shap_values(sample_features)
        assert sv.shape == (N_SAMPLES, INPUT_DIM, len(CLASS_IDX))

    def test_shape_all(self, explainer_all, sample_features):
        sv = explainer_all.shap_values(sample_features)
        assert sv.shape == (N_SAMPLES, INPUT_DIM, NUM_CATS)

    def test_dtype_float(self, explainer_subset, sample_features):
        sv = explainer_subset.shap_values(sample_features)
        assert sv.dtype in (np.float32, np.float64)

    def test_not_all_zero(self, explainer_subset, sample_features):
        sv = explainer_subset.shap_values(sample_features)
        assert np.any(sv != 0.0)


# ---------------------------------------------------------------------------
# explain_smiles
# ---------------------------------------------------------------------------


class TestExplainSmiles:
    def test_shape(self, explainer_subset):
        feat = NPClassifierFeaturizer()
        sv = explainer_subset.explain_smiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C", feat)
        assert sv.shape == (1, INPUT_DIM, len(CLASS_IDX))


# ---------------------------------------------------------------------------
# top_feature_indices
# ---------------------------------------------------------------------------


class TestTopFeatureIndices:
    @pytest.fixture
    def sv_1d(self):
        arr = np.zeros(INPUT_DIM)
        arr[10] = 0.9  # highest positive
        arr[500] = 0.5  # second positive
        arr[200] = -0.8  # negative (ignored in positive_only mode)
        arr[1] = 0.3  # third positive
        return arr

    def test_returns_list(self, explainer_subset, sv_1d):
        result = explainer_subset.top_feature_indices(sv_1d, k=3)
        assert isinstance(result, list)

    def test_top_k_length(self, explainer_subset, sv_1d):
        result = explainer_subset.top_feature_indices(sv_1d, k=3)
        assert len(result) == 3

    def test_top_k_order_positive_only(self, explainer_subset, sv_1d):
        result = explainer_subset.top_feature_indices(sv_1d, k=3, positive_only=True)
        assert result[0] == 10  # highest positive first
        assert result[1] == 500
        assert result[2] == 1

    def test_negative_excluded_in_positive_only(self, explainer_subset, sv_1d):
        result = explainer_subset.top_feature_indices(sv_1d, k=5, positive_only=True)
        assert 200 not in result  # negative SHAP → excluded

    def test_abs_mode_includes_negative(self, explainer_subset, sv_1d):
        result = explainer_subset.top_feature_indices(sv_1d, k=2, positive_only=False)
        assert 10 in result  # abs 0.9
        assert 200 in result  # abs 0.8

    def test_k_larger_than_nonzero(self, explainer_subset):
        arr = np.zeros(INPUT_DIM)
        arr[0] = 0.7
        result = explainer_subset.top_feature_indices(arr, k=10)
        assert len(result) == 1  # only one non-zero

    def test_all_zero_returns_empty(self, explainer_subset):
        arr = np.zeros(INPUT_DIM)
        result = explainer_subset.top_feature_indices(arr, k=5)
        assert result == []


# ---------------------------------------------------------------------------
# draw_explanation
# ---------------------------------------------------------------------------


class TestDrawExplanation:
    @pytest.fixture
    def sv_1d(self, explainer_subset, sample_features):
        sv = explainer_subset.shap_values(sample_features[:1])
        return sv[0, :, 0]

    def test_returns_svg_string(self, explainer_subset, sv_1d):
        feat = NPClassifierFeaturizer()
        result = explainer_subset.draw_explanation("Cn1cnc2c1c(=O)n(c(=O)n2C)C", sv_1d, feat, as_svg=True)
        assert isinstance(result, str)
        assert result.startswith("<?xml")

    def test_returns_image_when_not_svg(self, explainer_subset, sv_1d):
        from PIL import Image

        feat = NPClassifierFeaturizer()
        result = explainer_subset.draw_explanation("Cn1cnc2c1c(=O)n(c(=O)n2C)C", sv_1d, feat, as_svg=False)
        assert isinstance(result, Image.Image)

    def test_all_zero_shap_draws_without_error(self, explainer_subset):
        feat = NPClassifierFeaturizer()
        sv = np.zeros(INPUT_DIM)
        result = explainer_subset.draw_explanation("Cn1cnc2c1c(=O)n(c(=O)n2C)C", sv, feat, as_svg=True)
        assert isinstance(result, str)
