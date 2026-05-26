"""Tests for the bundled SHAP background dataset and pipeline integration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_np_classifier import NPClassifierPipeline

_DATA_DIR = Path(__file__).parent.parent / "src" / "torch_np_classifier" / "data"
_BG_CSV = _DATA_DIR / "shap_background.csv"

_INPUT_DIM = 6144
_EXPECTED_ROWS = 200


# ---------------------------------------------------------------------------
# Bundled CSV file
# ---------------------------------------------------------------------------


class TestShapBackgroundFile:
    def test_file_exists(self):
        assert _BG_CSV.exists(), f"shap_background.csv not found at {_BG_CSV}"

    def test_has_smiles_column(self):
        df = pd.read_csv(_BG_CSV)
        assert "SMILES" in df.columns

    def test_row_count(self):
        df = pd.read_csv(_BG_CSV)
        assert len(df) == _EXPECTED_ROWS

    def test_no_missing_smiles(self):
        df = pd.read_csv(_BG_CSV)
        assert df["SMILES"].notna().all()

    def test_no_empty_smiles(self):
        df = pd.read_csv(_BG_CSV)
        assert (df["SMILES"].str.strip() != "").all()

    def test_smiles_are_unique(self):
        df = pd.read_csv(_BG_CSV)
        assert df["SMILES"].nunique() == len(df)

    def test_only_smiles_column(self):
        """Background CSV should contain only SMILES — no label columns."""
        df = pd.read_csv(_BG_CSV)
        assert list(df.columns) == ["SMILES"]


# ---------------------------------------------------------------------------
# load_shap_background
# ---------------------------------------------------------------------------


class TestLoadShapBackground:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return NPClassifierPipeline()

    @pytest.fixture(scope="class")
    def background(self, pipeline):
        return pipeline.load_shap_background()

    def test_returns_ndarray(self, background):
        assert isinstance(background, np.ndarray)

    def test_shape(self, background):
        assert background.shape == (_EXPECTED_ROWS, _INPUT_DIM)

    def test_dtype_float32(self, background):
        assert background.dtype == np.float32

    def test_values_non_negative(self, background):
        assert background.min() >= 0.0

    def test_not_all_zero(self, background):
        assert np.any(background > 0)


# ---------------------------------------------------------------------------
# _resolve_background
# ---------------------------------------------------------------------------


class TestResolveBackground:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return NPClassifierPipeline()

    def test_none_returns_bundled_background(self, pipeline):
        result = pipeline._resolve_background(None)
        assert result.shape == (_EXPECTED_ROWS, _INPUT_DIM)
        assert result.dtype == np.float32

    def test_array_passthrough(self, pipeline):
        arr = np.ones((5, _INPUT_DIM), dtype=np.float64)
        result = pipeline._resolve_background(arr)
        assert result.dtype == np.float32
        assert result.shape == (5, _INPUT_DIM)
        np.testing.assert_array_equal(result, arr.astype(np.float32))

    def test_smiles_list_featurizes(self, pipeline):
        smiles = ["Cn1cnc2c1c(=O)n(c(=O)n2C)C", "c1ccccc1"]
        result = pipeline._resolve_background(smiles)
        assert result.shape == (2, _INPUT_DIM)
        assert result.dtype == np.float32

    def test_none_and_array_consistent(self, pipeline):
        """Background loaded via None should equal the featurized CSV SMILES."""
        from torch_np_classifier.featurization.np_classifier_fp import (
            NPClassifierFeaturizer,
        )

        smiles = pd.read_csv(_BG_CSV)["SMILES"].tolist()
        expected = NPClassifierFeaturizer().transform(smiles)
        result = pipeline._resolve_background(None)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# explain / explain_bits default background (signature check)
# ---------------------------------------------------------------------------


class TestExplainDefaultBackground:
    """Verify the public API accepts background=None without TypeError."""

    def test_explain_accepts_none_background(self):
        import inspect

        sig = inspect.signature(NPClassifierPipeline.explain)
        param = sig.parameters["background"]
        assert param.default is None

    def test_explain_bits_accepts_none_background(self):
        import inspect

        sig = inspect.signature(NPClassifierPipeline.explain_bits)
        param = sig.parameters["background"]
        assert param.default is None
