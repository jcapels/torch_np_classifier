"""Tests for NPClassifierFeaturizer and featurize_smiles."""

import numpy as np
import pytest

from torch_np_classifier.featurization.np_classifier_fp import (
    NPClassifierFeaturizer,
    featurize_smiles,
)

CAFFEINE = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"

VALID_SMILES = [CAFFEINE, ASPIRIN, ETHANOL]


class TestFeaturizeSmiles:
    def test_output_shape(self):
        fp = featurize_smiles(CAFFEINE)
        assert fp is not None
        assert fp.shape == (6144,)

    def test_output_dtype(self):
        fp = featurize_smiles(CAFFEINE)
        assert fp is not None
        assert fp.dtype == np.float32

    def test_invalid_smiles_returns_none(self):
        result = featurize_smiles("not_a_smiles")
        assert result is None

    def test_different_smiles_different_fp(self):
        fp1 = featurize_smiles(CAFFEINE)
        fp2 = featurize_smiles(ASPIRIN)
        assert fp1 is not None and fp2 is not None
        assert not np.array_equal(fp1, fp2)

    def test_same_smiles_same_fp(self):
        fp1 = featurize_smiles(CAFFEINE)
        fp2 = featurize_smiles(CAFFEINE)
        assert fp1 is not None and fp2 is not None
        np.testing.assert_array_equal(fp1, fp2)


class TestNPClassifierFeaturizer:
    def test_invalid_smiles_featurizer_returns_zeros(self):
        feat = NPClassifierFeaturizer()
        result = feat.transform(["invalid_smiles_xyz"])
        assert result.shape == (1, 6144)
        np.testing.assert_array_equal(result[0], np.zeros(6144, dtype=np.float32))

    def test_batch_transform(self):
        feat = NPClassifierFeaturizer()
        result = feat.transform(VALID_SMILES)
        assert result.shape == (3, 6144)

    def test_batch_transform_dtype(self):
        feat = NPClassifierFeaturizer()
        result = feat.transform(VALID_SMILES)
        assert result.dtype == np.float32

    def test_different_smiles_different_fp(self):
        feat = NPClassifierFeaturizer()
        result = feat.transform([CAFFEINE, ASPIRIN])
        assert not np.array_equal(result[0], result[1])

    def test_same_smiles_same_fp(self):
        feat = NPClassifierFeaturizer()
        result = feat.transform([CAFFEINE, CAFFEINE])
        np.testing.assert_array_equal(result[0], result[1])

    def test_feature_dim_property(self):
        feat = NPClassifierFeaturizer(radius=2)
        assert feat.feature_dim == 6144

    def test_feature_dim_custom_radius(self):
        feat = NPClassifierFeaturizer(radius=3)
        assert feat.feature_dim == 2048 * 4
