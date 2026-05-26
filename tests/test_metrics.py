import numpy as np
import pytest
from torch_np_classifier.utils.metrics import (
    f1_macro,
    average_precision_macro,
    f1_macro_by_group,
    average_precision_macro_by_group,
    cosine_similarity_mean_sd,
    cosine_similarity_mean_sd_by_group,
    map_mean_sd,
    map_mean_sd_by_group,
)


@pytest.fixture
def binary_labels():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=(100, 10))
    y_pred = np.random.randint(0, 2, size=(100, 10))
    return y_true, y_pred


@pytest.fixture
def perfect_labels():
    y = np.eye(10, dtype=int)
    return y, y


class TestF1Macro:
    def test_returns_float(self, binary_labels):
        y_true, y_pred = binary_labels
        score = f1_macro(y_true, y_pred)
        assert isinstance(score, float)

    def test_range(self, binary_labels):
        y_true, y_pred = binary_labels
        score = f1_macro(y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_perfect_score(self, perfect_labels):
        y_true, y_pred = perfect_labels
        assert f1_macro(y_true, y_pred) == pytest.approx(1.0)

    def test_zero_division_handled(self):
        y_true = np.zeros((10, 5), dtype=int)
        y_pred = np.zeros((10, 5), dtype=int)
        score = f1_macro(y_true, y_pred)
        assert score == pytest.approx(0.0)


class TestAveragePrecisionMacro:
    def test_returns_float(self, binary_labels):
        y_true, y_pred = binary_labels
        score = average_precision_macro(y_true, y_pred.astype(float))
        assert isinstance(score, float)

    def test_range(self, binary_labels):
        y_true, y_pred = binary_labels
        score = average_precision_macro(y_true, y_pred.astype(float))
        assert 0.0 <= score <= 1.0

    def test_perfect_score(self, perfect_labels):
        y_true, y_pred = perfect_labels
        assert average_precision_macro(y_true, y_pred.astype(float)) == pytest.approx(1.0)


class TestGroupMetrics:
    def test_f1_macro_by_group(self, binary_labels):
        y_true, y_pred = binary_labels
        indexes = [0, 1, 2]
        score_group = f1_macro_by_group(y_true, y_pred, indexes)
        score_direct = f1_macro(y_true[:, indexes], y_pred[:, indexes])
        assert score_group == pytest.approx(score_direct)

    def test_average_precision_by_group(self, binary_labels):
        y_true, y_pred = binary_labels
        indexes = [3, 4, 5]
        score_group = average_precision_macro_by_group(y_true, y_pred.astype(float), indexes)
        score_direct = average_precision_macro(y_true[:, indexes], y_pred[:, indexes].astype(float))
        assert score_group == pytest.approx(score_direct)


class TestCosineSimilarityMeanSD:
    def test_returns_two_floats(self, binary_labels):
        y_true, y_pred = binary_labels
        mean, sd = cosine_similarity_mean_sd(y_true.astype(float), y_pred.astype(float))
        assert isinstance(mean, float)
        assert isinstance(sd, float)

    def test_range(self, binary_labels):
        y_true, y_pred = binary_labels
        mean, sd = cosine_similarity_mean_sd(y_true.astype(float), y_pred.astype(float))
        assert -1.0 <= mean <= 1.0
        assert sd >= 0.0

    def test_perfect_score(self, perfect_labels):
        y_true, y_pred = perfect_labels
        mean, sd = cosine_similarity_mean_sd(y_true.astype(float), y_pred.astype(float))
        assert mean == pytest.approx(1.0)
        assert sd == pytest.approx(0.0)

    def test_all_zero_pred_gives_zero(self):
        y_true = np.ones((5, 4), dtype=float)
        y_pred = np.zeros((5, 4), dtype=float)
        mean, sd = cosine_similarity_mean_sd(y_true, y_pred)
        assert mean == pytest.approx(0.0)

    def test_by_group_matches_direct(self, binary_labels):
        y_true, y_pred = binary_labels
        indexes = [0, 1, 2]
        g_mean, g_sd = cosine_similarity_mean_sd_by_group(y_true.astype(float), y_pred.astype(float), indexes)
        d_mean, d_sd = cosine_similarity_mean_sd(y_true[:, indexes].astype(float), y_pred[:, indexes].astype(float))
        assert g_mean == pytest.approx(d_mean)
        assert g_sd == pytest.approx(d_sd)


class TestMAPMeanSD:
    def test_returns_two_floats(self, binary_labels):
        y_true, y_pred = binary_labels
        mean, sd = map_mean_sd(y_true.astype(float), y_pred.astype(float))
        assert isinstance(mean, float)
        assert isinstance(sd, float)

    def test_range(self, binary_labels):
        y_true, y_pred = binary_labels
        mean, sd = map_mean_sd(y_true.astype(float), y_pred.astype(float))
        assert 0.0 <= mean <= 1.0
        assert sd >= 0.0

    def test_perfect_score(self, perfect_labels):
        y_true, y_pred = perfect_labels
        mean, sd = map_mean_sd(y_true.astype(float), y_pred.astype(float))
        assert mean == pytest.approx(1.0)
        assert sd == pytest.approx(0.0)

    def test_all_zero_true_returns_zero(self):
        y_true = np.zeros((10, 5), dtype=float)
        y_pred = np.random.rand(10, 5)
        mean, sd = map_mean_sd(y_true, y_pred)
        assert mean == pytest.approx(0.0)
        assert sd == pytest.approx(0.0)

    def test_by_group_matches_direct(self, binary_labels):
        y_true, y_pred = binary_labels
        indexes = [3, 4, 5]
        g_mean, g_sd = map_mean_sd_by_group(y_true.astype(float), y_pred.astype(float), indexes)
        d_mean, d_sd = map_mean_sd(y_true[:, indexes].astype(float), y_pred[:, indexes].astype(float))
        assert g_mean == pytest.approx(d_mean)
        assert g_sd == pytest.approx(d_sd)
