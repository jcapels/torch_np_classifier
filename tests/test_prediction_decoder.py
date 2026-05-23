import pickle
import importlib.resources

import numpy as np
import pytest

from torch_np_classifier.utils.prediction_decoder import (
    CLASS_SLICE,
    PATHWAY_SLICE,
    SUPERCLASS_SLICE,
    decode_predictions,
)

NUM_LABELS = 730


@pytest.fixture(scope="session")
def label_names():
    ref = importlib.resources.files("torch_np_classifier.data").joinpath("label_names.pkl")
    with importlib.resources.as_file(ref) as path:
        with open(path, "rb") as f:
            return pickle.load(f)


@pytest.fixture
def single_probs():
    """Probabilities for one molecule: indices 0, 8, 100 above threshold."""
    probs = np.zeros(NUM_LABELS)
    probs[0] = 0.9    # pathway
    probs[8] = 0.8    # superclass
    probs[100] = 0.7  # class
    return probs


@pytest.fixture
def batch_probs(single_probs):
    """Batch of two molecules."""
    second = np.zeros(NUM_LABELS)
    second[3] = 0.95    # pathway
    second[10] = 0.6    # superclass
    second[200] = 0.51  # class
    return np.stack([single_probs, second])


class TestDecodeSlices:
    def test_pathway_slice(self):
        assert PATHWAY_SLICE == slice(None, 7)

    def test_superclass_slice(self):
        assert SUPERCLASS_SLICE == slice(7, 77)

    def test_class_slice(self):
        assert CLASS_SLICE == slice(77, None)


class TestLabelNamesPickle:
    def test_pickle_loads(self, label_names):
        assert isinstance(label_names, list)

    def test_pickle_count(self, label_names):
        assert len(label_names) == NUM_LABELS

    def test_first_pathway_label(self, label_names):
        assert label_names[0] == "Alkaloids"

    def test_pathway_range(self, label_names):
        assert len(label_names[PATHWAY_SLICE]) == 7

    def test_superclass_range(self, label_names):
        assert len(label_names[SUPERCLASS_SLICE]) == 70

    def test_class_range(self, label_names):
        assert len(label_names[CLASS_SLICE]) == NUM_LABELS - 77


class TestSingleMolecule:
    def test_returns_dict(self, single_probs, label_names):
        result = decode_predictions(single_probs, label_names)
        assert isinstance(result, dict)

    def test_keys(self, single_probs, label_names):
        result = decode_predictions(single_probs, label_names)
        assert set(result.keys()) == {"pathway", "superclass", "class"}

    def test_pathway_names(self, single_probs, label_names):
        result = decode_predictions(single_probs, label_names)
        assert result["pathway"] == [label_names[0]]

    def test_superclass_names(self, single_probs, label_names):
        result = decode_predictions(single_probs, label_names)
        assert result["superclass"] == [label_names[8]]

    def test_class_names(self, single_probs, label_names):
        result = decode_predictions(single_probs, label_names)
        assert result["class"] == [label_names[100]]

    def test_all_below_threshold(self, label_names):
        probs = np.zeros(NUM_LABELS)
        result = decode_predictions(probs, label_names)
        assert result == {"pathway": [], "superclass": [], "class": []}

    def test_all_above_threshold(self, label_names):
        probs = np.ones(NUM_LABELS)
        result = decode_predictions(probs, label_names)
        assert len(result["pathway"]) == 7
        assert len(result["superclass"]) == 70
        assert len(result["class"]) == NUM_LABELS - 77

    def test_all_above_threshold_names(self, label_names):
        probs = np.ones(NUM_LABELS)
        result = decode_predictions(probs, label_names)
        assert result["pathway"] == label_names[PATHWAY_SLICE]
        assert result["superclass"] == label_names[SUPERCLASS_SLICE]
        assert result["class"] == label_names[CLASS_SLICE]

    def test_boundary_at_threshold(self, label_names):
        probs = np.zeros(NUM_LABELS)
        probs[0] = 0.5    # exactly at threshold → included
        probs[1] = 0.499  # just below → excluded
        result = decode_predictions(probs, label_names, threshold=0.5)
        assert label_names[0] in result["pathway"]
        assert label_names[1] not in result["pathway"]

    def test_custom_threshold(self, single_probs, label_names):
        result = decode_predictions(single_probs, label_names, threshold=0.85)
        assert result["pathway"] == [label_names[0]]
        assert result["superclass"] == []
        assert result["class"] == []

    def test_list_input(self, label_names):
        probs = [0.0] * NUM_LABELS
        probs[2] = 0.9
        result = decode_predictions(probs, label_names)
        assert label_names[2] in result["pathway"]


class TestBatch:
    def test_returns_list(self, batch_probs, label_names):
        result = decode_predictions(batch_probs, label_names)
        assert isinstance(result, list)

    def test_list_length(self, batch_probs, label_names):
        result = decode_predictions(batch_probs, label_names)
        assert len(result) == 2

    def test_each_element_is_dict(self, batch_probs, label_names):
        result = decode_predictions(batch_probs, label_names)
        for item in result:
            assert isinstance(item, dict)
            assert set(item.keys()) == {"pathway", "superclass", "class"}

    def test_first_molecule(self, batch_probs, label_names):
        result = decode_predictions(batch_probs, label_names)
        assert result[0]["pathway"] == [label_names[0]]
        assert result[0]["superclass"] == [label_names[8]]
        assert result[0]["class"] == [label_names[100]]

    def test_second_molecule(self, batch_probs, label_names):
        result = decode_predictions(batch_probs, label_names)
        assert result[1]["pathway"] == [label_names[3]]
        assert result[1]["superclass"] == [label_names[10]]
        assert result[1]["class"] == [label_names[200]]

    def test_single_row_batch_matches_single(self, single_probs, label_names):
        single_result = decode_predictions(single_probs, label_names)
        batch_result  = decode_predictions(single_probs[np.newaxis, :], label_names)
        assert batch_result[0] == single_result
