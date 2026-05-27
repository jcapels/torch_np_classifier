from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torch-np-classifier")
except PackageNotFoundError:
    __version__ = "unknown"

from torch_np_classifier.models.np_classifier import NPClassifierDNN
from torch_np_classifier.models.lightning_module import NPClassifierLightning
from torch_np_classifier.models.ensemble import NPClassifierEnsemble
from torch_np_classifier.models.pipeline import NPClassifierPipeline
from torch_np_classifier.featurization.np_classifier_fp import (
    NPClassifierFeaturizer,
    featurize_smiles,
    fp_index_to_radius_bit,
    draw_bit_fragment,
)
from torch_np_classifier.data.dataset import NPClassifierDataset
from torch_np_classifier.data.datamodule import NPClassifierDataModule
from torch_np_classifier.callbacks.embedding_collector import EmbeddingCollector
from torch_np_classifier.utils.prediction_decoder import decode_predictions
from torch_np_classifier.explainability.shap_explainer import NPClassifierSHAP
from torch_np_classifier.explainability.ensemble_shap import NPClassifierEnsembleSHAP

__all__ = [
    "NPClassifierDNN",
    "NPClassifierLightning",
    "NPClassifierEnsemble",
    "NPClassifierPipeline",
    "NPClassifierFeaturizer",
    "featurize_smiles",
    "fp_index_to_radius_bit",
    "draw_bit_fragment",
    "NPClassifierDataset",
    "NPClassifierDataModule",
    "EmbeddingCollector",
    "decode_predictions",
    "NPClassifierSHAP",
    "NPClassifierEnsembleSHAP",
]
