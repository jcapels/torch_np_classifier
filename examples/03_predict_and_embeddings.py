"""Run inference and collect molecular embeddings from a trained checkpoint.

Produces:
  - predictions : (N, num_categories) float32 array of sigmoid probabilities
  - embeddings  : (N, 1536) float32 array of penultimate-layer representations

Run
---
    python examples/03_predict_and_embeddings.py --ckpt checkpoints/np_classifier.ckpt
"""

import argparse

import numpy as np
import lightning
from torch.utils.data import DataLoader

from torch_np_classifier import (
    NPClassifierFeaturizer,
    NPClassifierDataset,
    NPClassifierLightning,
    EmbeddingCollector,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Path to .ckpt file")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Molecules to predict
# ---------------------------------------------------------------------------
smiles_list = [
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # pyrene
]

featurizer = NPClassifierFeaturizer(radius=2)
features = featurizer.transform(smiles_list)  # (N, 6144)

# No labels needed for prediction
dataset = NPClassifierDataset(features)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ---------------------------------------------------------------------------
# Load model & predict
# ---------------------------------------------------------------------------
model = NPClassifierLightning.load_from_checkpoint(args.ckpt)

collector = EmbeddingCollector()
trainer = lightning.Trainer(callbacks=[collector], devices=1)
trainer.predict(model, loader)

predictions: np.ndarray = collector.predictions  # (N, num_categories)
embeddings: np.ndarray = collector.embeddings  # (N, 1536)

print("predictions shape:", predictions.shape)
print("embeddings  shape:", embeddings.shape)

# Save to disk
np.save("predictions.npy", predictions)
np.save("embeddings.npy", embeddings)
print("Saved predictions.npy and embeddings.npy")
