"""Train NPClassifier directly from SMILES lists and numpy arrays.

Useful when data is already in memory or comes from a non-CSV source.

Run
---
    python examples/02_train_from_arrays.py
"""

import numpy as np
import lightning
from torch.utils.data import DataLoader, random_split

from torch_np_classifier import (
    NPClassifierFeaturizer,
    NPClassifierDataset,
    NPClassifierLightning,
)

# ---------------------------------------------------------------------------
# Featurize
# ---------------------------------------------------------------------------
smiles_list = [
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",  # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # pyrene
    # add your molecules here …
]

# Dummy multi-label targets — replace with your real labels
NUM_CATEGORIES = 730
rng = np.random.default_rng(42)
labels = rng.integers(0, 2, size=(len(smiles_list), NUM_CATEGORIES)).astype(np.float32)

featurizer = NPClassifierFeaturizer(radius=2, n_jobs=-1)
features = featurizer.transform(smiles_list)  # (N, 6144)

# ---------------------------------------------------------------------------
# Dataset / DataLoaders
# ---------------------------------------------------------------------------
dataset = NPClassifierDataset(features, labels)

n_val = max(1, int(0.2 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# ---------------------------------------------------------------------------
# Model & Trainer
# ---------------------------------------------------------------------------
model = NPClassifierLightning(num_categories=NUM_CATEGORIES, lr=1e-5)

trainer = lightning.Trainer(max_epochs=10, log_every_n_steps=1)
trainer.fit(model, train_loader, val_loader)
