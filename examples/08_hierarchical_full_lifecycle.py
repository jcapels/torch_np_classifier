"""Hierarchical full lifecycle on merged_dataset.csv.

Trains one model per hierarchical level (pathway / superclass / class) on the
full merged dataset (no validation split), saves a checkpoint for each level,
then loads all three and predicts on example molecules.

Label layout (730 total, columns after 'key' and 'SMILES'):
    pathway    ->  labels[:, :7]    (7 classes)
    superclass ->  labels[:, 7:77]  (70 classes)
    class      ->  labels[:, 77:]   (653 classes)

Run
---
    python examples/08_hierarchical_full_lifecycle.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import lightning
from torch.utils.data import DataLoader

from torch_np_classifier import (
    NPClassifierDataset,
    NPClassifierFeaturizer,
    NPClassifierLightning,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# merged_dataset.csv is the full combined NP-Classifier training file.
# It must live in the working directory from which you run the script.
# Expected columns: key, SMILES, <730 label columns>
TRAIN_CSV  = "merged_dataset.csv"
SMILES_COL = "SMILES"
LABEL_START = 2   # CSV columns to skip: 'key' and 'SMILES'

BATCH_SIZE  = 128
MAX_EPOCHS  = 50
MAX_EPOCHS = {"pathway": 4, "superclass": 8, "class": 15}
LR          = 1e-5
CKPT_DIR    = "checkpoints/hierarchical_lifecycle/"

# Slices into the 730-dim label array (0-indexed from first label column)
LEVELS: Dict[str, slice] = {
    "pathway":    slice(None, 7),
    "superclass": slice(7, 77),
    "class":      slice(77, None),
}

# Example molecules to predict
PREDICT_SMILES: List[str] = [
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",           # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",                 # aspirin
    "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",  # quercetin
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",  # morphine
]

# ---------------------------------------------------------------------------
# 1. Load CSV and featurize (once — shared across all three models)
# ---------------------------------------------------------------------------
print(f"Loading {TRAIN_CSV} …")
df = pd.read_csv(TRAIN_CSV)

label_cols  = df.columns[LABEL_START:].tolist()
labels_all  = df[label_cols].values.astype(np.float32)
smiles_list = df[SMILES_COL].tolist()

print(f"  Molecules : {len(smiles_list)}")
print(f"  Label cols: {len(label_cols)}")

print("Featurizing …")
featurizer = NPClassifierFeaturizer(radius=2, n_jobs=-1)
features   = featurizer.transform(smiles_list)   # (N, 6144)
print(f"  Feature matrix: {features.shape}")

# ---------------------------------------------------------------------------
# 2. Train one model per level (no validation — full training set)
# ---------------------------------------------------------------------------
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_paths: Dict[str, str] = {}

for level_name, label_sl in LEVELS.items():
    print(f"\n{'='*60}")
    level_labels = labels_all[:, label_sl]
    num_cats     = level_labels.shape[1]
    print(f"Level: {level_name!r}   num_categories={num_cats}")

    loader = DataLoader(
        NPClassifierDataset(features, level_labels),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    )

    model = NPClassifierLightning(num_categories=num_cats, lr=LR, scheduler=False)

    trainer = lightning.Trainer(
        max_epochs=MAX_EPOCHS[level_name],
        log_every_n_steps=10,
        enable_progress_bar=True,
        devices=1,
    )
    trainer.fit(model, loader)

    ckpt_path = os.path.join(CKPT_DIR, f"{level_name}_np_classifier.ckpt")
    trainer.save_checkpoint(ckpt_path)
    ckpt_paths[level_name] = ckpt_path
    print(f"  Checkpoint saved → {ckpt_path}")

# ---------------------------------------------------------------------------
# 3. Load all three checkpoints
# ---------------------------------------------------------------------------
print("\n\nLoading checkpoints …")
loaded_models: Dict[str, NPClassifierLightning] = {}
for level_name, path in ckpt_paths.items():
    m = NPClassifierLightning.load_from_checkpoint(path)
    m.eval()
    loaded_models[level_name] = m
    print(f"  {level_name}: loaded from {path}  "
          f"(num_categories={m.hparams.num_categories})")

# ---------------------------------------------------------------------------
# 4. Predict on example molecules
# ---------------------------------------------------------------------------
print("\nFeaturizing example molecules …")
predict_features = featurizer.transform(PREDICT_SMILES)
predict_dataset  = NPClassifierDataset(predict_features)
predict_loader   = DataLoader(predict_dataset, batch_size=32, num_workers=0)

pred_trainer = lightning.Trainer(enable_progress_bar=False, logger=False, devices=1)

all_probs: Dict[str, np.ndarray] = {}
for level_name, m in loaded_models.items():
    raw = pred_trainer.predict(m, predict_loader)
    all_probs[level_name] = torch.cat(raw, dim=0).numpy()

# ---------------------------------------------------------------------------
# 5. Print top predictions per molecule per level
# ---------------------------------------------------------------------------
print("\n=== Predictions (top-3 per level) ===")
THRESHOLD = 0.5

for mol_idx, smi in enumerate(PREDICT_SMILES):
    print(f"\n{smi[:50]}")
    for level_name, label_sl in LEVELS.items():
        probs_mol = all_probs[level_name][mol_idx]          # shape: (num_cats_for_level,)
        level_label_names = np.array(label_cols)[label_sl]  # label names for this level

        # Labels above threshold
        positive = [
            (level_label_names[i], probs_mol[i])
            for i in np.argsort(probs_mol)[::-1]
            if probs_mol[i] >= THRESHOLD
        ][:3]

        if positive:
            top_str = ", ".join(f"{n} ({p:.2f})" for n, p in positive)
        else:
            # Fall back to top-1 regardless of threshold
            top_idx  = int(np.argmax(probs_mol))
            top_str  = f"{level_label_names[top_idx]} ({probs_mol[top_idx]:.2f}) [below threshold]"

        print(f"  {level_name:<12}: {top_str}")

print("\nDone.")
