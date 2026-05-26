"""Train one model per hierarchical level (pathway / superclass / class).

Label layout (730 total, columns after 'key' and 'SMILES'):
    pathway    ->  labels[:, :7]    (7 classes)
    superclass ->  labels[:, 7:77]  (70 classes)
    class      ->  labels[:, 77:]   (653 classes)

Each model is trained with early stopping and the best checkpoint is saved.
At the end, all three are evaluated on the test set and per-level F1 scores
are reported (comparable to 05_train_and_evaluate.py).

Run
---
    cd examples/
    python 07_hierarchical_train_and_evaluate.py
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from torch_np_classifier import (
    NPClassifierDataset,
    NPClassifierFeaturizer,
    NPClassifierLightning,
)
from torch_np_classifier.utils.metrics import map_mean_sd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_CSV = "train_dataset.csv"
VAL_CSV = "validation_dataset.csv"
TEST_CSV = "test_dataset.csv"
SMILES_COL = "SMILES"
LABEL_START = 2  # CSV columns to skip: 'key' and 'SMILES'

BATCH_SIZE = 128
MAX_EPOCHS = 150
LR = 1e-5
CKPT_DIR = "checkpoints/hierarchical/"
THRESHOLD = 0.5

# Slices into the 730-dim label array (0-indexed from first label column)
LEVELS: Dict[str, Tuple[slice, int]] = {
    "pathway": (slice(None, 7), 7),
    "superclass": (slice(7, 77), 70),
    "class": (slice(77, None), None),  # None → inferred from data
}

# ---------------------------------------------------------------------------
# 1. Load CSVs and featurize (once — shared across all three models)
# ---------------------------------------------------------------------------
print("Loading CSVs …")
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

label_cols = train_df.columns[LABEL_START:].tolist()
print(f"  Total label columns: {len(label_cols)}")

print("Featurizing SMILES …")
featurizer = NPClassifierFeaturizer(radius=2, n_jobs=-1)
train_features = featurizer.transform(train_df[SMILES_COL].tolist())
val_features = featurizer.transform(val_df[SMILES_COL].tolist())
test_features = featurizer.transform(test_df[SMILES_COL].tolist())
print(
    f"  train={train_features.shape}, val={val_features.shape}, test={test_features.shape}"
)

train_labels_all = train_df[label_cols].values.astype(np.float32)
val_labels_all = val_df[label_cols].values.astype(np.float32)
test_labels_all = test_df[label_cols].values.astype(np.float32)

# ---------------------------------------------------------------------------
# 2. Train one model per level
# ---------------------------------------------------------------------------
best_ckpt_paths: Dict[str, str] = {}

for level_name, (label_sl, num_cats_hint) in LEVELS.items():
    print(f"\n{'=' * 60}")
    train_labels = train_labels_all[:, label_sl]
    val_labels = val_labels_all[:, label_sl]
    num_cats = train_labels.shape[1]
    print(f"Level: {level_name!r}   num_categories={num_cats}")

    train_loader = DataLoader(
        NPClassifierDataset(train_features, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        NPClassifierDataset(val_features, val_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    model = NPClassifierLightning(num_categories=num_cats, lr=LR, scheduler=True)

    ckpt_dir = os.path.join(CKPT_DIR, level_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{level_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ckpt_cb,
    ]

    trainer = lightning.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        log_every_n_steps=10,
        devices=1,
    )
    trainer.fit(model, train_loader, val_loader)
    best_ckpt_paths[level_name] = ckpt_cb.best_model_path
    print(f"  Best checkpoint: {ckpt_cb.best_model_path}")

# ---------------------------------------------------------------------------
# 3. Evaluate each model on the test set
# ---------------------------------------------------------------------------
print("\n\n=== Test Evaluation ===")


def report_metrics(
    name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
) -> None:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    map_mean, map_sd = map_mean_sd(y_true, y_prob)
    print(
        f"  {name:<14}  macro F1 = {macro_f1:.4f}   micro F1 = {micro_f1:.4f}"
        f"   MAP = {map_mean:.4f} ± {map_sd:.4f}"
    )


pred_trainer = lightning.Trainer(enable_progress_bar=False, devices=1)
test_loader = DataLoader(
    NPClassifierDataset(test_features),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

for level_name, (label_sl, _) in LEVELS.items():
    ckpt_path = best_ckpt_paths[level_name]
    if not ckpt_path:
        print(f"  {level_name}: no checkpoint found, skipping.")
        continue

    level_model = NPClassifierLightning.load_from_checkpoint(ckpt_path)
    level_model.eval()

    raw = pred_trainer.predict(level_model, test_loader)
    probs = torch.cat(raw, dim=0).numpy()
    preds = (probs >= THRESHOLD).astype(np.int32)

    true_labels = test_labels_all[:, label_sl]
    report_metrics(level_name, true_labels, probs, preds)

print("\nDone.")
