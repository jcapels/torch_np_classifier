"""Full lifecycle: featurize → train (no validation) → save → load → predict.

Expects a CSV file at the path set by TRAIN_CSV with at least a SMILES column
and one or more label columns.  Example format:

    SMILES,class_0,class_1,...,class_N
    Cn1cnc2c1c(=O)n(c(=O)n2C)C,1,0,...,0
    ...

Run
---
    python examples/04_full_lifecycle.py
"""

import torch
import numpy as np
import pandas as pd
import lightning
from torch.utils.data import DataLoader

from torch_np_classifier import (
    NPClassifierFeaturizer,
    NPClassifierDataset,
    NPClassifierLightning,
    EmbeddingCollector,
)

# ---------------------------------------------------------------------------
# Config — adjust these to match your data
# ---------------------------------------------------------------------------
TRAIN_CSV     = "merged_dataset.csv"
SMILES_COL    = "SMILES"
LABEL_SLICE   = slice(1, None)   # all columns after SMILES are labels
NUM_CATEGORIES = 730
BATCH_SIZE    = 128
MAX_EPOCHS    = 150
LR            = 1e-5
CKPT_PATH     = "np_classifier.ckpt"

# ---------------------------------------------------------------------------
# 1. Load CSV & featurize
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_csv(TRAIN_CSV)

smiles_list  = df[SMILES_COL].tolist()
label_cols   = df.columns[LABEL_SLICE].tolist()
labels       = df[label_cols].values.astype(np.float32)

print(f"  molecules : {len(smiles_list)}")
print(f"  categories: {labels.shape[1]}")

print("Featurizing SMILES …")
featurizer = NPClassifierFeaturizer(radius=2, n_jobs=-1)
features   = featurizer.transform(smiles_list)   # (N, 6144)
print(f"  feature matrix: {features.shape}")

# ---------------------------------------------------------------------------
# 2. Dataset & DataLoader (no validation split)
# ---------------------------------------------------------------------------
dataset      = NPClassifierDataset(features, labels)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
print("\nTraining …")
model = NPClassifierLightning(
    num_categories=NUM_CATEGORIES,
    lr=LR,
    scheduler=False,
)

trainer = lightning.Trainer(
    max_epochs=MAX_EPOCHS,
    log_every_n_steps=10,
    enable_progress_bar=True,
)
trainer.fit(model, train_loader)   # no val_loader → no validation loop

# ---------------------------------------------------------------------------
# 4. Save checkpoint
# ---------------------------------------------------------------------------
trainer.save_checkpoint(CKPT_PATH)
print(f"\nCheckpoint saved → {CKPT_PATH}")

# ---------------------------------------------------------------------------
# 5. Load checkpoint
# ---------------------------------------------------------------------------
print("\nLoading checkpoint …")
loaded_model = NPClassifierLightning.load_from_checkpoint(CKPT_PATH)
loaded_model.eval()
print("  hparams:", dict(loaded_model.hparams))

# ---------------------------------------------------------------------------
# 6. Predict (sigmoid probabilities)
# ---------------------------------------------------------------------------
print("\nPredicting …")
predict_smiles = [
    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",   # caffeine
    "O=C(O)/C=C/c1ccccc1",           # cinnamic acid
]
predict_features = featurizer.transform(predict_smiles)
predict_ds       = NPClassifierDataset(predict_features)   # no labels needed
predict_loader   = DataLoader(predict_ds, batch_size=32, num_workers=0)

predict_trainer = lightning.Trainer(enable_progress_bar=False)
raw_outputs     = predict_trainer.predict(loaded_model, predict_loader)

predictions = torch.cat(raw_outputs, dim=0).numpy()   # (N, NUM_CATEGORIES)
print(f"Prediction shape: {predictions.shape}")

for smi, prob in zip(predict_smiles, predictions):
    top = np.argsort(prob)[::-1][:5]
    print(f"  {smi[:45]:<45}  top-5 classes: {top.tolist()}  probs: {prob[top].round(3).tolist()}")

# ---------------------------------------------------------------------------
# 7. Predict with embeddings
# ---------------------------------------------------------------------------
print("\nCollecting embeddings …")
collector     = EmbeddingCollector()
embed_trainer = lightning.Trainer(callbacks=[collector], enable_progress_bar=False)
embed_trainer.predict(loaded_model, predict_loader)

embeddings   = collector.embeddings    # (N, 1536)
predictions2 = collector.predictions   # (N, NUM_CATEGORIES)
print(f"  embeddings  shape: {embeddings.shape}")
print(f"  predictions shape: {predictions2.shape}")
