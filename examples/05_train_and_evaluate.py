"""Train, validate, and evaluate NPClassifier; report per-level F1 scores.

Label layout (730 total):
    Pathway    ->  [:7]
    Superclass ->  [7:77]
    Class      ->  [77:]

Run
---
    python examples/05_train_and_evaluate.py
"""

import numpy as np
import pandas as pd
import torch
import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from torch_np_classifier import (
    NPClassifierDataModule,
    NPClassifierDataset,
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
LABEL_SLICE = slice(2, None)  # skip 'key' and 'SMILES'

BATCH_SIZE = 128
MAX_EPOCHS = 150
LR = 1e-5
CKPT_DIR = "checkpoints/"
THRESHOLD = 0.5

PATHWAY_SLICE = slice(None, 7)
SUPERCLASS_SLICE = slice(7, 77)
CLASS_SLICE = slice(77, None)

# ---------------------------------------------------------------------------
# Data module (train + val)
# ---------------------------------------------------------------------------
dm = NPClassifierDataModule(
    train_csv=TRAIN_CSV,
    val_csv=VAL_CSV,
    smiles_col=SMILES_COL,
    label_slice=LABEL_SLICE,
    batch_size=BATCH_SIZE,
    num_workers=4,
)
dm.setup("fit")
num_categories = dm._train_dataset.labels.shape[1]
print(f"Categories: {num_categories}  (pathway=7, superclass=70, class={num_categories - 77})")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = NPClassifierLightning(
    num_categories=num_categories,
    lr=LR,
    scheduler=True,
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
ckpt_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="np_classifier-{epoch:02d}-{val_loss:.4f}",
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

trainer.fit(model, datamodule=dm)
print(f"\nBest checkpoint: {ckpt_cb.best_model_path}")

# ---------------------------------------------------------------------------
# Load best model
# ---------------------------------------------------------------------------
best_model = NPClassifierLightning.load_from_checkpoint(ckpt_cb.best_model_path)
best_model.eval()

# ---------------------------------------------------------------------------
# Test: featurize and predict
# ---------------------------------------------------------------------------
print("\nLoading and featurizing test set …")
test_df = pd.read_csv(TEST_CSV)
smiles_list = test_df[SMILES_COL].tolist()
label_cols = test_df.columns[LABEL_SLICE].tolist()
true_labels = test_df[label_cols].values.astype(np.int32)

features = dm.featurizer.transform(smiles_list)
test_loader = DataLoader(
    NPClassifierDataset(features),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

pred_trainer = lightning.Trainer(enable_progress_bar=False, devices=1)
raw_outputs = pred_trainer.predict(best_model, test_loader)
probs = torch.cat(raw_outputs, dim=0).numpy()
preds = (probs >= THRESHOLD).astype(np.int32)


# ---------------------------------------------------------------------------
# Metrics per level
# ---------------------------------------------------------------------------
def report_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> None:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    map_mean, map_sd = map_mean_sd(y_true, y_prob)
    print(f"  {name:<25}  macro F1 = {macro_f1:.4f}   micro F1 = {micro_f1:.4f}   MAP = {map_mean:.4f} ± {map_sd:.4f}")


print("\n=== Test Scores ===")
report_metrics(
    "Pathway    [:7]",
    true_labels[:, PATHWAY_SLICE],
    probs[:, PATHWAY_SLICE],
    preds[:, PATHWAY_SLICE],
)
report_metrics(
    "Superclass [7:77]",
    true_labels[:, SUPERCLASS_SLICE],
    probs[:, SUPERCLASS_SLICE],
    preds[:, SUPERCLASS_SLICE],
)
report_metrics(
    "Class      [77:]",
    true_labels[:, CLASS_SLICE],
    probs[:, CLASS_SLICE],
    preds[:, CLASS_SLICE],
)
