"""Train NPClassifier from CSV files using NPClassifierDataModule.

Expected CSV format (any split):

    SMILES,class_0,class_1,...,class_N
    Cn1cnc2c1c(=O)n(c(=O)n2C)C,1,0,...,0
    ...

Run
---
    python examples/01_train_from_csv.py
"""

import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from torch_np_classifier import NPClassifierDataModule, NPClassifierLightning

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dm = NPClassifierDataModule(
    train_csv="data/train.csv",
    val_csv="data/val.csv",
    smiles_col="SMILES",
    label_slice=slice(1, None),   # all columns after SMILES are labels
    batch_size=128,
    num_workers=4,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = NPClassifierLightning(
    num_categories=730,
    lr=1e-5,
    scheduler=True,
)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ModelCheckpoint(
        dirpath="checkpoints/",
        filename="np_classifier-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
    ),
]

trainer = lightning.Trainer(
    max_epochs=50,
    callbacks=callbacks,
    log_every_n_steps=10,
)

trainer.fit(model, datamodule=dm)
print("Best checkpoint:", callbacks[1].best_model_path)
