"""SHAP-based explainability: load → predict → explain → draw bits.

Produces one ``explanation_<name>.png`` and one ``bits_<name>.png`` per
molecule, saved to ``--outdir``.

Usage
-----
    python examples/06_shap_explainability.py \\
        --ckpt np_classifier.ckpt \\
        --background-csv examples/train_dataset.csv \\
        --k 6 --outdir explanations/
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import lightning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torch_np_classifier import (
    NPClassifierFeaturizer,
    NPClassifierDataset,
    NPClassifierLightning,
    NPClassifierSHAP,
    decode_predictions,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="SHAP explainability for NP Classifier")
parser.add_argument("--ckpt", required=True, help="Path to .ckpt checkpoint")
parser.add_argument(
    "--background-csv", default=None,
    help="CSV with SMILES column to build SHAP background (recommended)",
)
parser.add_argument(
    "--smiles-col", default="SMILES",
    help="Name of the SMILES column in the background CSV",
)
parser.add_argument("--n-background", type=int, default=100)
parser.add_argument("--k", type=int, default=6, help="Top-k bits to highlight")
parser.add_argument("--outdir", default=".", help="Output directory for PNG files")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Test molecules — well-known natural products
# ---------------------------------------------------------------------------
MOLECULES = {
    "caffeine":  "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "morphine":  "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
    "aspirin":   "CC(=O)Oc1ccccc1C(=O)O",
    "cholesterol": "[C@@H]1(CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)[C@H](CCCC(C)C)C)C",
}

# ---------------------------------------------------------------------------
# 1. Featurize
# ---------------------------------------------------------------------------
print("Featurizing …")
featurizer = NPClassifierFeaturizer(radius=2, n_jobs=-1)
smiles_list = list(MOLECULES.values())
names_list  = list(MOLECULES.keys())
features    = featurizer.transform(smiles_list)

# ---------------------------------------------------------------------------
# 2. Load model & predict
# ---------------------------------------------------------------------------
print(f"Loading model from {args.ckpt} …")
model = NPClassifierLightning.load_from_checkpoint(args.ckpt)
model.eval()

dataset = NPClassifierDataset(features)
loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
trainer = lightning.Trainer(enable_progress_bar=False, logger=False)
raw     = trainer.predict(model, loader)
probs   = torch.cat(raw, dim=0).numpy()  # (N, num_categories)

# ---------------------------------------------------------------------------
# 3. Decode predictions (requires label names from the CSV)
# ---------------------------------------------------------------------------
label_names = None
if args.background_csv:
    df_bg = pd.read_csv(args.background_csv)
    num_cols = [c for c in df_bg.columns if c not in (args.smiles_col, "key")]
    numeric_cols = df_bg[num_cols].select_dtypes(include=[np.number]).columns.tolist()
    label_names = np.array(numeric_cols)

if label_names is not None:
    decoded = decode_predictions(probs, label_names)
    print("\n=== Predictions ===")
    for name, dec in zip(names_list, decoded):
        print(f"\n{name}:")
        for level, preds in dec.items():
            if preds:
                print(f"  {level}: {', '.join(preds[:3])}")
else:
    print("\n=== Top-5 class indices ===")
    for name, prob in zip(names_list, probs):
        top5 = np.argsort(prob)[::-1][:5]
        print(f"  {name}: {top5.tolist()} (p={prob[top5].round(3).tolist()})")

# ---------------------------------------------------------------------------
# 4. Build SHAP background fingerprints
# ---------------------------------------------------------------------------
if args.background_csv:
    print(f"\nSampling {args.n_background} background molecules from {args.background_csv} …")
    df_bg = pd.read_csv(args.background_csv)
    bg_smiles_all = df_bg[args.smiles_col].dropna().tolist()
    rng = np.random.default_rng(42)
    chosen = rng.choice(
        len(bg_smiles_all),
        min(args.n_background, len(bg_smiles_all)),
        replace=False,
    )
    bg_features = featurizer.transform([bg_smiles_all[i] for i in chosen])
else:
    print("\nNo --background-csv provided — using random background (less accurate).")
    rng = np.random.default_rng(42)
    bg_features = rng.random(
        (args.n_background, featurizer.feature_dim)
    ).astype(np.float32)

# ---------------------------------------------------------------------------
# 5. SHAP explanation per molecule
# ---------------------------------------------------------------------------
os.makedirs(args.outdir, exist_ok=True)

print("\n=== Generating SHAP explanations ===")
for mol_name, smi, prob in zip(names_list, smiles_list, probs):
    top_class = int(np.argmax(prob))
    class_label = (
        str(label_names[top_class]) if label_names is not None else f"class_{top_class}"
    )
    print(
        f"\n{mol_name}: top class = {class_label!r}  "
        f"(idx={top_class}, p={prob[top_class]:.3f})"
    )

    explainer = NPClassifierSHAP(model, bg_features, class_indices=[top_class])
    sv        = explainer.explain_smiles(smi, featurizer)  # (1, 6144, 1)
    sv_1d     = sv[0, :, 0]

    n_top = len(explainer.top_feature_indices(sv_1d, k=args.k))
    print(f"  {n_top} active positive SHAP bits in top-{args.k}")

    # Full explanation figure (molecule + bar chart + fragments)
    fig = explainer.explanation_figure(
        smi, sv_1d, featurizer,
        class_name=class_label, k=args.k,
    )
    out_full = os.path.join(args.outdir, f"explanation_{mol_name}.png")
    fig.savefig(out_full, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_full}")

    # Fragment-only grid
    fig_frags = explainer.draw_bit_fragments(
        smi, sv_1d, featurizer, k=args.k
    )
    out_frags = os.path.join(args.outdir, f"bits_{mol_name}.png")
    fig_frags.savefig(out_frags, dpi=150, bbox_inches="tight")
    plt.close(fig_frags)
    print(f"  Saved {out_frags}")

print("\nDone.")
