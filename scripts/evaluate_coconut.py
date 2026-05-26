"""Evaluate NPClassifierPipeline predictions against COCONUT ground truth.

Usage
-----
    python evaluate_coconut.py \\
        --pathway-ckpt    checkpoints/pathway/best.ckpt \\
        --superclass-ckpt checkpoints/superclass/best.ckpt \\
        --class-ckpt      checkpoints/class/best.ckpt \\
        [--csv coconut_csv-05-2026.csv] \\
        [--output results.csv]

Metrics reported (per level)
-----------------------------
- hit_rate    : fraction of molecules where the true label appears in the
                predicted set (multi-label recall at k).
- precision   : mean fraction of predicted labels that are correct.
- empty_rate  : fraction of molecules where the model predicted nothing.
- glycoside_acc: boolean accuracy for the is_glycoside flag.
"""

import argparse
import time
import tracemalloc
import warnings
from pathlib import Path

import pandas as pd

from torch_np_classifier import NPClassifierPipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate pipeline on COCONUT CSV")
parser.add_argument("--pathway-ckpt", required=True, help="pathway .ckpt path")
parser.add_argument("--superclass-ckpt", required=True, help="superclass .ckpt path")
parser.add_argument("--class-ckpt", required=True, help="class .ckpt path")
parser.add_argument(
    "--csv",
    default="coconut_csv-05-2026.csv",
    help="path to COCONUT CSV (default: coconut_csv-05-2026.csv)",
)
parser.add_argument(
    "--output", default=None, help="optional path to save per-molecule results CSV"
)
parser.add_argument("--pathway-threshold", type=float, default=0.5)
parser.add_argument("--superclass-threshold", type=float, default=0.3)
parser.add_argument("--class-threshold", type=float, default=0.1)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(f"Reading {args.csv} …")
df = pd.read_csv(args.csv, low_memory=False)

# Keep only rows with a SMILES string
df = df.dropna(subset=["canonical_smiles"]).copy()
print(f"  {len(df):,} rows with SMILES")

SMILES_COL = "canonical_smiles"
GT_PATHWAY = "np_classifier_pathway"
GT_SUPERCLASS = "np_classifier_superclass"
GT_CLASS = "np_classifier_class"
GT_GLYCOSIDE = "np_classifier_is_glycoside"

smiles_list = df[SMILES_COL].tolist()

# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
print("\nLoading pipeline from checkpoints …")
pipeline = NPClassifierPipeline.from_checkpoints(
    pathway_ckpt=args.pathway_ckpt,
    superclass_ckpt=args.superclass_ckpt,
    class_ckpt=args.class_ckpt,
    pathway_threshold=args.pathway_threshold,
    superclass_threshold=args.superclass_threshold,
    class_threshold=args.class_threshold,
)
print("Pipeline ready.\n")

# ---------------------------------------------------------------------------
# Predict whole dataset at once
# ---------------------------------------------------------------------------
n = len(smiles_list)

print(f"Predicting {n:,} molecules …")
tracemalloc.start()
pred_start = time.perf_counter()

all_results = pipeline.predict(smiles_list, check_glycoside=True)
if isinstance(all_results, dict):
    all_results = [all_results]

pred_elapsed = time.perf_counter() - pred_start
_, peak_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mb = peak_bytes / 1024**2

# ---------------------------------------------------------------------------
# Build comparison dataframe
# ---------------------------------------------------------------------------
pred_pathway = [r["pathway"] for r in all_results]
pred_superclass = [r["superclass"] for r in all_results]
pred_class = [r["class"] for r in all_results]
pred_glycoside = [r["isglycoside"] for r in all_results]

df = df.reset_index(drop=True)
df["pred_pathway"] = pred_pathway
df["pred_superclass"] = pred_superclass
df["pred_class"] = pred_class
df["pred_glycoside"] = pred_glycoside

# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def evaluate_level(gt_series: pd.Series, pred_series: pd.Series, level: str):
    """Compare single-label ground truth against multi-label predictions.

    Rows where GT is NaN are treated as "no prediction" from the ground truth.
    A molecule where GT is NaN and the model also predicts nothing counts as a
    correct hit; a model prediction when GT is NaN is counted as a false positive.
    """
    import numpy as np

    total = len(gt_series)
    if total == 0:
        print(f"\n[{level}] No data — skipped.")
        return {}

    hits = 0
    precisions = []
    for gt_val, pred_val in zip(gt_series, pred_series):
        gt_empty = pd.isna(gt_val)
        pred_empty = not pred_val

        if gt_empty and pred_empty:
            hits += 1  # correctly predicted no label
            precisions.append(1.0)
        elif gt_empty and not pred_empty:
            # false-positive predictions when GT has no label
            precisions.append(0.0)
        elif not gt_empty and pred_empty:
            # missed prediction — precision undefined for empty set
            precisions.append(float("nan"))
        else:
            if gt_val in pred_val:
                hits += 1
            precisions.append(1.0 if gt_val in pred_val else 0.0)

    prec_arr = [p for p in precisions if not pd.isna(p)]
    mean_precision = float(np.mean(prec_arr)) if prec_arr else float("nan")

    empty = sum(not p for p in pred_series)
    no_gt = int(gt_series.isna().sum())

    metrics = {
        "level": level,
        "total": total,
        "hit_rate": hits / total,
        "mean_precision": mean_precision,
        "empty_rate": empty / total,
    }

    print(f"\n[{level}]  (n={total:,}, no_gt={no_gt:,})")
    print(
        f"  hit_rate       : {metrics['hit_rate']:.4f}  "
        f"({hits:,}/{total:,} — true label in predicted set, or both empty)"
    )
    print(
        f"  mean_precision : {metrics['mean_precision']:.4f}  "
        f"(avg fraction of predicted labels that are correct)"
    )
    print(
        f"  empty_rate     : {metrics['empty_rate']:.4f}  "
        f"({empty:,} molecules with no prediction)"
    )
    return metrics


# ---------------------------------------------------------------------------
# Glycoside accuracy
# ---------------------------------------------------------------------------
def evaluate_glycoside(gt_series: pd.Series, pred_list: list):
    valid = gt_series.notna()
    gt = gt_series[valid].astype(bool)
    pred = pd.Series(pred_list)[valid]
    total = len(gt)
    correct = (gt == pred).sum()
    acc = correct / total if total else float("nan")
    print(f"\n[is_glycoside]  (n={total:,})")
    print(f"  accuracy : {acc:.4f}  ({correct:,}/{total:,})")
    return {"level": "is_glycoside", "total": total, "accuracy": acc}


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PREDICTION PERFORMANCE")
print("=" * 60)
hours, remainder = divmod(pred_elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
time_str = f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"
print(f"  total time   : {time_str}  ({pred_elapsed:.2f}s)")
print(f"  throughput   : {n / pred_elapsed:,.1f} molecules/s")
print(f"  peak memory  : {peak_mb:.1f} MB  (prediction loop only)")

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

m_pathway = evaluate_level(df[GT_PATHWAY], df["pred_pathway"], "pathway")
m_superclass = evaluate_level(df[GT_SUPERCLASS], df["pred_superclass"], "superclass")
m_class = evaluate_level(df[GT_CLASS], df["pred_class"], "class")
m_glycoside = evaluate_glycoside(df[GT_GLYCOSIDE], pred_glycoside)

print("\n" + "=" * 60)

# ---------------------------------------------------------------------------
# Optional: save per-molecule results
# ---------------------------------------------------------------------------
if args.output:
    out_cols = [
        SMILES_COL,
        GT_PATHWAY,
        "pred_pathway",
        GT_SUPERCLASS,
        "pred_superclass",
        GT_CLASS,
        "pred_class",
        GT_GLYCOSIDE,
        "pred_glycoside",
    ]
    out_df = df[out_cols].copy()
    # Store list columns as pipe-separated strings for readability
    for col in ("pred_pathway", "pred_superclass", "pred_class"):
        out_df[col] = out_df[col].apply(lambda x: "|".join(x) if x else "")
    out_df.to_csv(args.output, index=False)
    print(f"\nPer-molecule results saved to: {args.output}")
