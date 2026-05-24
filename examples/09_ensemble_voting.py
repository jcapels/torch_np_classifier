"""Ensemble voting with three hierarchical NP-Classifier models.

Loads the three checkpoints produced by ``07_`` or ``08_``, wraps them in
``NPClassifierEnsemble``, and runs ontology-based voting predictions.

Run
---
    python examples/09_ensemble_voting.py \\
        --pathway-ckpt    checkpoints/hierarchical/pathway/best.ckpt \\
        --superclass-ckpt checkpoints/hierarchical/superclass/best.ckpt \\
        --class-ckpt      checkpoints/hierarchical/class/best.ckpt
"""

import argparse
import warnings

from torch_np_classifier import NPClassifierEnsemble

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ensemble voting prediction")
parser.add_argument("--pathway-ckpt",    required=True)
parser.add_argument("--superclass-ckpt", required=True)
parser.add_argument("--class-ckpt",      required=True)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build ensemble (uses bundled label_names.pkl + index_v1.json)
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")   # suppress ontology mismatch warnings

print("Loading models …")
ensemble = NPClassifierEnsemble.from_checkpoints(
    pathway_ckpt    = args.pathway_ckpt,
    superclass_ckpt = args.superclass_ckpt,
    class_ckpt      = args.class_ckpt,
    # Thresholds match the original NP-Classifier:
    pathway_threshold    = 0.5,
    superclass_threshold = 0.3,
    class_threshold      = 0.1,
)
print("Ensemble ready.\n")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
MOLECULES = {
    "caffeine":    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "quercetin":   "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "morphine":    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@@H]3[C@@H]1C5",
    "cholesterol": "[C@@H]1(CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)"
                   "[C@H](CCCC(C)C)C)C",
    "sucrose":     "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
                   "[C@@H](O)[C@@H]1O",
}

results = ensemble.predict(list(MOLECULES.values()), check_glycoside=True)

print("=== Voting Predictions ===\n")
for name, result in zip(MOLECULES.keys(), results):
    print(f"{name}  (glycoside={result['isglycoside']})")
    print(f"  pathway    : {result['pathway']}")
    print(f"  superclass : {result['superclass']}")
    print(f"  class      : {result['class']}")
    print()
