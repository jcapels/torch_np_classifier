"""NPClassifierEnsemble — three-model wrapper with ontology-based voting.

The voting algorithm is a faithful port of the original NP-Classifier
``vote_classification`` / ``classifier`` logic (Classyfire hierarchy).

Typical workflow
----------------
>>> from torch_np_classifier import NPClassifierEnsemble, NPClassifierLightning
>>> pathway_model    = NPClassifierLightning.load_from_checkpoint("pathway.ckpt")
>>> superclass_model = NPClassifierLightning.load_from_checkpoint("superclass.ckpt")
>>> class_model      = NPClassifierLightning.load_from_checkpoint("class.ckpt")
>>>
>>> ensemble = NPClassifierEnsemble.from_label_names_file(
...     pathway_model, superclass_model, class_model
... )
>>> result = ensemble.predict("Cn1cnc2c1c(=O)n(c(=O)n2C)C")
>>> result["pathway"]
['Alkaloids']
"""

from __future__ import annotations

import itertools
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_np_classifier.featurization.np_classifier_fp import NPClassifierFeaturizer
from torch_np_classifier.models.lightning_module import NPClassifierLightning

# ── bundled data paths ──────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"
_ONTOLOGY_PATH = _DATA_DIR / "index_v1.json"
_LABEL_PKL = _DATA_DIR / "label_names.pkl"

# Label layout within the 730-dim label vector
_N_PATHWAY = 7
_N_SUPERCLASS = 70  # indices  7:77
_N_CLASS = 653  # indices 77:730

# Known typos in training-CSV label names vs. ontology keys
_DEFAULT_ALIASES: Dict[str, str] = {
    "Alkylresorsinols": "Alkylresorcinols",
    "Spingolipids": "Sphingolipids",
}

# Glycoside SMARTS from the original NP-Classifier
_HEXA_PYRANOSE = "[O]C1C([O])C([O])C(C[O])OC1[*]"
_PENTA_FURANOSE = "[O]CC1OC([*])C([O])C1[O]"


# ── helpers ─────────────────────────────────────────────────────────────────


def _load_ontology(source: Union[str, Path, dict, None]) -> dict:
    if source is None:
        source = _ONTOLOGY_PATH
    if isinstance(source, dict):
        return source
    with open(source) as f:
        return json.load(f)


def _is_glycoside(smiles: str) -> bool:
    """Return True if *smiles* contains a pyranose or furanose sugar unit."""
    try:
        from rdkit import Chem

        hexa = Chem.MolFromSmarts(_HEXA_PYRANOSE)
        penta = Chem.MolFromSmarts(_PENTA_FURANOSE)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return mol.HasSubstructMatch(hexa) or mol.HasSubstructMatch(penta)
    except Exception:
        return False


# ── main class ──────────────────────────────────────────────────────────────


class NPClassifierEnsemble:
    """Wrapper around three hierarchical NPClassifier models with voting.

    Each of the three models covers one level of the NP-Classifier hierarchy:

    - **pathway model**    — 7 outputs
    - **superclass model** — 70 outputs
    - **class model**      — 653 outputs

    At prediction time the three models vote on the pathway, and the result is
    refined level by level using the ontology hierarchy, mirroring the
    original NP-Classifier ``vote_classification`` logic.

    Parameters
    ----------
    pathway_model:
        ``NPClassifierLightning`` trained on pathway labels (7 outputs).
    superclass_model:
        ``NPClassifierLightning`` trained on superclass labels (70 outputs).
    class_model:
        ``NPClassifierLightning`` trained on class labels (653 outputs).
    pathway_labels:
        Ordered list of pathway label names (length 7).
    superclass_labels:
        Ordered list of superclass label names (length 70).
    class_labels:
        Ordered list of class label names (length 653).
    ontology:
        Path to ``index_v1.json``, a pre-loaded dict, or ``None`` to use the
        bundled copy.
    pathway_threshold:
        Probability threshold for pathway predictions (default 0.6).
    superclass_threshold:
        Probability threshold for superclass predictions (default 0.6).
    class_threshold:
        Probability threshold for class predictions (default 0.6).
    featurizer:
        ``NPClassifierFeaturizer`` used by :meth:`predict`.  Defaults to
        ``NPClassifierFeaturizer()``.
    name_aliases:
        Extra ``{csv_name: ontology_name}`` mappings to handle label-name
        discrepancies between the training CSV and the ontology.  The
        built-in aliases (``Alkylresorsinols`` → ``Alkylresorcinols``,
        ``Spingolipids`` → ``Sphingolipids``) are always applied.
    """

    def __init__(
        self,
        pathway_model: NPClassifierLightning,
        superclass_model: NPClassifierLightning,
        class_model: NPClassifierLightning,
        pathway_labels: List[str],
        superclass_labels: List[str],
        class_labels: List[str],
        ontology: Union[str, Path, dict, None] = None,
        pathway_threshold: float = 0.6,
        superclass_threshold: float = 0.6,
        class_threshold: float = 0.6,
        featurizer: Optional[NPClassifierFeaturizer] = None,
        name_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        for m in (pathway_model, superclass_model, class_model):
            m.eval()

        self.pathway_model = pathway_model
        self.superclass_model = superclass_model
        self.class_model = class_model

        self._pathway_labels = list(pathway_labels)
        self._superclass_labels = list(superclass_labels)
        self._class_labels = list(class_labels)

        self.pathway_threshold = pathway_threshold
        self.superclass_threshold = superclass_threshold
        self.class_threshold = class_threshold

        self.featurizer = (
            featurizer if featurizer is not None else NPClassifierFeaturizer()
        )

        # Merge aliases
        aliases: Dict[str, str] = dict(_DEFAULT_ALIASES)
        if name_aliases:
            aliases.update(name_aliases)

        self._onto = _load_ontology(ontology)
        self._build_hierarchy(aliases)

        self._trainer = lightning.Trainer(
            enable_progress_bar=False,
            logger=False,
            accelerator="auto",
            devices=1,
        )

    # ── construction helpers ────────────────────────────────────────────────

    @classmethod
    def from_label_names_file(
        cls,
        pathway_model: NPClassifierLightning,
        superclass_model: NPClassifierLightning,
        class_model: NPClassifierLightning,
        label_names_pkl: Union[str, Path, None] = None,
        **kwargs,
    ) -> "NPClassifierEnsemble":
        """Construct from the bundled ``label_names.pkl`` (or a custom one).

        The pickle file must contain a flat list of 730 label names in the
        canonical order: 7 pathway + 70 superclass + 653 class.

        Parameters
        ----------
        label_names_pkl:
            Path to the pickle file.  ``None`` uses the bundled copy.
        **kwargs:
            Forwarded to :class:`NPClassifierEnsemble`.
        """
        path = label_names_pkl or _LABEL_PKL
        with open(path, "rb") as f:
            all_names: List[str] = pickle.load(f)
        if len(all_names) != 730:
            raise ValueError(
                f"Expected 730 label names, got {len(all_names)}. "
                "Provide pathway_labels / superclass_labels / class_labels explicitly."
            )
        return cls(
            pathway_model,
            superclass_model,
            class_model,
            pathway_labels=all_names[:_N_PATHWAY],
            superclass_labels=all_names[_N_PATHWAY : _N_PATHWAY + _N_SUPERCLASS],
            class_labels=all_names[_N_PATHWAY + _N_SUPERCLASS :],
            **kwargs,
        )

    @classmethod
    def from_checkpoints(
        cls,
        pathway_ckpt: Union[str, Path],
        superclass_ckpt: Union[str, Path],
        class_ckpt: Union[str, Path],
        label_names_pkl: Union[str, Path, None] = None,
        pathway_labels: Optional[List[str]] = None,
        superclass_labels: Optional[List[str]] = None,
        class_labels: Optional[List[str]] = None,
        **kwargs,
    ) -> "NPClassifierEnsemble":
        """Load three checkpoints and build the ensemble in one call.

        Parameters
        ----------
        pathway_ckpt, superclass_ckpt, class_ckpt:
            Paths to ``.ckpt`` files produced by the hierarchical training
            scripts (``07_`` or ``08_``).
        label_names_pkl:
            Path to a label-names pickle (730 names).  Falls back to the
            bundled copy when ``None`` and no explicit label lists given.
        pathway_labels, superclass_labels, class_labels:
            Explicit label lists — take precedence over ``label_names_pkl``.
        **kwargs:
            Forwarded to :class:`NPClassifierEnsemble`.
        """
        pathway_model = NPClassifierLightning.load_from_checkpoint(str(pathway_ckpt))
        superclass_model = NPClassifierLightning.load_from_checkpoint(
            str(superclass_ckpt)
        )
        class_model = NPClassifierLightning.load_from_checkpoint(str(class_ckpt))

        if (
            pathway_labels is not None
            and superclass_labels is not None
            and class_labels is not None
        ):
            return cls(
                pathway_model,
                superclass_model,
                class_model,
                pathway_labels=pathway_labels,
                superclass_labels=superclass_labels,
                class_labels=class_labels,
                **kwargs,
            )
        return cls.from_label_names_file(
            pathway_model,
            superclass_model,
            class_model,
            label_names_pkl=label_names_pkl,
            **kwargs,
        )

    # ── hierarchy building ──────────────────────────────────────────────────

    def _resolve_name(self, name: str, aliases: Dict[str, str]) -> str:
        """Apply alias map to normalise a CSV label name."""
        return aliases.get(name, name)

    def _build_hierarchy(self, aliases: Dict[str, str]) -> None:
        """Build {our_idx → [our_idx, ...]} mappings from the ontology."""
        onto = self._onto

        onto_pathway_names = list(onto["Pathway"].keys())
        onto_superclass_names = list(onto["Superclass"].keys())
        onto_class_names = list(onto["Class"].keys())

        # Name → position in ontology lists
        onto_s_idx = {n: i for i, n in enumerate(onto_superclass_names)}
        onto_c_idx = {n: i for i, n in enumerate(onto_class_names)}

        # Name → position in OUR label lists
        our_p_idx = {n: i for i, n in enumerate(self._pathway_labels)}
        our_s_idx = {n: i for i, n in enumerate(self._superclass_labels)}

        missing_super: List[str] = []
        missing_class: List[str] = []

        # class_idx → list of our pathway indices
        # class_idx → list of our superclass indices
        self._class_to_pathways: Dict[int, List[int]] = {}
        self._class_to_superclasses: Dict[int, List[int]] = {}
        for our_c_i, c_name in enumerate(self._class_labels):
            resolved = self._resolve_name(c_name, aliases)
            onto_c_i = onto_c_idx.get(resolved)
            if onto_c_i is None:
                missing_class.append(c_name)
                self._class_to_pathways[our_c_i] = []
                self._class_to_superclasses[our_c_i] = []
                continue
            hier = onto["Class_hierarchy"].get(str(onto_c_i), {})

            our_paths = []
            for onto_p_i in hier.get("Pathway", []):
                p_name = onto_pathway_names[onto_p_i]
                if p_name in our_p_idx:
                    our_paths.append(our_p_idx[p_name])
            self._class_to_pathways[our_c_i] = our_paths

            our_supers = []
            for onto_s_i in hier.get("Superclass", []):
                s_name = onto_superclass_names[onto_s_i]
                resolved_s = self._resolve_name(s_name, aliases)
                # look up resolved name in our labels (note: our labels may
                # use different names, so check both original and resolved)
                idx = our_s_idx.get(s_name) or our_s_idx.get(resolved_s)
                if idx is not None:
                    our_supers.append(idx)
            self._class_to_superclasses[our_c_i] = our_supers

        # superclass_idx → list of our pathway indices
        self._super_to_pathways: Dict[int, List[int]] = {}
        for our_s_i, s_name in enumerate(self._superclass_labels):
            resolved = self._resolve_name(s_name, aliases)
            onto_s_i = onto_s_idx.get(resolved)
            if onto_s_i is None:
                missing_super.append(s_name)
                self._super_to_pathways[our_s_i] = []
                continue
            hier = onto["Super_hierarchy"].get(str(onto_s_i), {})
            our_paths = []
            for onto_p_i in hier.get("Pathway", []):
                p_name = onto_pathway_names[onto_p_i]
                if p_name in our_p_idx:
                    our_paths.append(our_p_idx[p_name])
            self._super_to_pathways[our_s_i] = our_paths

        if missing_super:
            warnings.warn(
                f"{len(missing_super)} superclass label(s) not found in ontology "
                f"and will not contribute to hierarchy voting: {missing_super}",
                UserWarning,
                stacklevel=3,
            )
        if missing_class:
            warnings.warn(
                f"{len(missing_class)} class label(s) not found in ontology "
                f"and will not contribute to hierarchy voting: {missing_class}",
                UserWarning,
                stacklevel=3,
            )

    # ── inference ───────────────────────────────────────────────────────────

    def _run_model(
        self, model: NPClassifierLightning, features: np.ndarray
    ) -> np.ndarray:
        loader = DataLoader(
            TensorDataset(torch.tensor(features, dtype=torch.float32)),
            batch_size=512,
            shuffle=False,
            num_workers=0,
        )
        batches = self._trainer.predict(model, loader)
        logits = torch.cat(batches, dim=0)
        return torch.sigmoid(logits).cpu().numpy()

    def predict_from_features(
        self,
        features: np.ndarray,
        smiles_list: Optional[Sequence[str]] = None,
        check_glycoside: bool = True,
    ) -> List[Dict[str, object]]:
        """Run voting prediction on pre-computed fingerprint features.

        Parameters
        ----------
        features:
            Float32 array of shape ``(N, 6144)``.
        smiles_list:
            Optional list of SMILES (same length as *features*) used for
            glycoside detection.  When ``None`` or ``check_glycoside=False``,
            the ``"isglycoside"`` field is always ``False``.
        check_glycoside:
            Whether to run the glycoside SMARTS check.

        Returns
        -------
        List of dicts with keys ``"pathway"``, ``"superclass"``, ``"class"``,
        ``"isglycoside"``.  Each value is a list of label-name strings (or
        ``bool`` for ``"isglycoside"``).
        """
        path_probs = self._run_model(self.pathway_model, features)
        super_probs = self._run_model(self.superclass_model, features)
        class_probs = self._run_model(self.class_model, features)

        results = []
        for i in range(len(features)):
            p_prob = path_probs[i]
            s_prob = super_probs[i]
            c_prob = class_probs[i]

            n_path = list(np.where(p_prob >= self.pathway_threshold)[0])
            n_super = list(np.where(s_prob >= self.superclass_threshold)[0])
            n_class = list(np.where(c_prob >= self.class_threshold)[0])

            if not n_path:
                n_path = [int(np.argmax(p_prob))]
            if not n_super:
                n_super = [int(np.argmax(s_prob))]
            if not n_class:
                n_class = [int(np.argmax(c_prob))]

            glycoside = (
                _is_glycoside(smiles_list[i])
                if check_glycoside and smiles_list is not None
                else False
            )

            voted = self._vote(n_path, n_super, n_class, s_prob, c_prob, glycoside)
            results.append(voted)

        return results

    def predict(
        self,
        smiles: Union[str, Sequence[str]],
        check_glycoside: bool = True,
    ) -> Union[Dict[str, object], List[Dict[str, object]]]:
        """Featurize *smiles* and return voted predictions.

        Parameters
        ----------
        smiles:
            A single SMILES string or a list of SMILES strings.
        check_glycoside:
            Whether to run the glycoside SMARTS check.

        Returns
        -------
        For a single SMILES: a ``dict`` with keys ``"pathway"``,
        ``"superclass"``, ``"class"``, ``"isglycoside"``.
        For a list: a list of such dicts.
        """
        single = isinstance(smiles, str)
        smiles_list = [smiles] if single else list(smiles)

        features = self.featurizer.transform(smiles_list)
        results = self.predict_from_features(
            features, smiles_list, check_glycoside=check_glycoside
        )
        return results[0] if single else results

    # ── voting ───────────────────────────────────────────────────────────────

    def _vote(
        self,
        n_path: List[int],
        n_super: List[int],
        n_class: List[int],
        pred_super: np.ndarray,
        pred_class: np.ndarray,
        isglycoside: bool,
    ) -> Dict[str, object]:
        """Apply the NP-Classifier voting algorithm (original logic, our index space).

        Returns a dict with ``"pathway"``, ``"superclass"``, ``"class"``,
        ``"isglycoside"`` keys.
        """
        # Pathway votes derived from each model level
        path_from_class = list(
            set(p for c in n_class for p in self._class_to_pathways.get(c, []))
        )
        path_from_superclass = list(
            set(p for s in n_super for p in self._super_to_pathways.get(s, []))
        )

        # --- Step 1: find agreed pathways ---
        path_for_vote = n_path + path_from_class + path_from_superclass
        path = list(set(k for k in path_for_vote if path_for_vote.count(k) == 3))

        if not path:
            path = list(set(k for k in path_for_vote if path_for_vote.count(k) == 2))
            if len(path) > 1:
                # drop class votes and retry
                path_for_vote = n_path + path_from_superclass
                n_class = []
                path = list(
                    set(k for k in path_for_vote if path_for_vote.count(k) == 2)
                )

        # No pathway agreement → return pathway-only prediction
        if not path:
            return {
                "pathway": [self._pathway_labels[w] for w in n_path],
                "superclass": [],
                "class": [],
                "isglycoside": isglycoside,
            }

        # --- Step 2: refine superclass and class ---
        if set(n_path) & set(path):
            # pathway model agrees
            if set(path) & set(path_from_superclass):
                # superclass-derived pathway agrees
                n_super = [
                    s
                    for s in n_super
                    if set(path) & set(self._super_to_pathways.get(s, []))
                ]
                if not n_super:
                    n_class = [
                        m
                        for m in n_class
                        if set(path) & set(self._class_to_pathways.get(m, []))
                    ]
                    n_super = list(
                        set(
                            itertools.chain.from_iterable(
                                self._class_to_superclasses.get(c, []) for c in n_class
                            )
                        )
                    )
                elif len(n_super) > 1:
                    n_class = [
                        u
                        for u in n_class
                        if set(path) & set(self._class_to_pathways.get(u, []))
                    ]
                    if n_class:
                        n_super = list(
                            set(
                                itertools.chain.from_iterable(
                                    self._class_to_superclasses.get(c, [])
                                    for c in n_class
                                )
                            )
                        )
                        n_path = list(
                            set(
                                itertools.chain.from_iterable(
                                    self._class_to_pathways.get(c, []) for c in n_class
                                )
                            )
                        )
                    elif len(path) == 1:
                        n_super = [int(np.argmax(pred_super))]
                        n_class = [
                            m
                            for m in [int(np.argmax(pred_class))]
                            if set(n_super)
                            & set(self._class_to_superclasses.get(m, []))
                        ]
                else:
                    n_class = [
                        o
                        for o in n_class
                        if set(n_super) & set(self._class_to_superclasses.get(o, []))
                    ]
                    if not n_class:
                        n_class = [
                            m
                            for m in [int(np.argmax(pred_class))]
                            if set(n_super)
                            & set(self._class_to_superclasses.get(m, []))
                        ]
            else:
                # class-derived pathway agrees but superclass does not
                n_class = [
                    p
                    for p in n_class
                    if set(path) & set(self._class_to_pathways.get(p, []))
                ]
                n_super = list(
                    set(
                        itertools.chain.from_iterable(
                            self._class_to_superclasses.get(c, []) for c in n_class
                        )
                    )
                )
        else:
            # pathway model does not agree — trust class/superclass
            n_super = [
                s
                for s in n_super
                if set(path) & set(self._super_to_pathways.get(s, []))
            ]
            if not n_super:
                n_class = [
                    m
                    for m in n_class
                    if set(path) & set(self._class_to_pathways.get(m, []))
                ]
                n_super = list(
                    set(
                        itertools.chain.from_iterable(
                            self._class_to_superclasses.get(c, []) for c in n_class
                        )
                    )
                )
                n_path = list(
                    set(
                        itertools.chain.from_iterable(
                            self._class_to_pathways.get(c, []) for c in n_class
                        )
                    )
                )
            elif len(n_super) > 1:
                n_class = [
                    u
                    for u in n_class
                    if set(path) & set(self._class_to_pathways.get(u, []))
                ]
                n_super = list(
                    set(
                        itertools.chain.from_iterable(
                            self._class_to_superclasses.get(c, []) for c in n_class
                        )
                    )
                )
                n_path = list(
                    set(
                        itertools.chain.from_iterable(
                            self._class_to_pathways.get(c, []) for c in n_class
                        )
                    )
                )
            else:
                n_class = [
                    o
                    for o in n_class
                    if set(path) & set(self._class_to_pathways.get(o, []))
                ]
                n_super = list(
                    set(
                        itertools.chain.from_iterable(
                            self._class_to_superclasses.get(c, []) for c in n_class
                        )
                    )
                )
                n_path = list(
                    set(
                        itertools.chain.from_iterable(
                            self._class_to_pathways.get(c, []) for c in n_class
                        )
                    )
                )

        if not n_super:
            n_super = [int(np.argmax(pred_super))]
        if not n_class:
            n_class = [int(np.argmax(pred_class))]

        return {
            "pathway": [self._pathway_labels[r] for r in path],
            "superclass": [self._superclass_labels[s] for s in n_super],
            "class": [self._class_labels[t] for t in n_class],
            "isglycoside": isglycoside,
        }
