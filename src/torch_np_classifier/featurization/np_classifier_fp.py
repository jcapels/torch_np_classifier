"""
Standalone NP Classifier fingerprint featurizer.

Generates a 6144-dim float32 vector from a SMILES string using Morgan
fingerprints without any deepmol or plants-sm dependencies.

Fingerprint layout
------------------
- binary  : shape (2048 * radius,) = (4096,) for radius=2
            For each radius r in 1..radius, position 2048*(r-1)+i stores the
            number of atoms whose Morgan environment at exactly radius r maps
            to bit i.
- formula : shape (2048,)
            Radius-0 counts — number of atoms whose single-atom environment
            maps to each bit (essentially atom-type counts).
- output  : np.concatenate((binary, formula))  →  total 6144 dims.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def featurize_smiles(
    smiles: str,
    radius: int = 2,
    use_chirality: bool = False,
) -> Optional[np.ndarray]:
    """Convert one SMILES string to a 6144-dim float32 fingerprint array.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    radius:
        Maximum Morgan radius.  The output length is ``2048 * (radius + 1)``.
    use_chirality:
        Whether to include chirality in the Morgan fingerprint.

    Returns
    -------
    np.ndarray of shape (2048 * (radius + 1),) and dtype float32, or None if
    RDKit cannot parse the molecule.

    Examples
    --------
    >>> fp = featurize_smiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C")  # caffeine
    >>> fp.shape
    (6144,)
    >>> fp.dtype
    dtype('float32')
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    binary = np.zeros(2048 * radius, dtype=int)
    formula = np.zeros(2048, dtype=int)

    mol_bi: dict = {}
    for r in range(radius + 1):
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=r,
            fpSize=2048,
            includeChirality=use_chirality,
        )
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        mol_fp = gen.GetFingerprint(mol, additionalOutput=ao)

        # Merge this radius's bitInfoMap into the shared accumulator
        for bit, infos in ao.GetBitInfoMap().items():
            if bit not in mol_bi:
                mol_bi[bit] = list(infos)
            else:
                existing = set(mol_bi[bit])
                for info in infos:
                    if info not in existing:
                        mol_bi[bit].append(info)

        # Collect bits whose *shortest* contributing environment equals r
        active_bits_at_r: List[int] = []
        for bit in mol_fp.GetOnBits():
            for _atom_idx, env_radius in mol_bi[bit]:
                if env_radius == r:
                    active_bits_at_r.append(bit)
                    break

        if r == 0:
            for bit in active_bits_at_r:
                formula[bit] = len([k for k in mol_bi[bit] if k[1] == 0])
        else:
            for bit in active_bits_at_r:
                binary[2048 * (r - 1) + bit] = len(
                    [k for k in mol_bi[bit] if k[1] == r]
                )

    return np.concatenate((binary, formula), dtype=np.float32)


class NPClassifierFeaturizer:
    """Batch featurizer that converts a list of SMILES to a float32 matrix.

    Parameters
    ----------
    radius:
        Morgan radius.  ``feature_dim = 2048 * (radius + 1)``.
    use_chirality:
        Whether chirality is included in the Morgan fingerprint.
    n_jobs:
        Number of parallel jobs (forwarded to ``joblib.Parallel``).
        ``-1`` uses all available CPUs.

    Examples
    --------
    >>> feat = NPClassifierFeaturizer()
    >>> X = feat.transform(["Cn1cnc2c1c(=O)n(c(=O)n2C)C", "C"])
    >>> X.shape
    (2, 6144)
    """

    def __init__(
        self,
        radius: int = 2,
        use_chirality: bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.radius = radius
        self.use_chirality = use_chirality
        self.n_jobs = n_jobs

    @property
    def feature_dim(self) -> int:
        """Dimensionality of each output fingerprint vector."""
        return 2048 * (self.radius + 1)

    def transform(self, smiles_list: List[str]) -> np.ndarray:
        """Featurize a list of SMILES strings.

        Invalid SMILES (those that RDKit cannot parse) produce zero vectors.

        Parameters
        ----------
        smiles_list:
            Iterable of SMILES strings.

        Returns
        -------
        np.ndarray of shape ``(len(smiles_list), feature_dim)`` and dtype
        float32.
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(featurize_smiles)(smi, self.radius, self.use_chirality)
            for smi in smiles_list
        )

        out = np.zeros((len(smiles_list), self.feature_dim), dtype=np.float32)
        for i, fp in enumerate(results):
            if fp is not None:
                out[i] = fp
        return out
