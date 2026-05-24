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

Bit-info / explainability
-------------------------
``featurize_smiles(..., return_bit_info=True)`` returns the accumulated bit-info
map ``{bit: [(atom_idx, env_radius), ...]}``.  Use ``fp_index_to_radius_bit``
to translate a feature-vector index back to (env_radius, morgan_bit), then
``NPClassifierFeaturizer.draw_bits`` to draw the molecule with the
corresponding structural environments highlighted.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

# Palette of (R, G, B) floats in [0, 1] for multi-feature highlighting
_COLORS: List[Tuple[float, float, float]] = [
    (0.12, 0.47, 0.71),  # blue
    (0.89, 0.10, 0.11),  # red
    (0.20, 0.63, 0.17),  # green
    (1.00, 0.50, 0.00),  # orange
    (0.42, 0.24, 0.60),  # purple
    (0.69, 0.35, 0.16),  # brown
    (0.65, 0.81, 0.89),  # light blue
    (0.98, 0.60, 0.60),  # pink
]

# Bit-info type alias
BitInfoMap = Dict[int, List[Tuple[int, int]]]


def fp_index_to_radius_bit(fp_index: int, radius: int = 2) -> Tuple[int, int]:
    """Translate a fingerprint vector index to ``(env_radius, morgan_bit)``.

    The fingerprint layout for radius *r* is::

        fp[0 : 2048*r]          binary part  →  env_radius = fp_index // 2048 + 1
        fp[2048*r : 2048*(r+1)] formula part →  env_radius = 0

    Parameters
    ----------
    fp_index:
        Index in the flat fingerprint vector (0 ≤ fp_index < 2048*(radius+1)).
    radius:
        Morgan radius used when the fingerprint was generated.

    Returns
    -------
    (env_radius, morgan_bit) : Tuple[int, int]
    """
    if fp_index < 2048 * radius:
        env_radius = fp_index // 2048 + 1
        morgan_bit = fp_index % 2048
    else:
        env_radius = 0
        morgan_bit = fp_index - 2048 * radius
    return env_radius, morgan_bit


def featurize_smiles(
    smiles: str,
    radius: int = 2,
    use_chirality: bool = False,
    return_bit_info: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], BitInfoMap]]:
    """Convert one SMILES string to a 6144-dim float32 fingerprint array.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    radius:
        Maximum Morgan radius.  The output length is ``2048 * (radius + 1)``.
    use_chirality:
        Whether to include chirality in the Morgan fingerprint.
    return_bit_info:
        When ``True``, return a tuple ``(fp, bit_info_map)`` where
        ``bit_info_map`` maps each active Morgan bit to a list of
        ``(atom_idx, env_radius)`` tuples.

    Returns
    -------
    np.ndarray of shape (2048 * (radius + 1),) and dtype float32, or None if
    RDKit cannot parse the molecule.  If ``return_bit_info`` is True, returns
    a ``(fp, bit_info_map)`` tuple (``fp`` may be None).

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
        return (None, {}) if return_bit_info else None

    binary = np.zeros(2048 * radius, dtype=int)
    formula = np.zeros(2048, dtype=int)

    mol_bi: BitInfoMap = {}
    for r in range(radius + 1):
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=r,
            fpSize=2048,
            includeChirality=use_chirality,
        )
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        mol_fp = gen.GetFingerprint(mol, additionalOutput=ao)

        for bit, infos in ao.GetBitInfoMap().items():
            if bit not in mol_bi:
                mol_bi[bit] = list(infos)
            else:
                existing = set(mol_bi[bit])
                for info in infos:
                    if info not in existing:
                        mol_bi[bit].append(info)

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

    fp = np.concatenate((binary, formula), dtype=np.float32)
    if return_bit_info:
        return fp, copy.deepcopy(mol_bi)
    return fp


def _environment_atoms_and_bonds(
    mol: Chem.Mol, atom_idx: int, env_radius: int
) -> Tuple[List[int], List[int]]:
    """Return all atom and bond indices in the Morgan environment of *atom_idx*."""
    if env_radius == 0:
        return [atom_idx], []
    env_bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, atom_idx))
    atoms = {atom_idx}
    for bond_idx in env_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return list(atoms), env_bonds


def draw_bit_fragment(
    mol: Chem.Mol,
    bit_info: BitInfoMap,
    fp_index: int,
    radius: int = 2,
    size: Tuple[int, int] = (150, 150),
    as_svg: bool = False,
) -> Optional[Union[str, "PIL.Image.Image"]]:  # noqa: F821
    """Draw the Morgan environment for *fp_index* as an isolated fragment image.

    Uses :func:`~rdkit.Chem.Draw.DrawMorganEnv` to render only the relevant
    substructure (centre atom + its environment up to the bit radius).

    Parameters
    ----------
    mol:
        RDKit molecule.
    bit_info:
        Bit-info map from :func:`featurize_smiles` or
        :meth:`NPClassifierFeaturizer.get_bit_info`.
    fp_index:
        Flat fingerprint vector index (0 ≤ fp_index < feature_dim).
    radius:
        Morgan radius used when the fingerprint was generated.
    size:
        ``(width, height)`` in pixels.
    as_svg:
        Return an SVG string instead of a PIL Image.

    Returns
    -------
    PIL Image, SVG string, or ``None`` if *fp_index* is not active for this molecule.
    """
    import io
    from rdkit.Chem.Draw import DrawMorganEnv

    env_radius, morgan_bit = fp_index_to_radius_bit(fp_index, radius)
    if morgan_bit not in bit_info:
        return None
    atom_idx = next(
        (a for a, r in bit_info[morgan_bit] if r == env_radius),
        None,
    )
    if atom_idx is None:
        return None

    result = DrawMorganEnv(mol, atom_idx, env_radius, molSize=size, useSVG=as_svg)
    if as_svg:
        return result  # SVG string
    # DrawMorganEnv returns raw PNG bytes when useSVG=False
    try:
        from PIL import Image as _PIL_Image

        return _PIL_Image.open(io.BytesIO(result))
    except ImportError:
        return result  # fall back to raw bytes


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

    # ------------------------------------------------------------------
    # Explainability helpers
    # ------------------------------------------------------------------

    def get_bit_info(self, smiles: str) -> BitInfoMap:
        """Return the Morgan bit-info map for *smiles*.

        The map has the form ``{morgan_bit: [(atom_idx, env_radius), ...]}``.
        Use :func:`fp_index_to_radius_bit` to translate a feature-vector
        index to the ``(env_radius, morgan_bit)`` key needed here.

        Parameters
        ----------
        smiles:
            Input SMILES string.

        Returns
        -------
        BitInfoMap — empty dict if the SMILES cannot be parsed.
        """
        _, bit_info = featurize_smiles(
            smiles, self.radius, self.use_chirality, return_bit_info=True
        )
        return bit_info

    def get_highlighted_atoms(
        self,
        mol: Chem.Mol,
        bit_info: BitInfoMap,
        feature_indices: Sequence[int],
    ) -> Tuple[List[int], List[int]]:
        """Collect all atom and bond indices covered by *feature_indices*.

        Each feature index is mapped to an ``(env_radius, morgan_bit)`` pair.
        All atoms/bonds belonging to the corresponding Morgan environments are
        returned as flat lists (union across all requested features).

        Parameters
        ----------
        mol:
            RDKit molecule (must match the molecule used to build *bit_info*).
        bit_info:
            Bit-info map from :meth:`get_bit_info`.
        feature_indices:
            Fingerprint vector indices of the features to highlight.

        Returns
        -------
        (atom_indices, bond_indices) : Tuple[List[int], List[int]]
        """
        all_atoms: set = set()
        all_bonds: set = set()
        for fp_idx in feature_indices:
            env_radius, morgan_bit = fp_index_to_radius_bit(fp_idx, self.radius)
            if morgan_bit not in bit_info:
                continue
            for atom_idx, info_radius in bit_info[morgan_bit]:
                if info_radius != env_radius:
                    continue
                atoms, bonds = _environment_atoms_and_bonds(mol, atom_idx, env_radius)
                all_atoms.update(atoms)
                all_bonds.update(bonds)
        return list(all_atoms), list(all_bonds)

    def draw_bits(
        self,
        smiles: str,
        feature_indices: Sequence[int],
        size: Tuple[int, int] = (400, 300),
        as_svg: bool = True,
        per_feature_colors: bool = False,
    ) -> Union[str, "PIL.Image.Image"]:  # noqa: F821
        """Draw *smiles* with highlighted Morgan environments for *feature_indices*.

        Parameters
        ----------
        smiles:
            Input SMILES string.
        feature_indices:
            Fingerprint vector indices whose structural environments should be
            highlighted (e.g. the top-k most important features from SHAP or
            model weights).
        size:
            ``(width, height)`` in pixels.
        as_svg:
            If ``True`` (default), return an SVG string.  Otherwise return a
            PIL ``Image``.
        per_feature_colors:
            If ``True``, assign a distinct colour to each feature's environment
            so overlapping environments can be distinguished.  If ``False``,
            all environments share the same colour.

        Returns
        -------
        SVG string or PIL Image.

        Raises
        ------
        ValueError
            If *smiles* cannot be parsed by RDKit.

        Examples
        --------
        >>> feat = NPClassifierFeaturizer()
        >>> bit_info = feat.get_bit_info("Cn1cnc2c1c(=O)n(c(=O)n2C)C")
        >>> active = [i for i, v in enumerate(feat.transform(["Cn1cnc2c1c(=O)n(c(=O)n2C)C"])[0]) if v > 0]
        >>> svg = feat.draw_bits("Cn1cnc2c1c(=O)n(c(=O)n2C)C", active[:5])
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")

        bit_info = self.get_bit_info(smiles)

        if not per_feature_colors:
            highlight_atoms, highlight_bonds = self.get_highlighted_atoms(
                mol, bit_info, feature_indices
            )
            atom_colors: Dict[int, Tuple] = {a: _COLORS[0] for a in highlight_atoms}
            bond_colors: Dict[int, Tuple] = {b: _COLORS[0] for b in highlight_bonds}
        else:
            atom_colors = {}
            bond_colors = {}
            highlight_atoms_set: set = set()
            highlight_bonds_set: set = set()
            for i, fp_idx in enumerate(feature_indices):
                color = _COLORS[i % len(_COLORS)]
                env_radius, morgan_bit = fp_index_to_radius_bit(fp_idx, self.radius)
                if morgan_bit not in bit_info:
                    continue
                for atom_idx, info_radius in bit_info[morgan_bit]:
                    if info_radius != env_radius:
                        continue
                    atoms, bonds = _environment_atoms_and_bonds(
                        mol, atom_idx, env_radius
                    )
                    for a in atoms:
                        highlight_atoms_set.add(a)
                        atom_colors.setdefault(a, color)
                    for b in bonds:
                        highlight_bonds_set.add(b)
                        bond_colors.setdefault(b, color)
            highlight_atoms = list(highlight_atoms_set)
            highlight_bonds = list(highlight_bonds_set)

        if as_svg:
            drawer = rdMolDraw2D.MolDraw2DSVG(*size)
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=bond_colors,
            )
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
        else:
            return Draw.MolToImage(
                mol,
                size=size,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=atom_colors,
            )

    def draw_bit_fragment(
        self,
        smiles: str,
        fp_index: int,
        size: Tuple[int, int] = (150, 150),
        as_svg: bool = False,
    ) -> Optional[Union[str, "PIL.Image.Image"]]:  # noqa: F821
        """Draw the Morgan environment fragment for a single fingerprint vector index.

        Delegates to :func:`draw_bit_fragment` (module-level).

        Parameters
        ----------
        smiles:
            Input SMILES string.
        fp_index:
            Flat fingerprint vector index.
        size:
            ``(width, height)`` in pixels.
        as_svg:
            Return SVG string instead of PIL Image.

        Returns
        -------
        PIL Image, SVG string, or ``None`` if the bit is not active or the
        SMILES cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        bit_info = self.get_bit_info(smiles)
        return draw_bit_fragment(mol, bit_info, fp_index, self.radius, size, as_svg)
