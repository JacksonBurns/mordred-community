"""Mordred-name 2D descriptors computed with RDKit only."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cached_property
import csv
import math
from pathlib import Path
import re
from typing import NamedTuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdPartialCharges
from rdkit.Chem.EState import AtomTypes
from rdkit.Chem.EState.EState import EStateIndices

from .mordred_rdkit_registry import (
    ATOMIC_ID_DESCRIPTORS,
    AUTOCORRELATION_DESCRIPTORS,
    BCUT_DESCRIPTORS,
    CARBON_TYPES_DESCRIPTORS,
    CHI_DESCRIPTORS,
    ESTATE_ATOM_TYPE_DESCRIPTORS,
    ETA_DESCRIPTORS,
    INFORMATION_CONTENT_DESCRIPTORS,
    ESTATE_ATOM_TYPE_MAXMIN_DESCRIPTORS,
    ESTATE_ATOM_TYPE_SUM_DESCRIPTORS,
    MDE_DESCRIPTORS,
    PATH_COUNT_DESCRIPTORS,
    CONSTITUTIONAL_DESCRIPTORS,
    PHYSICAL_PROPERTY_DESCRIPTORS,
    SPECTRAL_DESCRIPTORS,
    TOPOLOGICAL_CHARGE_DESCRIPTORS,
    RING_COUNT_DESCRIPTORS,
    SMALL_GRAPH_FORMULA_DESCRIPTORS,
    SUPPORTED_MORDRED_2D_DESCRIPTORS,
    WALK_COUNT_DESCRIPTORS,
    _BARYSZ_PROP_CODES,
    _SPECTRAL_METHODS,
)

DescriptorValue = float | int
DescriptorFunction = Callable[["_DescriptorContext"], DescriptorValue]


class _RingMetadata(NamedTuple):
    atoms: frozenset[int]
    size: int
    is_aromatic: bool
    has_hetero: bool

_RING_COUNT_PATTERN = re.compile(r"^n(?:(G12|\d+))?(F)?([aA])?(H)?Ring$")
_PATH_COUNT_PATTERN = re.compile(r"^(T)?(?:(pi)PC|MPC)(\d+)$")
_WALK_COUNT_PATTERN = re.compile(r"^(T)?(?:(M)WC|(SR)W)(\d+)$")
_AUTOCORRELATION_PATTERN = re.compile(
    r"^(AATSC|AATS|ATSC|ATS|MATS|GATS)(\d+)(are|se|pe|dv|Z|m|v|p|i|d|s|c)$"
)
_HALOGEN_ATOMIC_NUMBERS = {9, 17, 35, 53}
_MORDRED_HALOGEN_ATOMIC_NUMBERS = {9, 17, 35, 53, 85, 117}
_ACID_GROUP_SMARTS = (
    "[O;H1]-[C,S,P]=O",
    "[*;-;!$(*~[*;+])]",
    "[NH](S(=O)=O)C(F)(F)F",
    "n1nnnc1",
)
_BASE_GROUP_SMARTS = (
    "[NH2]-[CX4]",
    "[NH](-[CX4])-[CX4]",
    "N(-[CX4])(-[CX4])-[CX4]",
    "[*;+;!$(*~[*;-])]",
    "N=C-N",
    "N-C=N",
)
_ATOM_SYMBOLS_BY_DESCRIPTOR = {
    "nB": "B",
    "nC": "C",
    "nN": "N",
    "nO": "O",
    "nS": "S",
    "nP": "P",
    "nF": "F",
    "nCl": "Cl",
    "nBr": "Br",
    "nI": "I",
}


def _load_atomic_property_tables() -> dict[str, dict[int, float]]:
    """Load per-element property tables from the bundled CSV data file.

    Each column becomes a ``{atomic_number: value}`` dict; blank cells are
    treated as missing so lookups fall back to NaN via
    :func:`_atomic_property_value`.
    """

    path = Path(__file__).with_name("atomic_properties.csv")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = [name for name in reader.fieldnames if name != "atomic_number"]
        tables: dict[str, dict[int, float]] = {name: {} for name in columns}
        for row in reader:
            atomic_num = int(row["atomic_number"])
            for name in columns:
                cell = row[name]
                if cell != "":
                    tables[name][atomic_num] = float(cell)
    return tables


_ATOMIC_PROPERTY_TABLES = _load_atomic_property_tables()
_MASS_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["mass"]
_VDW_VOLUME_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["vdw_volume"]
_SANDERSON_EN_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["sanderson_en"]
_PAULING_EN_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["pauling_en"]
_ALLRED_ROCOW_EN_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["allred_rocow_en"]
_POLARIZABILITY_94_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["polarizability"]
_IONIZATION_POTENTIAL_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["ionization_potential"]
_MCGOWAN_VOLUME_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["mcgowan_volume"]
_PERIOD_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["period"]
_BONDI_RADII_BY_ATOMIC_NUM = _ATOMIC_PROPERTY_TABLES["bondi_radius"]

_VABC_ATOM_CONTRIBUTION_BY_ATOMIC_NUM = {
    atomic_num: 4.0 / 3.0 * math.pi * radius**3
    for atomic_num, radius in _BONDI_RADII_BY_ATOMIC_NUM.items()
}

# Canonical autocorrelation property-suffix order. The property matrix rows and
# the precomputed descriptor keys are both driven by this tuple so they cannot
# drift apart.
_AC_SUFFIXES: tuple[str, ...] = (
    "Z", "m", "v", "se", "pe", "are", "p", "i", "d", "dv", "s", "c",
)
# Descriptor names depend only on (family, order, suffix), never on the molecule,
# so build them once instead of formatting ~600 f-strings per molecule.
_AC_KEYS: dict[str, tuple[tuple[str, ...], ...]] = {
    family: tuple(
        tuple(f"{family}{order}{suffix}" for suffix in _AC_SUFFIXES)
        for order in range(9)
    )
    for family in ("ATS", "ATSC", "AATS", "AATSC", "MATS", "GATS")
}


# Barysz matrix properties (same 8 table-based props as autocorrelation,
# no valence/charge variants).  Carbon reference values are looked up once.
_BARYSZ_TABLES: tuple[tuple[str, dict[int, float], float], ...] = tuple(
    (
        prop_code,
        {
            "Z":   {z: float(z) for z in range(1, 119)},
            "m":   _MASS_BY_ATOMIC_NUM,
            "v":   _VDW_VOLUME_BY_ATOMIC_NUM,
            "se":  _SANDERSON_EN_BY_ATOMIC_NUM,
            "pe":  _PAULING_EN_BY_ATOMIC_NUM,
            "are": _ALLRED_ROCOW_EN_BY_ATOMIC_NUM,
            "p":   _POLARIZABILITY_94_BY_ATOMIC_NUM,
            "i":   _IONIZATION_POTENTIAL_BY_ATOMIC_NUM,
        }[prop_code],
        {
            "Z":   6.0,
            "m":   _MASS_BY_ATOMIC_NUM[6],
            "v":   _VDW_VOLUME_BY_ATOMIC_NUM[6],
            "se":  _SANDERSON_EN_BY_ATOMIC_NUM[6],
            "pe":  _PAULING_EN_BY_ATOMIC_NUM[6],
            "are": _ALLRED_ROCOW_EN_BY_ATOMIC_NUM[6],
            "p":   _POLARIZABILITY_94_BY_ATOMIC_NUM[6],
            "i":   _IONIZATION_POTENTIAL_BY_ATOMIC_NUM[6],
        }[prop_code],
    )
    for prop_code in _BARYSZ_PROP_CODES
)


class _DescriptorContext:
    """Per-molecule cache for descriptor calculations."""

    def __init__(self, mol: Chem.Mol) -> None:
        self.mol = mol
        self._adjacency_power_cache = {}
        self._path_count_cache = {}

    @cached_property
    def atoms(self) -> tuple[rdchem.Atom, ...]:
        return tuple(self.mol.GetAtoms())

    @cached_property
    def bonds(self) -> tuple[rdchem.Bond, ...]:
        return tuple(self.mol.GetBonds())

    @cached_property
    def bond_atom_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in self.bonds
        )

    @cached_property
    def bond_orders(self) -> tuple[float, ...]:
        return tuple(bond.GetBondTypeAsDouble() for bond in self.bonds)

    @cached_property
    def sigma_electron_counts(self) -> tuple[int, ...]:
        # Mordred ``d`` per heavy-atom; each call iterates GetNeighbors(), so cache
        # once and share across chi, bcut, and intrinsic-state consumers.
        return tuple(_sigma_electron_count(atom) for atom in self.atoms)

    @cached_property
    def valence_electron_counts(self) -> tuple[float, ...]:
        # Mordred ``dv`` per heavy-atom (also iterates GetNeighbors()).
        return tuple(_valence_electron_count(atom) for atom in self.atoms)

    @cached_property
    def intrinsic_states(self) -> tuple[float, ...]:
        # Mordred ``s`` per heavy-atom, derived from the cached sigma/valence.
        return tuple(
            _intrinsic_state_from(atom.GetAtomicNum(), sigma, valence)
            for atom, sigma, valence in zip(
                self.atoms, self.sigma_electron_counts, self.valence_electron_counts
            )
        )

    @cached_property
    def distance_matrix(self):
        return self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        return Chem.GetDistanceMatrix(self.mol, force=True)

    @cached_property
    def adjacency_matrix(self):
        return self._compute_adjacency_matrix()

    def _compute_adjacency_matrix(self):
        return Chem.GetAdjacencyMatrix(self.mol, useBO=False, force=True)

    @cached_property
    def adjacency_valences(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.adjacency_matrix.sum(axis=0))

    def adjacency_power(self, order: int):
        if order not in self._adjacency_power_cache:
            if order == 1:
                self._adjacency_power_cache[order] = self.adjacency_matrix
            else:
                self._adjacency_power_cache[order] = self.adjacency_power(
                    order - 1
                ).dot(self.adjacency_matrix)
        return self._adjacency_power_cache[order]

    def path_count(self, order: int) -> tuple[int, float]:
        if order not in self._path_count_cache:
            self._path_count_cache[order] = self._compute_path_count(order)
        return self._path_count_cache[order]

    def _compute_path_count(self, order: int) -> tuple[int, float]:
        # RDKit paths never repeat a bond, so a path of ``order`` bonds is an
        # atom-simple path iff it touches exactly ``order + 1`` distinct atoms.
        # That lets us count and pi-weight each path in a single pass over its
        # bonds, without reconstructing the ordered atom sequence.
        path_count = 0
        pi_path_count = 0.0
        expected_atom_count = order + 1
        bond_atom_pairs = self.bond_atom_pairs
        bond_orders = self.bond_orders

        for path in Chem.FindAllPathsOfLengthN(self.mol, order):
            atom_ids: set[int] = set()
            pi_weight = 1.0
            for bond_index in path:
                begin_atom, end_atom = bond_atom_pairs[bond_index]
                atom_ids.add(begin_atom)
                atom_ids.add(end_atom)
                pi_weight *= bond_orders[bond_index]

            if len(atom_ids) == expected_atom_count:
                path_count += 1
                pi_path_count += pi_weight

        return path_count, pi_path_count

    @cached_property
    def ring_atom_sets(self) -> tuple[set[int], ...]:
        return self._compute_ring_atom_sets()

    def _compute_ring_atom_sets(self) -> tuple[set[int], ...]:
        return tuple(set(ring) for ring in self.mol.GetRingInfo().AtomRings())

    @cached_property
    def fused_ring_systems(self) -> tuple[set[int], ...]:
        return self._compute_fused_ring_systems()

    def _compute_fused_ring_systems(self) -> tuple[set[int], ...]:
        rings = self.ring_atom_sets
        parent = list(range(len(rings)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for left, left_ring in enumerate(rings):
            for right in range(left + 1, len(rings)):
                if len(left_ring & rings[right]) >= 2:
                    union(left, right)

        component_atoms: dict[int, set[int]] = {}
        component_ring_counts: dict[int, int] = {}
        for index, ring in enumerate(rings):
            root = find(index)
            component_atoms.setdefault(root, set()).update(ring)
            component_ring_counts[root] = component_ring_counts.get(root, 0) + 1

        return tuple(
            atoms
            for root, atoms in component_atoms.items()
            if component_ring_counts[root] >= 2
        )

    @cached_property
    def ring_metadata(self) -> tuple[_RingMetadata, ...]:
        return self._compute_ring_metadata(self.ring_atom_sets)

    @cached_property
    def fused_ring_metadata(self) -> tuple[_RingMetadata, ...]:
        return self._compute_ring_metadata(self.fused_ring_systems)

    def _compute_ring_metadata(
        self,
        ring_sets: tuple[set[int], ...],
    ) -> tuple[_RingMetadata, ...]:
        metadata = []
        for atoms in ring_sets:
            atom_tuple = tuple(self.mol.GetAtomWithIdx(atom) for atom in atoms)
            metadata.append(
                _RingMetadata(
                    atoms=frozenset(atoms),
                    size=len(atoms),
                    is_aromatic=all(atom.GetIsAromatic() for atom in atom_tuple),
                    has_hetero=any(atom.GetAtomicNum() != 6 for atom in atom_tuple),
                )
            )
        return tuple(metadata)

    @cached_property
    def framework_linker_atoms(self) -> set[int]:
        return self._compute_framework_linker_atoms()

    def _compute_framework_linker_atoms(self) -> set[int]:
        rings = self.ring_atom_sets
        if len(rings) < 2:
            return set()

        atom_to_node: dict[int, tuple[str, int]] = {}
        ring_nodes: set[tuple[str, int]] = set()
        for ring_index, ring in enumerate(rings):
            ring_node = ("R", ring_index)
            ring_nodes.add(ring_node)
            for atom_index in ring:
                atom_to_node[atom_index] = ring_node

        graph: dict[tuple[str, int], set[tuple[str, int]]] = {}
        for bond in self.bonds:
            begin_index = bond.GetBeginAtomIdx()
            end_index = bond.GetEndAtomIdx()
            begin = atom_to_node.get(begin_index, ("A", begin_index))
            end = atom_to_node.get(end_index, ("A", end_index))
            if begin == end:
                continue
            graph.setdefault(begin, set()).add(end)
            graph.setdefault(end, set()).add(begin)

        linkers: set[int] = set()
        ring_node_list = list(ring_nodes)
        for left_index, left in enumerate(ring_node_list):
            for right in ring_node_list[left_index + 1:]:
                path = _shortest_path(graph, left, right)
                if path is not None:
                    linkers.update(
                        index for node_type, index in path if node_type == "A"
                    )
        return linkers

    @cached_property
    def implicit_hydrogen_count(self) -> int:
        return self._compute_implicit_hydrogen_count()

    def _compute_implicit_hydrogen_count(self) -> int:
        return sum(atom.GetTotalNumHs() for atom in self.atoms)

    @cached_property
    def kekulized_mol(self) -> Chem.Mol:
        return self._compute_kekulized_mol()

    def _compute_kekulized_mol(self) -> Chem.Mol:
        kekule_mol = Chem.Mol(self.mol)
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
        return kekule_mol

    @cached_property
    def explicit_hydrogen_mol(self) -> Chem.Mol:
        return Chem.AddHs(self.mol)

    @cached_property
    def autocorrelation_distance_matrix(self):
        return Chem.GetDistanceMatrix(self.explicit_hydrogen_mol, force=True)

    @cached_property
    def gasteiger_charges(self) -> tuple[float, ...]:
        # ComputeGasteigerCharges annotates atoms in place, so work on a copy to
        # avoid mutating the cached explicit-hydrogen molecule used elsewhere.
        mol = Chem.Mol(self.explicit_hydrogen_mol)
        rdPartialCharges.ComputeGasteigerCharges(mol)
        return tuple(
            float(atom.GetDoubleProp("_GasteigerCharge")) for atom in mol.GetAtoms()
        )

    @cached_property
    def autocorrelation_property_vectors(self) -> dict[str, tuple[float, ...]]:
        """Per-atom property vectors keyed by Mordred property suffix.

        The atom list is walked once: atomic numbers feed every element-table
        property via a dict lookup (instead of re-iterating the molecule per
        table), and the sigma/valence counts are computed a single time and
        reused to derive the intrinsic state.
        """

        atoms = list(self.explicit_hydrogen_mol.GetAtoms())
        atomic_nums = [atom.GetAtomicNum() for atom in atoms]
        sigma = [_sigma_electron_count(atom) for atom in atoms]
        valence = [_valence_electron_count(atom) for atom in atoms]

        def from_table(table: dict[int, float]) -> tuple[float, ...]:
            nan = float("nan")
            return tuple(table.get(z, nan) for z in atomic_nums)

        return {
            "Z": tuple(float(z) for z in atomic_nums),
            "m": from_table(_MASS_BY_ATOMIC_NUM),
            "v": from_table(_VDW_VOLUME_BY_ATOMIC_NUM),
            "se": from_table(_SANDERSON_EN_BY_ATOMIC_NUM),
            "pe": from_table(_PAULING_EN_BY_ATOMIC_NUM),
            "are": from_table(_ALLRED_ROCOW_EN_BY_ATOMIC_NUM),
            "p": from_table(_POLARIZABILITY_94_BY_ATOMIC_NUM),
            "i": from_table(_IONIZATION_POTENTIAL_BY_ATOMIC_NUM),
            "d": tuple(float(s) for s in sigma),
            "dv": tuple(valence),
            "s": tuple(
                _intrinsic_state_from(z, s, v)
                for z, s, v in zip(atomic_nums, sigma, valence)
            ),
            "c": self.gasteiger_charges,
        }

    @cached_property
    def autocorrelation_values(self) -> dict[str, float]:
        """Moreau-Broto/Moran/Geary autocorrelations for every property and lag.

        All property vectors are stacked and processed together so the shared
        per-lag graph-distance work (the adjacency-at-distance matrix and its
        quadratic forms) is computed once per lag instead of once per property.
        """

        vectors = self.autocorrelation_property_vectors
        property_matrix = np.array([vectors[s] for s in _AC_SUFFIXES], dtype=float)
        atom_count = property_matrix.shape[1]

        centered_matrix = property_matrix - property_matrix.mean(axis=1, keepdims=True)
        centered_square_sums = (centered_matrix**2).sum(axis=1)
        distance_matrix = np.asarray(self.autocorrelation_distance_matrix)

        nan = float("nan")
        property_count = len(_AC_SUFFIXES)
        nan_row = [nan] * property_count
        zero_css_mask = centered_square_sums == 0

        results: dict[str, float] = {}
        # Each family's per-suffix values are produced as one numpy row, then
        # zipped against precomputed keys; the arithmetic order matches the
        # earlier scalar code exactly so values are bit-for-bit identical.
        for order in range(0, 9):
            if order == 0:
                pair_count = atom_count
                ats = (property_matrix**2).sum(axis=1)
                atsc = centered_square_sums
                degrees = None
            else:
                adjacency = (distance_matrix == order).astype(float)
                pair_count = int(adjacency.sum() // 2)
                ats = 0.5 * ((property_matrix @ adjacency) * property_matrix).sum(axis=1)
                atsc = 0.5 * ((centered_matrix @ adjacency) * centered_matrix).sum(axis=1)
                degrees = adjacency.sum(axis=1)

            if pair_count:
                aats = (ats / pair_count).tolist()
                aatsc_row = atsc / pair_count
                aatsc = aatsc_row.tolist()
            else:
                aats = nan_row
                aatsc = nan_row

            for key, value in zip(_AC_KEYS["ATS"][order], ats.tolist()):
                results[key] = value
            for key, value in zip(_AC_KEYS["ATSC"][order], atsc.tolist()):
                results[key] = value
            for key, value in zip(_AC_KEYS["AATS"][order], aats):
                results[key] = value
            for key, value in zip(_AC_KEYS["AATSC"][order], aatsc):
                results[key] = value

            if order == 0:
                continue

            # MATS and GATS, vectorized across properties. With B the
            # adjacency-at-distance matrix and w a property vector,
            #   sum_ij B_ij (w_i - w_j)^2 = 2 (w^2 . deg) - 2 (w^T B w)
            # and w^T B w = 2 * ATS, so the Geary numerator is
            #   2 (w^2 . deg) - 4 * ATS, divided by 4 * pair_count.
            with np.errstate(divide="ignore", invalid="ignore"):
                if pair_count:
                    mats_row = np.where(
                        zero_css_mask,
                        nan,
                        atom_count * aatsc_row / centered_square_sums,
                    )
                else:
                    mats_row = np.full(property_count, nan)

                if pair_count and atom_count > 1:
                    weighted_degree = (property_matrix**2) @ degrees
                    geary_numerators = (
                        2.0 * weighted_degree - 4.0 * ats
                    ) / (4.0 * pair_count)
                    geary_denominators = centered_square_sums / (atom_count - 1)
                    gats_row = np.where(
                        zero_css_mask,
                        nan,
                        geary_numerators / geary_denominators,
                    )
                else:
                    gats_row = np.full(property_count, nan)

            for key, value in zip(_AC_KEYS["MATS"][order], mats_row.tolist()):
                results[key] = value
            for key, value in zip(_AC_KEYS["GATS"][order], gats_row.tolist()):
                results[key] = value

        return results

    @cached_property
    def _burden_matrix_base(self) -> np.ndarray:
        n = len(self.atoms)
        B = 0.001 * np.ones((n, n))
        for bond in self.bonds:
            i = bond.GetBeginAtom().GetIdx()
            j = bond.GetEndAtom().GetIdx()
            w = bond.GetBondTypeAsDouble() / 10.0
            if bond.GetBeginAtom().GetDegree() == 1 or bond.GetEndAtom().GetDegree() == 1:
                w += 0.01
            B[i, j] = w
            B[j, i] = w
        return B

    @cached_property
    def _bcut_gasteiger_diagonal(self) -> list[float]:
        # BCUT uses the heavy-atom mol; _GasteigerHCharge holds the implicit-H contribution.
        mol = Chem.Mol(self.mol)
        rdPartialCharges.ComputeGasteigerCharges(mol)
        result = []
        for atom in mol.GetAtoms():
            q = atom.GetDoubleProp("_GasteigerCharge")
            if atom.HasProp("_GasteigerHCharge"):
                q += atom.GetDoubleProp("_GasteigerHCharge")
            result.append(q)
        return result

    @cached_property
    def bcut_values(self) -> dict[str, float]:
        atoms = self.atoms
        n = len(atoms)
        props = ("Z", "m", "v", "se", "pe", "are", "p", "i", "d", "dv", "s", "c")
        diag_matrix = np.array(
            [
                [float(a.GetAtomicNum()) for a in atoms],
                [_MASS_BY_ATOMIC_NUM.get(a.GetAtomicNum(), float("nan")) for a in atoms],
                [_VDW_VOLUME_BY_ATOMIC_NUM.get(a.GetAtomicNum(), float("nan")) for a in atoms],
                [_SANDERSON_EN_BY_ATOMIC_NUM.get(a.GetAtomicNum(), float("nan")) for a in atoms],
                [_PAULING_EN_BY_ATOMIC_NUM.get(a.GetAtomicNum(), float("nan")) for a in atoms],
                [_ALLRED_ROCOW_EN_BY_ATOMIC_NUM.get(a.GetAtomicNum(), float("nan")) for a in atoms],
                [_POLARIZABILITY_94_BY_ATOMIC_NUM.get(a.GetAtomicNum(), float("nan")) for a in atoms],
                [_IONIZATION_POTENTIAL_BY_ATOMIC_NUM.get(a.GetAtomicNum(), float("nan")) for a in atoms],
                [float(s) for s in self.sigma_electron_counts],
                list(self.valence_electron_counts),
                list(self.intrinsic_states),
                self._bcut_gasteiger_diagonal,
            ],
            dtype=float,
        )  # shape (12, n)

        # Build a batch of 12 Burden matrices differing only in their diagonals.
        nan_rows = np.isnan(diag_matrix).any(axis=1)
        safe_diag = diag_matrix.copy()
        safe_diag[nan_rows] = 0.0  # sentinel to avoid LinAlgError; results overwritten below

        batch = np.broadcast_to(self._burden_matrix_base, (len(props), n, n)).copy()
        idx = np.arange(n)
        batch[:, idx, idx] = safe_diag

        ev = np.linalg.eig(batch)[0]  # (12, n), one call for all properties
        if np.iscomplexobj(ev):
            ev = ev.real
        sorted_ev = np.sort(ev, axis=1)[:, ::-1]  # descending per row
        sorted_ev[nan_rows] = float("nan")

        results: dict[str, float] = {}
        for k, prop in enumerate(props):
            results[f"BCUT{prop}-1h"] = float(sorted_ev[k, 0])
            results[f"BCUT{prop}-1l"] = float(sorted_ev[k, -1])
        return results

    @cached_property
    def constitutional_values(self) -> dict[str, float]:
        """Constitutional sums S_p = Σ(p_i/p_C) and means M_p = S_p/A over explicit-H atoms."""
        atoms = list(self.explicit_hydrogen_mol.GetAtoms())
        n = len(atoms)
        carbon = 6
        prop_funs: list[tuple[str, dict[int, float]]] = [
            ("Z",   {z: float(z) for z in range(1, 119)}),
            ("m",   _MASS_BY_ATOMIC_NUM),
            ("v",   _VDW_VOLUME_BY_ATOMIC_NUM),
            ("se",  _SANDERSON_EN_BY_ATOMIC_NUM),
            ("pe",  _PAULING_EN_BY_ATOMIC_NUM),
            ("are", _ALLRED_ROCOW_EN_BY_ATOMIC_NUM),
            ("p",   _POLARIZABILITY_94_BY_ATOMIC_NUM),
            ("i",   _IONIZATION_POTENTIAL_BY_ATOMIC_NUM),
        ]
        results: dict[str, float] = {}
        for suffix, table in prop_funs:
            carbon_val = table[carbon]
            vals = [table.get(a.GetAtomicNum(), float("nan")) / carbon_val for a in atoms]
            s = sum(vals) if not any(math.isnan(v) for v in vals) else float("nan")
            results[f"S{suffix}"] = s
            results[f"M{suffix}"] = s / n if not math.isnan(s) else float("nan")
        return results

    @cached_property
    def topological_charge_values(self) -> dict[str, float]:
        """GGI/JGI/JGT topological charge descriptors (heavy-atom mol)."""
        D = self.distance_matrix.astype(float)
        A = self.adjacency_matrix.astype(float)
        n = len(self.atoms)

        D2 = D.copy()
        D2[D2 != 0] **= -2
        np.fill_diagonal(D2, 0)
        M = A @ D2
        CT = M - M.T  # antisymmetric charge-term matrix

        D_lower = D * np.tri(n)
        D_lower[D_lower == 0] = np.inf

        results: dict[str, float] = {}
        jgt = 0.0
        for k in range(1, 11):
            f = D_lower == k
            ct_k = CT[f]
            count_k = len(ct_k)
            results[f"GGI{k}"] = float(np.abs(ct_k).sum())
            jgi_k = float((np.abs(ct_k) / count_k).sum()) if count_k > 0 else 0.0
            results[f"JGI{k}"] = jgi_k
            jgt += jgi_k
        results["JGT10"] = jgt
        return results

    @cached_property
    def chi_subgraph_accumulators(self):
        """Enumerate connected subgraphs (orders 2-7) once, classify and accumulate.

        Returns ``(sums, nan_flag, counts)`` keyed by chi subgraph type/order. The
        single ``FindAllSubgraphsOfLengthN`` sweep is shared by both ``chi_values``
        and ``kier_values`` (which only needs the path counts).
        """
        mol = self.mol
        d_vals = [float(s) for s in self.sigma_electron_counts]
        dv_vals = list(self.valence_electron_counts)
        bond_pairs = self.bond_atom_pairs

        _type_ranges: dict[str, range] = {
            "path": range(2, 8),
            "cluster": range(3, 7),
            "path_cluster": range(4, 7),
            "chain": range(3, 8),
        }
        sums: dict[tuple[str, int, str], float] = {}
        nan_flag: dict[tuple[str, int, str], bool] = {}
        counts: dict[tuple[str, int], int] = {}
        for chi_type, rng in _type_ranges.items():
            for order in rng:
                counts[(chi_type, order)] = 0
                for pname in ("d", "dv"):
                    sums[(chi_type, order, pname)] = 0.0
                    nan_flag[(chi_type, order, pname)] = False

        for order in range(2, 8):
            for use_bonds in Chem.FindAllSubgraphsOfLengthN(mol, order):
                chi_type, nodes = _classify_chi_subgraph(bond_pairs, use_bonds)
                k = (chi_type, order)
                if k not in counts:
                    continue
                counts[k] += 1
                for pname, vals in (("d", d_vals), ("dv", dv_vals)):
                    key = (chi_type, order, pname)
                    if nan_flag[key]:
                        continue
                    c = 1.0
                    bad = False
                    for node in nodes:
                        v = vals[node]
                        if math.isnan(v) or v <= 0:
                            bad = True
                            break
                        c *= v
                    if bad:
                        nan_flag[key] = True
                    else:
                        sums[key] += c ** -0.5

        return sums, nan_flag, counts

    @cached_property
    def chi_values(self) -> dict[str, float]:
        """Kier-Hall chi connectivity indices (Mordred Xp-*/Xc-*/Xch-*/Xpc-*/AXp-*)."""
        n = len(self.atoms)

        d_vals = [float(s) for s in self.sigma_electron_counts]
        dv_vals = list(self.valence_electron_counts)
        bond_pairs = self.bond_atom_pairs

        results: dict[str, float] = {}

        # Order 0: each heavy atom is its own singleton "path"
        for pname, vals in (("d", d_vals), ("dv", dv_vals)):
            has_bad = any(math.isnan(v) or v <= 0 for v in vals)
            s = float("nan") if has_bad else sum(v ** -0.5 for v in vals)
            results[f"Xp-0{pname}"] = s
            results[f"AXp-0{pname}"] = float("nan") if math.isnan(s) or n == 0 else s / n

        # Order 1: each bond is a 2-node "path"
        cnt1 = len(bond_pairs)
        for pname, vals in (("d", d_vals), ("dv", dv_vals)):
            s = 0.0
            has_bad = False
            for a, b in bond_pairs:
                va, vb = vals[a], vals[b]
                if math.isnan(va) or va <= 0 or math.isnan(vb) or vb <= 0:
                    has_bad = True
                else:
                    s += (va * vb) ** -0.5
            val = float("nan") if has_bad else s
            results[f"Xp-1{pname}"] = val
            results[f"AXp-1{pname}"] = (
                float("nan") if math.isnan(val) or cnt1 == 0 else val / cnt1
            )

        # Orders 2-7: enumerate connected subgraphs, classify each by DFS
        _type_ranges: dict[str, range] = {
            "path": range(2, 8),
            "cluster": range(3, 7),
            "path_cluster": range(4, 7),
            "chain": range(3, 8),
        }
        _type_prefix: dict[str, str] = {
            "path": "Xp",
            "cluster": "Xc",
            "path_cluster": "Xpc",
            "chain": "Xch",
        }

        sums, nan_flag, counts = self.chi_subgraph_accumulators

        for chi_type, prefix in _type_prefix.items():
            for order in _type_ranges[chi_type]:
                k = (chi_type, order)
                cnt = counts[k]
                for pname in ("d", "dv"):
                    key = (chi_type, order, pname)
                    val = float("nan") if nan_flag[key] else sums[key]
                    results[f"{prefix}-{order}{pname}"] = val
                    if chi_type == "path":
                        avg = float("nan") if math.isnan(val) or cnt == 0 else val / cnt
                        results[f"AXp-{order}{pname}"] = avg

        return results

    @cached_property
    def detour_matrix(self) -> np.ndarray | None:
        return _compute_detour_matrix(len(self.atoms), self.bond_atom_pairs)

    @cached_property
    def spectral_values(self) -> dict[str, float]:
        """Matrix-spectral descriptors from A, D, Dt, and Barysz matrices.

        Every graph matrix for the molecule is the same ``n×n`` shape, so they
        are stacked into one ``(k, n, n)`` batch and fed through a single
        ``eigh`` call (the same amortization the BCUT family uses).  The eight
        Barysz matrices likewise share one batched Floyd-Warshall, and all
        spectral aggregates are computed across the batch at once.
        """
        n = len(self.atoms)
        bond_pairs = self.bond_atom_pairs
        bond_orders = self.bond_orders
        nan = float("nan")
        results: dict[str, float] = {}

        # Collect every matrix to eigendecompose as (suffix, matrix, include_sm1).
        specs: list[tuple[str, np.ndarray, bool]] = [
            ("A", self.adjacency_matrix.astype(float), False),
            ("D", self.distance_matrix.astype(float), False),
        ]

        # ── Detour matrix (with SM1) ──────────────────────────────────────
        dt = self.detour_matrix
        if dt is not None:
            specs.append(("Dt", dt, True))
        else:
            # Disconnected molecule: Mordred require_connected=True → Missing.
            results["SM1_Dt"] = nan
            for m in _SPECTRAL_METHODS:
                results[f"{m}_Dt"] = nan

        # ── Barysz matrices (8 property variants, each with SM1) ──────────
        # Build the valid weight matrices and solve them with one batched FW.
        atomic_nums = [a.GetAtomicNum() for a in self.atoms]
        barysz_jobs: list[tuple[str, np.ndarray, float]] = []
        for prop_code, table, carbon_ref in _BARYSZ_TABLES:
            suffix = f"Dz{prop_code}"
            prop_vals = np.array([table.get(z, nan) for z in atomic_nums])
            if np.any(np.isnan(prop_vals)):
                results[f"SM1_{suffix}"] = nan
                for m in _SPECTRAL_METHODS:
                    results[f"{m}_{suffix}"] = nan
                continue
            barysz_jobs.append((suffix, prop_vals, carbon_ref))

        if barysz_jobs:
            barysz_batch = _compute_barysz_matrices(
                n, bond_pairs, bond_orders, barysz_jobs
            )
            for index, (suffix, _, _) in enumerate(barysz_jobs):
                specs.append((suffix, barysz_batch[index], True))

        _fill_spectral_batch(results, specs, bond_pairs, n)

        # SM1_Dt = trace(Dt). For n==1 the detour diagonal is 0, so SM1_Dt is 0.0
        # (same as n>=2).  Override the nan that _fill_spectral_batch would emit
        # for a 1×1 matrix.
        if dt is not None and n == 1:
            results["SM1_Dt"] = 0.0

        return results

    @cached_property
    def exact_molecular_weight(self) -> float:
        return Descriptors.ExactMolWt(self.mol)

    @cached_property
    def mol_log_p(self) -> float:
        return Crippen.MolLogP(self.mol)

    @cached_property
    def mol_mr(self) -> float:
        return Crippen.MolMR(self.mol)

    @cached_property
    def num_hba(self) -> int:
        return rdMolDescriptors.CalcNumHBA(self.mol)

    @cached_property
    def num_hbd(self) -> int:
        return rdMolDescriptors.CalcNumHBD(self.mol)

    @cached_property
    def total_atom_count_including_hydrogen(self) -> int:
        return self.mol.GetNumAtoms() + self.implicit_hydrogen_count

    @cached_property
    def atom_symbol_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for atom in self.atoms:
            symbol = atom.GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
        return counts

    @cached_property
    def aromatic_atom_count(self) -> int:
        return sum(1 for atom in self.atoms if atom.GetIsAromatic())

    @cached_property
    def halogen_count(self) -> int:
        return sum(
            1
            for atom in self.atoms
            if atom.GetAtomicNum() in _HALOGEN_ATOMIC_NUMBERS
        )

    @cached_property
    def _estate_grouped(self) -> dict[str, list[float]]:
        # TypeAtoms and EStateIndices are both O(n) C++ calls; run each once and
        # share the result across the count, maxmin, and sum descriptor families.
        grouped: dict[str, list[float]] = {}
        for atom_types, es_val in zip(AtomTypes.TypeAtoms(self.mol), EStateIndices(self.mol)):
            fval = float(es_val)
            for t in atom_types:
                lst = grouped.get(t)
                if lst is None:
                    grouped[t] = [fval]
                else:
                    lst.append(fval)
        return grouped

    @cached_property
    def estate_atom_type_counts(self) -> dict[str, int]:
        grouped = self._estate_grouped
        counts = {name: 0 for name in ESTATE_ATOM_TYPE_DESCRIPTORS}
        for t, vals in grouped.items():
            name = f"N{t}"
            if name in counts:
                counts[name] = len(vals)
        return counts

    @cached_property
    def estate_atom_type_agg(self) -> dict[str, float]:
        nan = float("nan")
        grouped = self._estate_grouped
        result: dict[str, float] = {}
        for name in ESTATE_ATOM_TYPE_MAXMIN_DESCRIPTORS:
            es_type = name[3:]
            vals = grouped.get(es_type)
            if vals is None:
                result[name] = nan
            elif name[1] == "A":
                result[name] = max(vals)
            else:
                result[name] = min(vals)
        for name in ESTATE_ATOM_TYPE_SUM_DESCRIPTORS:
            es_type = name[1:]
            vals = grouped.get(es_type)
            result[name] = sum(vals) if vals else 0
        return result

    @cached_property
    def information_content_values(self) -> dict[str, float]:
        mol = Chem.Mol(self.explicit_hydrogen_mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        n = mol.GetNumAtoms()
        nan = float("nan")
        result: dict[str, float] = {}

        if n == 0:
            for name in INFORMATION_CONTENT_DESCRIPTORS:
                result[name] = nan
            return result

        log2_n = math.log2(n) if n > 1 else 0.0
        sum_bo = sum(b.GetBondTypeAsDouble() for b in mol.GetBonds())
        log2_bo = math.log2(sum_bo) if sum_bo > 0 else nan

        bonds_dict: dict[tuple[int, int], int] = {}
        for b in mol.GetBonds():
            s, d = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            t = int(b.GetBondType())
            bonds_dict[s, d] = t
            bonds_dict[d, s] = t
        atoms = list(mol.GetAtoms())
        atom_info = [(a.GetAtomicNum(), a.GetDegree()) for a in atoms]
        masses = [a.GetMass() for a in atoms]
        znums = [info[0] for info in atom_info]
        adj = [[] for _ in range(n)]
        for b in mol.GetBonds():
            s, d = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            adj[s].append(d)
            adj[d].append(s)

        # Build every atom's order 1..5 code in a single tree walk per root; order 0
        # is just the atomic number.
        codes_by_order: list = [znums]
        for _ in range(5):
            codes_by_order.append([None] * n)
        for root in range(n):
            root_codes = _ic_root_codes(root, adj, bonds_dict, atom_info, 5)
            for order in range(1, 6):
                codes_by_order[order][root] = root_codes[order]

        for order in range(6):
            codes = codes_by_order[order]
            groups: dict = {}
            for i, code in enumerate(codes):
                g = groups.get(code)
                if g is None:
                    groups[code] = (i, 1)
                else:
                    groups[code] = (g[0], g[1] + 1)

            ic = 0.0
            mic = 0.0
            zmic = 0.0
            for rep_idx, count in groups.values():
                p = count / n
                lp = math.log2(p)
                ic -= p * lp
                mic -= masses[rep_idx] * p * lp
                zmic -= count * znums[rep_idx] * p * lp

            result[f"IC{order}"] = ic
            result[f"TIC{order}"] = n * ic
            result[f"SIC{order}"] = ic / log2_n if log2_n > 0 else nan
            result[f"BIC{order}"] = ic / log2_bo if (not math.isnan(log2_bo) and log2_bo != 0.0) else nan
            result[f"CIC{order}"] = log2_n - ic
            result[f"MIC{order}"] = mic
            result[f"ZMIC{order}"] = zmic

        return result

    @cached_property
    def _eta_kekulized_mol(self) -> Chem.Mol:
        """Heavy-atom mol, kekulized with aromatic flags preserved (for ETA/AETA)."""
        mol = Chem.Mol(self.mol)
        Chem.Kekulize(mol)
        return mol

    @cached_property
    def _eta_reference_mol(self) -> Chem.Mol | None:
        """All-C/all-SINGLE reference alkane on the heavy-atom skeleton (for ETA/AETA)."""
        return _eta_build_reference_mol(self._eta_kekulized_mol)

    @cached_property
    def _eta_reference_mol_with_h(self) -> Chem.Mol | None:
        """Reference alkane with explicit H added (for ETA epsilon_3)."""
        rmol = self._eta_reference_mol
        if rmol is None:
            return None
        mol_ref = Chem.AddHs(rmol)
        Chem.Kekulize(mol_ref)
        return mol_ref

    @cached_property
    def _eta_saturated_mol(self) -> Chem.Mol | None:
        """Saturated skeleton (C-C bonds → SINGLE, original heteroatoms kept, with H) for ETA epsilon_4."""
        return _eta_saturated_mol(self._eta_kekulized_mol)

    @cached_property
    def eta_values(self) -> dict[str, float]:
        """Extended topochemical atom (ETA/AETA) descriptors.

        All 45 descriptors require a connected molecule (EtaBase.require_connected=True).
        The reference alkane and saturated-skeleton mols are built lazily and return
        NaN for any descriptor that depends on them when construction fails.
        """
        nan = float("nan")
        n = len(self.atoms)

        if n == 0 or len(Chem.GetMolFrags(self.mol)) > 1:
            return {name: nan for name in ETA_DESCRIPTORS}

        mol = self._eta_kekulized_mol

        alpha, eps, beta_sigma, beta_ns, beta_d = _eta_atom_properties(mol)

        gamma: list[float] = []
        for i in range(n):
            denom = beta_sigma[i] + beta_ns[i] + beta_d[i]
            gamma.append(alpha[i] / denom if denom != 0.0 else nan)

        D = self.distance_matrix
        result: dict[str, float] = {}

        alpha_total = sum(alpha)
        result["ETA_alpha"] = alpha_total
        result["AETA_alpha"] = alpha_total / n

        beta_s = sum(b * 0.5 for b in beta_sigma)
        result["ETA_beta_s"] = beta_s
        result["AETA_beta_s"] = beta_s / n

        beta_ns_d = sum(beta_d)
        result["ETA_beta_ns_d"] = beta_ns_d
        result["AETA_beta_ns_d"] = beta_ns_d / n

        beta_ns_total = sum(b * 0.5 + d for b, d in zip(beta_ns, beta_d))
        result["ETA_beta_ns"] = beta_ns_total
        result["AETA_beta_ns"] = beta_ns_total / n

        beta_total = beta_s + beta_ns_total
        result["ETA_beta"] = beta_total
        result["AETA_beta"] = beta_total / n

        d_beta = beta_ns_total - beta_s
        result["ETA_dBeta"] = d_beta
        result["AETA_dBeta"] = d_beta / n

        eta = _eta_composite(gamma, D, local=False)
        eta_L = _eta_composite(gamma, D, local=True)
        result["ETA_eta"] = eta
        result["AETA_eta"] = eta / n
        result["ETA_eta_L"] = eta_L
        result["AETA_eta_L"] = eta_L / n

        # Reference mol (all-C, all-SINGLE): topology unchanged, gamma values updated.
        # EtaCompositeIndex uses the ORIGINAL mol's distance matrix even for reference.
        rmol = self._eta_reference_mol
        if rmol is not None:
            ralpha, _reps, rbeta_sigma, rbeta_ns, rbeta_d = _eta_atom_properties(rmol)
            rgamma: list[float] = []
            for i in range(n):
                rd = rbeta_sigma[i] + rbeta_ns[i] + rbeta_d[i]
                rgamma.append(ralpha[i] / rd if rd != 0.0 else nan)
            eta_R = _eta_composite(rgamma, D, local=False)
            eta_RL = _eta_composite(rgamma, D, local=True)
            alpha_R = sum(ralpha)

            result["ETA_eta_R"] = eta_R
            result["AETA_eta_R"] = eta_R / n
            result["ETA_eta_RL"] = eta_RL
            result["AETA_eta_RL"] = eta_RL / n
            result["ETA_eta_F"] = eta_R - eta
            result["AETA_eta_F"] = (eta_R - eta) / n
            result["ETA_eta_FL"] = eta_RL - eta_L
            result["AETA_eta_FL"] = (eta_RL - eta_L) / n
            result["ETA_dAlpha_A"] = max((alpha_total - alpha_R) / n, 0.0)
            result["ETA_dAlpha_B"] = max((alpha_R - alpha_total) / n, 0.0)

            if n <= 1:
                result["ETA_eta_B"] = nan
                result["AETA_eta_B"] = nan
                result["ETA_eta_BR"] = nan
                result["AETA_eta_BR"] = nan
            else:
                eta_NL = 1.0 if n == 2 else math.sqrt(2) + 0.5 * (n - 3)
                ring_count = mol.GetRingInfo().NumRings()
                eta_B = eta_NL - eta_RL
                eta_BR = eta_B + 0.086 * ring_count
                result["ETA_eta_B"] = eta_B
                result["AETA_eta_B"] = eta_B / n
                result["ETA_eta_BR"] = eta_BR
                result["AETA_eta_BR"] = eta_BR / n
        else:
            for nm in (
                "ETA_eta_R", "AETA_eta_R", "ETA_eta_RL", "AETA_eta_RL",
                "ETA_eta_F", "AETA_eta_F", "ETA_eta_FL", "AETA_eta_FL",
                "ETA_dAlpha_A", "ETA_dAlpha_B",
                "ETA_eta_B", "AETA_eta_B", "ETA_eta_BR", "AETA_eta_BR",
            ):
                result[nm] = nan

        # Epsilon variants (kekulized explicit-H mol, aromatic flags preserved)
        mol_h = Chem.Mol(self.explicit_hydrogen_mol)
        Chem.Kekulize(mol_h)

        eps2 = sum(eps) / n
        eps1 = _eta_eps_mean(mol_h)
        eps5 = _eta_eps_mean5(mol_h)

        rmol_h = self._eta_reference_mol_with_h
        eps3 = _eta_eps_mean(rmol_h) if rmol_h is not None else nan

        sat_mol = self._eta_saturated_mol
        eps4 = _eta_eps_mean(sat_mol) if sat_mol is not None else nan

        result["ETA_epsilon_1"] = eps1
        result["ETA_epsilon_2"] = eps2
        result["ETA_epsilon_3"] = eps3
        result["ETA_epsilon_4"] = eps4
        result["ETA_epsilon_5"] = eps5
        result["ETA_dEpsilon_A"] = eps1 - eps3
        result["ETA_dEpsilon_B"] = eps1 - eps4
        result["ETA_dEpsilon_C"] = eps3 - eps4
        result["ETA_dEpsilon_D"] = eps2 - eps5

        psi1 = alpha_total / (n * eps2) if eps2 != 0.0 else nan
        result["ETA_psi_1"] = psi1
        result["ETA_dPsi_A"] = max(0.714 - psi1, 0.0) if not math.isnan(psi1) else nan
        result["ETA_dPsi_B"] = max(psi1 - 0.714, 0.0) if not math.isnan(psi1) else nan

        alpha_total_safe = alpha_total if alpha_total != 0.0 else nan
        result["ETA_shape_p"] = (
            sum(alpha[a.GetIdx()] for a in mol.GetAtoms() if a.GetDegree() == 1)
            / alpha_total_safe
        )
        result["ETA_shape_y"] = (
            sum(alpha[a.GetIdx()] for a in mol.GetAtoms() if a.GetDegree() == 3)
            / alpha_total_safe
        )
        result["ETA_shape_x"] = (
            sum(alpha[a.GetIdx()] for a in mol.GetAtoms() if a.GetDegree() == 4)
            / alpha_total_safe
        )

        return result

    @cached_property
    def mde_values(self) -> dict[str, float]:
        """Molecular distance edge descriptors (MDEC-*/MDEN-*/MDEO-*)."""
        nan = float("nan")
        D = self.distance_matrix
        sym_map = {"C": 6, "N": 7, "O": 8}

        # Group heavy atom indices by (element_symbol, heavy-atom degree)
        by_elem_deg: dict[tuple[str, int], list[int]] = {}
        for a in self.atoms:
            key = (a.GetSymbol(), a.GetDegree())
            by_elem_deg.setdefault(key, []).append(a.GetIdx())

        result: dict[str, float] = {}
        for name in MDE_DESCRIPTORS:
            sym = name[3]        # 'C', 'N', or 'O'
            v1, v2 = int(name[5]), int(name[6])
            list1 = by_elem_deg.get((sym, v1), [])
            list2 = by_elem_deg.get((sym, v2), [])

            log_sum = 0.0
            count = 0
            if v1 == v2:
                m = len(list1)
                for ii in range(m):
                    for jj in range(ii + 1, m):
                        log_sum += math.log(float(D[list1[ii], list1[jj]]))
                        count += 1
            else:
                for ii in list1:
                    for jj in list2:
                        log_sum += math.log(float(D[ii, jj]))
                        count += 1

            result[name] = count / math.exp(log_sum / count) if count > 0 else nan

        return result

    @cached_property
    def carbon_types_values(self) -> dict[str, float | int]:
        """Carbon hybridization type counts (CarbonTypes) and HybRatio."""
        from collections import defaultdict
        counts: dict[tuple[int | None, int], int] = defaultdict(int)
        for a in self.mol.GetAtoms():
            if a.GetAtomicNum() != 6:
                continue
            sp = _HYBRIDIZATION_SP.get(a.GetHybridization())
            c_nb = sum(nb.GetAtomicNum() == 6 for nb in a.GetNeighbors())
            counts[(sp, c_nb)] += 1
        sp2 = sum(counts[(2, k)] for k in range(5))
        sp3 = sum(counts[(3, k)] for k in range(5))
        hyb_ratio = sp3 / (sp2 + sp3) if (sp2 + sp3) > 0 else float("nan")
        return {
            "C1SP1": counts[(1, 1)], "C2SP1": counts[(1, 2)],
            "C1SP2": counts[(2, 1)], "C2SP2": counts[(2, 2)], "C3SP2": counts[(2, 3)],
            "C1SP3": counts[(3, 1)], "C2SP3": counts[(3, 2)],
            "C3SP3": counts[(3, 3)], "C4SP3": counts[(3, 4)],
            "HybRatio": hyb_ratio,
        }

    @cached_property
    def rncg_rpcg_values(self) -> dict[str, float]:
        """Relative negative/positive charge descriptors (Gasteiger charges)."""
        nan = float("nan")
        charges = np.array(self.gasteiger_charges)
        if np.any(np.isnan(charges)):
            return {"RNCG": nan, "RPCG": nan}
        neg = charges[charges < 0.0]
        pos = charges[charges > 0.0]
        rncg = float(neg[np.argmax(np.abs(neg))] / neg.sum()) if len(neg) > 0 else 0.0
        rpcg = float(pos[np.argmax(np.abs(pos))] / pos.sum()) if len(pos) > 0 else 0.0
        return {"RNCG": rncg, "RPCG": rpcg}

    @cached_property
    def kier_values(self) -> dict[str, float]:
        """Kier kappa shape indices 1-3 (Mordred KappaShapeIndex formula)."""
        nan = float("nan")
        A = len(self.atoms)
        bond_pairs = self.bond_atom_pairs
        P1 = len(bond_pairs)

        # Path counts for orders 2 and 3 reuse the shared chi subgraph enumeration.
        _, _, counts = self.chi_subgraph_accumulators
        P2 = counts[("path", 2)]
        P3 = counts[("path", 3)]

        def _kier(P: int, order: int) -> float:
            if P == 0:
                return nan
            Pmin = A - order
            if order == 1:
                Pmax = 0.5 * A * (A - 1)
                return 2.0 * Pmax * Pmin / (P * P)
            elif order == 2:
                Pmax = 0.5 * (A - 1) * (A - 2)
                return 2.0 * Pmax * Pmin / (P * P)
            else:
                Pmax = 0.25 * (A - 2) ** 2 if A % 2 == 0 else 0.25 * (A - 1) * (A - 3)
                return 4.0 * Pmax * Pmin / (P * P)

        return {"Kier1": _kier(P1, 1), "Kier2": _kier(P2, 2), "Kier3": _kier(P3, 3)}

    @cached_property
    def atomic_id_values(self) -> dict[str, float]:
        """Molecular ID descriptors (MID/AMID family)."""
        nan = float("nan")
        n = len(self.atoms)
        if n == 0 or len(Chem.GetMolFrags(self.mol)) > 1:
            return {name: nan for name in ATOMIC_ID_DESCRIPTORS}

        atomic_ids = _atomic_id_per_atom(self)
        sums = dict.fromkeys(
            ("MID", "MID_h", "MID_C", "MID_N", "MID_O", "MID_X"), 0.0
        )
        for aid, atom in zip(atomic_ids, self.atoms, strict=True):
            atomic_num = atom.GetAtomicNum()
            sums["MID"] += aid
            if atomic_num != 1 and atomic_num != 6:
                sums["MID_h"] += aid
            if atomic_num == 6:
                sums["MID_C"] += aid
            elif atomic_num == 7:
                sums["MID_N"] += aid
            elif atomic_num == 8:
                sums["MID_O"] += aid
            elif atomic_num in _MORDRED_HALOGEN_ATOMIC_NUMBERS:
                sums["MID_X"] += aid
        return {
            **sums,
            "AMID": sums["MID"] / n,
            "AMID_h": sums["MID_h"] / n,
            "AMID_C": sums["MID_C"] / n,
            "AMID_N": sums["MID_N"] / n,
            "AMID_O": sums["MID_O"] / n,
            "AMID_X": sums["MID_X"] / n,
        }


def _average_molecular_weight(ctx: _DescriptorContext) -> float:
    atom_count = ctx.total_atom_count_including_hydrogen
    if atom_count == 0:
        return float("nan")
    return ctx.exact_molecular_weight / atom_count


def _diameter(ctx: _DescriptorContext) -> int:
    if ctx.distance_matrix.size == 0:
        return 0
    return int(ctx.distance_matrix.max())


def _radius(ctx: _DescriptorContext) -> int:
    if ctx.distance_matrix.size == 0:
        return 0
    return int(ctx.distance_matrix.max(axis=0).min())


def _topological_shape_index(ctx: _DescriptorContext) -> float:
    radius = _radius(ctx)
    if radius == 0:
        return float("nan")
    return (_diameter(ctx) - radius) / radius


def _petitjean_index(ctx: _DescriptorContext) -> float:
    diameter = _diameter(ctx)
    if diameter == 0:
        return float("nan")
    return (_diameter(ctx) - _radius(ctx)) / diameter


def _wiener_path_index(ctx: _DescriptorContext) -> int:
    return int(0.5 * ctx.distance_matrix.sum())


def _wiener_polarity_index(ctx: _DescriptorContext) -> int:
    return int(0.5 * (ctx.distance_matrix == 3).sum())


def _zagreb_index_1(ctx: _DescriptorContext) -> float:
    return sum(valence**2 for valence in ctx.adjacency_valences)


def _zagreb_index_2(ctx: _DescriptorContext) -> float:
    valences = ctx.adjacency_valences
    return float(
        sum(
            valences[bond.GetBeginAtomIdx()] * valences[bond.GetEndAtomIdx()]
            for bond in ctx.bonds
        )
    )


def _modified_zagreb_index_1(ctx: _DescriptorContext) -> float:
    if any(valence == 0 for valence in ctx.adjacency_valences):
        return float("nan")
    return sum(valence**-2 for valence in ctx.adjacency_valences)


def _modified_zagreb_index_2(ctx: _DescriptorContext) -> float:
    valences = ctx.adjacency_valences
    return float(
        sum(
            (valences[bond.GetBeginAtomIdx()] * valences[bond.GetEndAtomIdx()]) ** -1
            for bond in ctx.bonds
        )
    )


def _atomic_id_per_atom(ctx: _DescriptorContext) -> list[float]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in ctx.atoms]
    for bond in ctx.bonds:
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        weight = ctx.atoms[begin].GetDegree() * ctx.atoms[end].GetDegree()
        adjacency[begin].append((end, weight))
        adjacency[end].append((begin, weight))

    limit = int(1.0 / (1e-10 ** 2))

    def search(atom_idx: int, current_weight: int, visited: set[int]) -> float:
        value = 0.0
        for next_atom, edge_weight in adjacency[atom_idx]:
            if next_atom in visited:
                continue

            weight = current_weight * edge_weight
            visited.add(next_atom)
            value += 1.0 / math.sqrt(weight)
            if weight < limit:
                value += search(next_atom, weight, visited)
            visited.remove(next_atom)

        return value

    return [
        1.0 + search(atom.GetIdx(), 1, {atom.GetIdx()}) / 2.0
        for atom in ctx.atoms
    ]


def _shortest_path(
    graph: dict[tuple[str, int], set[tuple[str, int]]],
    start: tuple[str, int],
    end: tuple[str, int],
) -> tuple[tuple[str, int], ...] | None:
    queue = [(start, (start,))]
    seen = {start}
    for node, path in queue:
        if node == end:
            return path
        for neighbor in graph.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, (*path, neighbor)))
    return None


def _atom_bond_connectivity_index(ctx: _DescriptorContext) -> float:
    valences = ctx.adjacency_valences
    value = 0.0
    for bond in ctx.bonds:
        begin_valence = valences[bond.GetBeginAtomIdx()]
        end_valence = valences[bond.GetEndAtomIdx()]
        denominator = begin_valence * end_valence
        if denominator:
            value += math.sqrt((begin_valence + end_valence - 2) / denominator)
    return value


def _graovac_ghorbani_index(ctx: _DescriptorContext) -> float:
    distance_matrix = ctx.distance_matrix
    value = 0.0
    for bond in ctx.bonds:
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        begin_count = int(np.sum(distance_matrix[begin, :] < distance_matrix[end, :]))
        end_count = int(np.sum(distance_matrix[end, :] < distance_matrix[begin, :]))
        denominator = begin_count * end_count
        if denominator:
            value += math.sqrt((begin_count + end_count - 2) / denominator)
    return value


def _eccentric_connectivity_index(ctx: _DescriptorContext) -> int:
    if ctx.distance_matrix.size == 0:
        return 0
    return int(
        sum(
            valence * eccentricity
            for valence, eccentricity in zip(
                ctx.adjacency_valences,
                ctx.distance_matrix.max(axis=0),
                strict=True,
            )
        )
    )


def _fragment_complexity(ctx: _DescriptorContext) -> float:
    atom_count = ctx.mol.GetNumAtoms()
    bond_count = ctx.mol.GetNumBonds()
    hetero_atom_count = sum(1 for atom in ctx.atoms if atom.GetAtomicNum() != 6)
    return abs(bond_count**2 - atom_count**2 + atom_count) + hetero_atom_count / 100


def _framework_molecular_fraction(ctx: _DescriptorContext) -> float:
    atom_count = ctx.total_atom_count_including_hydrogen
    if atom_count == 0:
        return float("nan")
    framework_atoms = set().union(*ctx.ring_atom_sets) if ctx.ring_atom_sets else set()
    framework_atoms.update(ctx.framework_linker_atoms)
    return len(framework_atoms) / atom_count


def _atomic_property_value(table: dict[int, float], atomic_num: int) -> float:
    return table.get(atomic_num, float("nan"))


_PERIODIC_TABLE = Chem.GetPeriodicTable()


def _sigma_electron_count(atom: Chem.Atom) -> int:
    """Number of non-hydrogen neighbors (Mordred ``d`` property)."""

    return sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() != 1)


def _valence_electron_count(atom: Chem.Atom) -> float:
    """Kier-Hall valence-electron count (Mordred ``dv`` property)."""

    atomic_num = atom.GetAtomicNum()
    if atomic_num == 1:
        return 0.0
    formal_charge = atom.GetFormalCharge()
    outer = _PERIODIC_TABLE.GetNOuterElecs(atomic_num) - formal_charge
    core = atomic_num - formal_charge
    hydrogens = atom.GetTotalNumHs() + sum(
        1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1
    )
    return (outer - hydrogens) / (core - outer - 1)


def _intrinsic_state_from(atomic_num: int, sigma: float, valence: float) -> float:
    """Electrotopological intrinsic state from precomputed sigma/valence counts."""

    if sigma == 0:
        return float("nan")
    period = _PERIOD_BY_ATOMIC_NUM.get(atomic_num, float("nan"))
    return ((2.0 / period) ** 2 * valence + 1) / sigma


def _ic_expand_tree(tree: dict, visited: set, adj: list) -> None:
    for src, children in tree.items():
        visited.add(src)
        if not children:
            tree[src] = {nb: () for nb in adj[src] if nb not in visited}
        else:
            _ic_expand_tree(children, visited, adj)


def _ic_walk(subtree, before, trail, depth, max_order, order_trails, bonds_dict, atom_info):
    # Single DFS over the depth-``max_order`` BFS tree, collecting the root-to-node
    # trail for every position. A position at depth ``d`` is a frontier leaf of the
    # order-``d`` tree, so it contributes to ``order_trails[d]``; a *natural* leaf
    # (no children in the full tree) stays a leaf at every deeper order, so it also
    # contributes to ``order_trails[d+1 .. max_order]``.
    for src, children in subtree.items():
        if before is None:
            ntrail = (atom_info[src],)
        else:
            ntrail = trail + (bonds_dict[before, src], atom_info[src])
        order_trails[depth].append(ntrail)
        if children:
            _ic_walk(children, src, ntrail, depth + 1, max_order, order_trails, bonds_dict, atom_info)
        else:
            for k in range(depth + 1, max_order + 1):
                order_trails[k].append(ntrail)


def _ic_root_codes(root, adj, bonds_dict, atom_info, max_order):
    # Build the BFS tree once to depth ``max_order`` (each expansion only adds a
    # deeper level, so the depth-``k`` truncation equals an independently built
    # order-``k`` tree), then emit the order 1..max_order codes in a single walk.
    tree: dict = {root: ()}
    visited = {root}
    for _ in range(max_order):
        _ic_expand_tree(tree, visited, adj)
    order_trails: list = [[] for _ in range(max_order + 1)]
    _ic_walk(tree, None, (), 0, max_order, order_trails, bonds_dict, atom_info)
    return [tuple(sorted(order_trails[o])) for o in range(max_order + 1)]


# CarbonTypes hybridization SP mapping (Mordred CarbonTypes.py)
_HYBRIDIZATION_SP: dict[rdchem.HybridizationType, int] = {
    rdchem.HybridizationType.SP:    1,
    rdchem.HybridizationType.SP2:   2,
    rdchem.HybridizationType.SP3:   3,
    rdchem.HybridizationType.SP3D:  3,
    rdchem.HybridizationType.SP3D2: 3,
}

# FilterItLogS SMARTS patterns and coefficients (mordred/LogS.py)
_FILTER_IT_LOGS_SMARTS: tuple[tuple[Chem.Mol, float], ...] = tuple(
    (Chem.MolFromSmarts(smarts), coef)
    for smarts, coef in (
        ("[NH0;X3;v3]",  0.71535),
        ("[NH2;X3;v3]",  0.41056),
        ("[nH0;X3]",     0.82535),
        ("[OH0;X2;v2]",  0.31464),
        ("[OH0;X1;v2]",  0.14787),
        ("[OH1;X2;v2]",  0.62998),
        ("[CH2;!R]",    -0.35634),
        ("[CH3;!R]",    -0.33888),
        ("[CH0;R]",     -0.21912),
        ("[CH2;R]",     -0.23057),
        ("[ch0]",       -0.37570),
        ("[ch1]",       -0.22435),
        ("F",           -0.21728),
        ("Cl",          -0.49721),
        ("Br",          -0.57982),
        ("I",           -0.51547),
    )
)

# Period numbers 1-indexed by atomic number (Z=1→period 1, Z=118→period 7).
_ETA_PERIODS: list[int] = (
    [0] + [1] * 2 + [2] * 8 + [3] * 8 + [4] * 18 + [5] * 18 + [6] * 32 + [7] * 32
)


def _eta_nonsigma_contribution(bond: rdchem.Bond, eps_i: float, eps_j: float) -> float:
    if bond.GetBondType() is rdchem.BondType.SINGLE:
        return 0.0
    f = 2.0 if bond.GetBondTypeAsDouble() == rdchem.BondType.TRIPLE else 1.0
    if bond.GetIsAromatic():
        y = 2.0
    elif abs(eps_i - eps_j) > 0.3:
        y = 1.5
    else:
        y = 1.0
    return f * y


def _eta_build_reference_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """All heavy atoms → C, all bonds → SINGLE; returns None on failure."""
    new = Chem.RWMol()
    ids: dict[int, int] = {}
    for a in mol.GetAtoms():
        ids[a.GetIdx()] = new.AddAtom(Chem.Atom(6))
    for bond in mol.GetBonds():
        ai_a, aj_a = bond.GetBeginAtom(), bond.GetEndAtom()
        if ai_a.GetDegree() > 4 or aj_a.GetDegree() > 4:
            return None
        new.AddBond(ids[ai_a.GetIdx()], ids[aj_a.GetIdx()], rdchem.BondType.SINGLE)
    mol_ref = new.GetMol()
    if Chem.SanitizeMol(mol_ref, catchErrors=True):
        return None
    Chem.Kekulize(mol_ref)
    return mol_ref


def _eta_reference_mol_with_h(mol: Chem.Mol) -> Chem.Mol | None:
    """Reference alkane (all-C/all-SINGLE) with RDKit-assigned H added."""
    mol_ref = _eta_build_reference_mol(mol)
    if mol_ref is None:
        return None
    mol_ref = Chem.AddHs(mol_ref)
    Chem.Kekulize(mol_ref)
    return mol_ref


def _eta_saturated_mol(mol: Chem.Mol) -> Chem.Mol | None:
    """Saturated carbon skeleton: keep original atom types; C-C bonds → SINGLE."""
    new = Chem.RWMol()
    ids: dict[int, int] = {}
    for a in mol.GetAtoms():
        new_a = Chem.Atom(a.GetAtomicNum())
        new_a.SetFormalCharge(a.GetFormalCharge())
        ids[a.GetIdx()] = new.AddAtom(new_a)
    for bond in mol.GetBonds():
        ai_a, aj_a = bond.GetBeginAtom(), bond.GetEndAtom()
        i, j = ids[ai_a.GetIdx()], ids[aj_a.GetIdx()]
        if ai_a.GetAtomicNum() == 6 and aj_a.GetAtomicNum() == 6:
            new.AddBond(i, j, rdchem.BondType.SINGLE)
        else:
            new.AddBond(i, j, bond.GetBondType())
    mol_sat = new.GetMol()
    if Chem.SanitizeMol(mol_sat, catchErrors=True):
        return None
    mol_sat = Chem.AddHs(mol_sat)
    Chem.Kekulize(mol_sat)
    return mol_sat


def _eta_atom_properties(
    mol: Chem.Mol,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Return (alpha, eps, beta_sigma, beta_ns, beta_d) per atom for ETA.

    The mol must be kekulized with aromatic flags preserved so that
    ``GetIsAromatic()`` and ``GetBondType()`` both work correctly.
    """
    pt = _PERIODIC_TABLE
    atoms = list(mol.GetAtoms())
    n = len(atoms)

    alpha: list[float] = []
    eps: list[float] = []
    for a in atoms:
        Z = a.GetAtomicNum()
        Zv = pt.GetNOuterElecs(Z)
        pn = _ETA_PERIODS[Z] if 0 < Z <= 118 else 0
        al = 0.0 if Z == 1 or pn <= 1 else (Z - Zv) / (Zv * (pn - 1))
        alpha.append(al)
        eps.append(0.3 * Zv - al)

    beta_sigma: list[float] = [0.0] * n
    for i, a in enumerate(atoms):
        s = 0.0
        for nb in a.GetNeighbors():
            if nb.GetAtomicNum() == 1:
                continue
            s += 0.5 if abs(eps[i] - eps[nb.GetIdx()]) <= 0.3 else 0.75
        beta_sigma[i] = s

    beta_ns: list[float] = [0.0] * n
    for bond in mol.GetBonds():
        ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if atoms[ai].GetAtomicNum() == 1 or atoms[aj].GetAtomicNum() == 1:
            continue
        c = _eta_nonsigma_contribution(bond, eps[ai], eps[aj])
        beta_ns[ai] += c
        beta_ns[aj] += c

    beta_d: list[float] = [0.0] * n
    for i, a in enumerate(atoms):
        if a.GetIsAromatic() or a.IsInRing():
            continue
        if pt.GetNOuterElecs(a.GetAtomicNum()) <= a.GetTotalValence():
            continue
        for nb in a.GetNeighbors():
            if nb.GetAtomicNum() != 1 and nb.GetIsAromatic():
                beta_d[i] = 0.5
                break

    return alpha, eps, beta_sigma, beta_ns, beta_d


def _eta_composite(gamma: list[float], D: np.ndarray, *, local: bool) -> float:
    total = 0.0
    n = len(gamma)
    for i in range(n):
        gi = gamma[i]
        if math.isnan(gi):
            continue
        for j in range(i + 1, n):
            r = float(D[i, j])
            if local:
                if r != 1.0:
                    continue
            elif r == 0.0:
                continue
            gj = gamma[j]
            if math.isnan(gj):
                continue
            total += math.sqrt(gi * gj) / r
    return total


def _eta_eps_mean(mol: Chem.Mol) -> float:
    """Mean ETA epsilon over all atoms in mol."""
    pt = _PERIODIC_TABLE
    total = 0.0
    count = 0
    for a in mol.GetAtoms():
        Z = a.GetAtomicNum()
        Zv = pt.GetNOuterElecs(Z)
        pn = _ETA_PERIODS[Z] if 0 < Z <= 118 else 0
        al = 0.0 if Z == 1 or pn <= 1 else (Z - Zv) / (Zv * (pn - 1))
        total += 0.3 * Zv - al
        count += 1
    return total / count if count > 0 else float("nan")


def _eta_eps_mean5(mol_with_h: Chem.Mol) -> float:
    """Mean ETA epsilon over heavy atoms + H bonded to heteroatoms (type 5)."""
    pt = _PERIODIC_TABLE
    total = 0.0
    count = 0
    for a in mol_with_h.GetAtoms():
        Z = a.GetAtomicNum()
        if Z == 1:
            nbs = a.GetNeighbors()
            if nbs and nbs[0].GetAtomicNum() == 6:
                continue
        Zv = pt.GetNOuterElecs(Z)
        pn = _ETA_PERIODS[Z] if 0 < Z <= 118 else 0
        al = 0.0 if Z == 1 or pn <= 1 else (Z - Zv) / (Zv * (pn - 1))
        total += 0.3 * Zv - al
        count += 1
    return total / count if count > 0 else float("nan")


def _compute_detour_matrix(
    n: int, bond_pairs: tuple[tuple[int, int], ...]
) -> np.ndarray | None:
    """Longest-simple-path distance matrix (heavy atoms, unweighted).

    Returns None for disconnected multi-atom molecules (Mordred
    ``require_connected = True`` → Missing).  For n==0 also returns None.
    Single-atom molecules return a 1×1 zero matrix so that most Dt descriptors
    can be computed normally; the caller special-cases SM1_Dt for n==1 because
    Mordred returns np.int64(0) there (which the test's isinstance check treats
    as non-numeric → expects NaN from our side).
    """
    if n == 0:
        return None
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in bond_pairs:
        adj[i].append(j)
        adj[j].append(i)

    # BFS connectivity check
    seen: set[int] = {0}
    queue = [0]
    for node in queue:
        for nb in adj[node]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    if len(seen) != n:
        return None

    D = np.zeros((n, n), dtype=float)
    visited = bytearray(n)

    def dfs(u: int, dist: int, max_dist: list[int]) -> None:
        for v in adj[u]:
            if not visited[v]:
                new_d = dist + 1
                if new_d > max_dist[v]:
                    max_dist[v] = new_d
                visited[v] = 1
                dfs(v, new_d, max_dist)
                visited[v] = 0

    for src in range(n):
        max_dist = [0] * n
        visited[src] = 1
        dfs(src, 0, max_dist)
        visited[src] = 0
        D[src] = max_dist

    return np.maximum(D, D.T)


def _compute_barysz_matrices(
    n: int,
    bond_pairs: tuple[tuple[int, int], ...],
    bond_orders: tuple[float, ...],
    jobs: list[tuple[str, np.ndarray, float]],
) -> np.ndarray:
    """Build all Barysz (Dz) matrices in one batched Floyd-Warshall.

    Each job's edge weight is ``C² / (P[i] · P[j] · π_ij)`` (C = carbon
    reference) and its diagonal is ``1 − C/P[i]``, filled after the
    shortest-path relaxation.  The ``k`` matrices share one ``(k, n, n)``
    relaxation so the per-call numpy overhead is paid once for all properties
    instead of once each.
    """
    k = len(jobs)
    # Initialise with a large sentinel (not inf, to keep the additions finite).
    large = 1e15
    w = np.full((k, n, n), large, dtype=float)
    diag_idx = np.arange(n)
    w[:, diag_idx, diag_idx] = 0.0

    for slot, (_suffix, prop_vals, carbon_ref) in enumerate(jobs):
        cc = carbon_ref * carbon_ref
        for (i, j), bo in zip(bond_pairs, bond_orders):
            edge_w = cc / (prop_vals[i] * prop_vals[j] * bo)
            w[slot, i, j] = edge_w
            w[slot, j, i] = edge_w

    # Batched Floyd-Warshall: relax all k matrices against pivot k at once.
    for pivot in range(n):
        candidate = w[:, :, pivot : pivot + 1] + w[:, pivot : pivot + 1, :]
        np.minimum(w, candidate, out=w)

    # Replace any remaining sentinel with 0 (guards against disconnected pairs).
    w[w >= large] = 0.0

    # Fill each matrix's diagonal with 1 − C/P[i].
    for slot, (_suffix, prop_vals, carbon_ref) in enumerate(jobs):
        w[slot, diag_idx, diag_idx] = 1.0 - carbon_ref / prop_vals

    return w


def _fill_spectral_batch(
    results: dict[str, float],
    specs: list[tuple[str, np.ndarray, bool]],
    bond_pairs: tuple[tuple[int, int], ...],
    n: int,
) -> None:
    """Compute every spectral aggregate for a batch of matrices.

    All matrices share the molecule's ``n`` so they are stacked into one
    ``(k, n, n)`` array and diagonalized with a single ``eigh`` call (matching
    the BCUT batching pattern); the per-row aggregates are then vectorized
    across the batch.  ``eigh`` returns ascending eigenvalues, so the leading
    eigenvalue/eigenvector is always at index ``n−1``.
    """
    if not specs:
        return

    nan = float("nan")
    suffixes = [s for s, _, _ in specs]
    batch = np.stack([m for _, m, _ in specs])

    # SM1 (trace) needs no eigendecomposition; take all traces in one reduction.
    sm1 = np.trace(batch, axis1=1, axis2=2).tolist()
    for idx, (suffix, _matrix, include_sm1) in enumerate(specs):
        if include_sm1:
            results[f"SM1_{suffix}"] = sm1[idx]

    try:
        # eigvals: (k, n) ascending; eigvecs: (k, n, n).
        eigvals, eigvecs = np.linalg.eigh(batch)
    except np.linalg.LinAlgError:
        for suffix in suffixes:
            for m in _SPECTRAL_METHODS:
                results[f"{m}_{suffix}"] = nan
        return

    if np.iscomplexobj(eigvals):
        eigvals = eigvals.real
    if np.iscomplexobj(eigvecs):
        eigvecs = eigvecs.real

    max_eig = eigvals[:, -1]            # (k,)
    min_eig = eigvals[:, 0]             # (k,)
    sp_abs = np.abs(eigvals).sum(axis=1)
    mean_eig = eigvals.mean(axis=1)
    sp_ad = np.abs(eigvals - mean_eig[:, None]).sum(axis=1)
    sp_diam = max_eig - min_eig

    # LogEE via log-sum-exp, per row: a = max(λ_max, 0).
    a = np.maximum(max_eig, 0.0)
    sx = np.exp(eigvals - a[:, None]).sum(axis=1) + np.exp(-a)
    log_ee = a + np.log(sx)

    # Leading eigenvector per matrix: (k, n).
    lead = eigvecs[:, :, -1]
    ve1 = np.abs(lead).sum(axis=1)
    val_ve3 = 0.1 * n * ve1
    with np.errstate(divide="ignore", invalid="ignore"):
        ve3 = np.where(val_ve3 > 0.0, np.log(val_ve3), nan)

    # VR1 = Σ (v_i · v_j)^{−½} over bonds; NaN if any bond product ≤ 0.
    if bond_pairs:
        ai = np.fromiter((p[0] for p in bond_pairs), dtype=int, count=len(bond_pairs))
        aj = np.fromiter((p[1] for p in bond_pairs), dtype=int, count=len(bond_pairs))
        prods = lead[:, ai] * lead[:, aj]              # (k, n_bonds)
        bad = (prods <= 0.0).any(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            vr1 = np.where(bad, nan, (prods ** -0.5).sum(axis=1))
    else:
        vr1 = np.zeros(len(specs))
    val_vr3 = 0.1 * n * vr1
    with np.errstate(divide="ignore", invalid="ignore"):
        vr3 = np.where(val_vr3 > 0.0, np.log(val_vr3), nan)

    # Convert each aggregate to a Python list once (in C) instead of extracting
    # ~12·k numpy scalars one at a time — the same lesson as the autocorrelation
    # pass: per-element float() dominates once the array math is cheap.
    sp_abs_l = sp_abs.tolist()
    max_eig_l = max_eig.tolist()
    sp_diam_l = sp_diam.tolist()
    sp_ad_l = sp_ad.tolist()
    sp_mad_l = (sp_ad / n).tolist()
    log_ee_l = log_ee.tolist()
    ve1_l = ve1.tolist()
    ve2_l = (ve1 / n).tolist()
    ve3_l = ve3.tolist()
    vr1_l = vr1.tolist()
    vr2_l = (vr1 / n).tolist()
    vr3_l = vr3.tolist()

    for idx, suffix in enumerate(suffixes):
        results[f"SpAbs_{suffix}"] = sp_abs_l[idx]
        results[f"SpMax_{suffix}"] = max_eig_l[idx]
        results[f"SpDiam_{suffix}"] = sp_diam_l[idx]
        results[f"SpAD_{suffix}"] = sp_ad_l[idx]
        results[f"SpMAD_{suffix}"] = sp_mad_l[idx]
        results[f"LogEE_{suffix}"] = log_ee_l[idx]
        results[f"VE1_{suffix}"] = ve1_l[idx]
        results[f"VE2_{suffix}"] = ve2_l[idx]
        results[f"VE3_{suffix}"] = ve3_l[idx]
        results[f"VR1_{suffix}"] = vr1_l[idx]
        results[f"VR2_{suffix}"] = vr2_l[idx]
        results[f"VR3_{suffix}"] = vr3_l[idx]


def _classify_chi_subgraph(
    bond_endpoints: tuple[tuple[int, int], ...], use_bonds: tuple[int, ...]
) -> tuple[str, list[int]]:
    """Classify a connected bond-subgraph into its Mordred chi type.

    Returns (chi_type, [atom_idx]) where chi_type is one of 'path',
    'cluster', 'path_cluster', or 'chain'. Reproduces Mordred's recursive
    DFS classifier exactly, but without any traversal:

    ``FindAllSubgraphsOfLengthN`` yields *connected* subgraphs of exactly
    ``order`` bonds (E == order). A connected graph has a cycle iff
    E >= V, so the subgraph is a ``chain`` iff its distinct-node count is
    ``<= order`` (equivalently, it is a tree with ``order + 1`` nodes
    otherwise). For the tree case the type follows directly from the
    in-subgraph degree multiset: all degrees <= 2 is a ``path``; a degree-2
    node alongside a branch point is a ``path_cluster``; otherwise (only
    terminal and branch nodes, no degree 2) it is a ``cluster``.
    """
    degree: dict[int, int] = {}
    for bi in use_bonds:
        a, b = bond_endpoints[bi]
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    nodes = list(degree)
    if len(nodes) <= len(use_bonds):
        return "chain", nodes

    degrees = set(degree.values())
    if max(degrees) <= 2:
        return "path", nodes
    if 2 in degrees:
        return "path_cluster", nodes
    return "cluster", nodes


def _atomic_polarizability(ctx: _DescriptorContext) -> float:
    return sum(
        _atomic_property_value(_POLARIZABILITY_94_BY_ATOMIC_NUM, atom.GetAtomicNum())
        for atom in ctx.explicit_hydrogen_mol.GetAtoms()
    )


def _bond_polarizability(ctx: _DescriptorContext) -> float:
    value = 0.0
    for bond in ctx.explicit_hydrogen_mol.GetBonds():
        begin = bond.GetBeginAtom().GetAtomicNum()
        end = bond.GetEndAtom().GetAtomicNum()
        begin_pol = _atomic_property_value(_POLARIZABILITY_94_BY_ATOMIC_NUM, begin)
        end_pol = _atomic_property_value(_POLARIZABILITY_94_BY_ATOMIC_NUM, end)
        value += abs(begin_pol - end_pol)
    return value


def _mcgowan_volume(ctx: _DescriptorContext) -> float:
    mol = ctx.explicit_hydrogen_mol
    atom_sum = sum(
        _atomic_property_value(_MCGOWAN_VOLUME_BY_ATOMIC_NUM, atom.GetAtomicNum())
        for atom in mol.GetAtoms()
    )
    return atom_sum - mol.GetNumBonds() * 6.56


def _vabc_volume(ctx: _DescriptorContext) -> float:
    mol = ctx.explicit_hydrogen_mol
    atom_contribution = sum(
        _atomic_property_value(
            _VABC_ATOM_CONTRIBUTION_BY_ATOMIC_NUM,
            atom.GetAtomicNum(),
        )
        for atom in mol.GetAtoms()
    )
    aromatic_ring_count = _ring_metadata_count(
        ctx,
        size=None,
        fused=None,
        aromaticity="a",
        hetero=None,
    )
    aliphatic_ring_count = _ring_metadata_count(
        ctx,
        size=None,
        fused=None,
        aromaticity="A",
        hetero=None,
    )
    return (
        atom_contribution
        - 5.92 * mol.GetNumBonds()
        - 14.7 * aromatic_ring_count
        - 3.8 * aliphatic_ring_count
    )


def _hydrogen_atom_count(ctx: _DescriptorContext) -> int:
    return sum(
        1 if atom.GetAtomicNum() == 1 else atom.GetTotalNumHs()
        for atom in ctx.atoms
    )


def _rotatable_bond_ratio(ctx: _DescriptorContext) -> float:
    bond_count = len(ctx.bonds)
    if bond_count == 0:
        return float("nan")
    return rdMolDescriptors.CalcNumRotatableBonds(ctx.mol) / bond_count


def _autocorrelation_descriptor(name: str) -> DescriptorFunction:
    if _AUTOCORRELATION_PATTERN.match(name) is None:
        msg = f"unsupported autocorrelation descriptor: {name}"
        raise ValueError(msg)

    return lambda ctx: ctx.autocorrelation_values[name]


def _bcut_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.bcut_values[name]


def _constitutional_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.constitutional_values[name]


def _topological_charge_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.topological_charge_values[name]


def _chi_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.chi_values[name]


def _spectral_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.spectral_values[name]


def _information_content_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.information_content_values[name]


def _atom_count_by_symbol(symbol: str) -> DescriptorFunction:
    return lambda ctx: ctx.atom_symbol_counts.get(symbol, 0)


def _halogen_atom_count(ctx: _DescriptorContext) -> int:
    return ctx.halogen_count


def _aromatic_atom_count(ctx: _DescriptorContext) -> int:
    return ctx.aromatic_atom_count


def _estate_atom_type_count(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.estate_atom_type_counts[name]


def _estate_atom_type_maxmin(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.estate_atom_type_agg[name]


def _estate_atom_type_sum(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.estate_atom_type_agg[name]


def _smarts_count_descriptor(smarts: tuple[str, ...]) -> DescriptorFunction:
    pattern = Chem.MolFromSmarts("[" + ",".join(f"$({value})" for value in smarts) + "]")
    if pattern is None:
        msg = f"invalid SMARTS pattern set: {smarts!r}"
        raise ValueError(msg)

    return lambda ctx: len(ctx.mol.GetSubstructMatches(pattern))


def _is_single_bond(bond: rdchem.Bond) -> bool:
    return bond.GetBondType() == rdchem.BondType.SINGLE


def _is_double_bond(bond: rdchem.Bond) -> bool:
    return bond.GetBondType() == rdchem.BondType.DOUBLE


def _is_triple_bond(bond: rdchem.Bond) -> bool:
    return bond.GetBondType() == rdchem.BondType.TRIPLE


def _is_aromatic_bond(bond: rdchem.Bond) -> bool:
    return bond.GetIsAromatic()


def _implicit_hydrogen_bond_count(ctx: _DescriptorContext) -> int:
    return ctx.implicit_hydrogen_count


def _bond_count(ctx: _DescriptorContext) -> int:
    return len(ctx.bonds) + ctx.implicit_hydrogen_count


def _bond_count_by_predicate(
    ctx: _DescriptorContext,
    predicate: Callable[[rdchem.Bond], bool],
    *,
    include_implicit_hydrogen_bonds: bool = False,
) -> int:
    count = sum(1 for bond in ctx.bonds if predicate(bond))
    if include_implicit_hydrogen_bonds:
        count += ctx.implicit_hydrogen_count
    return count


def _multiple_bond_count(ctx: _DescriptorContext) -> int:
    return _bond_count_by_predicate(
        ctx,
        lambda bond: _is_double_bond(bond)
        or _is_triple_bond(bond)
        or _is_aromatic_bond(bond),
    )


def _kekulized_bond_count(
    ctx: _DescriptorContext,
    bond_type: rdchem.BondType,
    *,
    include_implicit_hydrogen_bonds: bool = False,
) -> int:
    count = sum(
        1 for bond in ctx.kekulized_mol.GetBonds() if bond.GetBondType() == bond_type
    )
    if include_implicit_hydrogen_bonds:
        count += ctx.implicit_hydrogen_count
    return count


def _ring_matches_filters(
    ctx: _DescriptorContext,
    atoms: set[int],
    size: str | None,
    aromaticity: str | None,
    hetero: str | None,
) -> bool:
    if size == "G12":
        if len(atoms) < 12:
            return False
    elif size is not None and len(atoms) != int(size):
        return False

    if aromaticity is not None:
        is_aromatic = all(ctx.mol.GetAtomWithIdx(atom).GetIsAromatic() for atom in atoms)
        if aromaticity == "a" and not is_aromatic:
            return False
        if aromaticity == "A" and is_aromatic:
            return False

    if hetero is not None and not any(
        ctx.mol.GetAtomWithIdx(atom).GetAtomicNum() != 6 for atom in atoms
    ):
        return False

    return True


def _ring_metadata_matches_filters(
    metadata: _RingMetadata,
    size: str | None,
    aromaticity: str | None,
    hetero: str | None,
) -> bool:
    if size == "G12":
        if metadata.size < 12:
            return False
    elif size is not None and metadata.size != int(size):
        return False

    if aromaticity == "a" and not metadata.is_aromatic:
        return False
    if aromaticity == "A" and metadata.is_aromatic:
        return False

    if hetero is not None and not metadata.has_hetero:
        return False

    return True


def _ring_metadata_count(
    ctx: _DescriptorContext,
    size: str | None,
    fused: str | None,
    aromaticity: str | None,
    hetero: str | None,
) -> int:
    metadata_values = ctx.fused_ring_metadata if fused is not None else ctx.ring_metadata
    return sum(
        1
        for metadata in metadata_values
        if _ring_metadata_matches_filters(metadata, size, aromaticity, hetero)
    )


def _ring_count_descriptor(name: str) -> DescriptorFunction:
    match = _RING_COUNT_PATTERN.match(name)
    if match is None:
        msg = f"unsupported ring count descriptor: {name}"
        raise ValueError(msg)

    size, fused, aromaticity, hetero = match.groups()

    def calc(ctx: _DescriptorContext) -> int:
        return _ring_metadata_count(ctx, size, fused, aromaticity, hetero)

    return calc


def _path_count_descriptor(name: str) -> DescriptorFunction:
    match = _PATH_COUNT_PATTERN.match(name)
    if match is None:
        msg = f"unsupported path count descriptor: {name}"
        raise ValueError(msg)

    total_prefix, pi_prefix, order_text = match.groups()
    order = int(order_text)
    use_pi = pi_prefix is not None
    total = total_prefix is not None

    def raw_value_for_order(ctx: _DescriptorContext, path_order: int) -> int | float:
        if path_order == 0:
            return ctx.mol.GetNumAtoms()
        path_count, pi_path_count = ctx.path_count(path_order)
        return pi_path_count if use_pi else path_count

    def value_for_order(ctx: _DescriptorContext, path_order: int) -> int | float:
        value = raw_value_for_order(ctx, path_order)
        if use_pi:
            return math.log(value + 1)
        return value

    def calc(ctx: _DescriptorContext) -> int | float:
        if total:
            value = sum(
                raw_value_for_order(ctx, path_order)
                for path_order in range(0, order + 1)
            )
            if use_pi:
                return math.log(value + 1)
            return value
        return value_for_order(ctx, order)

    return calc


def _walk_count_descriptor(name: str) -> DescriptorFunction:
    match = _WALK_COUNT_PATTERN.match(name)
    if match is None:
        msg = f"unsupported walk count descriptor: {name}"
        raise ValueError(msg)

    total_prefix, molecular_walk, self_returning_walk, order_text = match.groups()
    order = int(order_text)
    total = total_prefix is not None
    self_returning = self_returning_walk is not None

    def value_for_order(ctx: _DescriptorContext, walk_order: int) -> float:
        matrix_power = ctx.adjacency_power(walk_order)
        if self_returning:
            return math.log(matrix_power.trace() + 1)
        if walk_order == 1:
            return 0.5 * matrix_power.sum()
        return math.log(matrix_power.sum() + 1)

    def calc(ctx: _DescriptorContext) -> float:
        if total:
            values = [float(ctx.mol.GetNumAtoms())]
            start = 2 if self_returning else 1
            values.extend(
                value_for_order(ctx, walk_order)
                for walk_order in range(start, order + 1)
            )
            return sum(values)
        return value_for_order(ctx, order)

    return calc


def _rdkit_descriptor(
    function: Callable[[Chem.Mol], DescriptorValue],
) -> DescriptorFunction:
    return lambda ctx: function(ctx.mol)


_DESCRIPTOR_FUNCTIONS: dict[str, DescriptorFunction] = {
    "ABC": _atom_bond_connectivity_index,
    "ABCGG": _graovac_ghorbani_index,
    "AMW": _average_molecular_weight,
    "BertzCT": _rdkit_descriptor(Descriptors.BertzCT),
    "Diameter": _diameter,
    "ECIndex": _eccentric_connectivity_index,
    "FCSP3": _rdkit_descriptor(rdMolDescriptors.CalcFractionCSP3),
    "MW": lambda ctx: ctx.exact_molecular_weight,
    "PetitjeanIndex": _petitjean_index,
    "Radius": _radius,
    "RotRatio": _rotatable_bond_ratio,
    "SMR": lambda ctx: ctx.mol_mr,
    "SLogP": lambda ctx: ctx.mol_log_p,
    "TopoPSA": lambda ctx: rdMolDescriptors.CalcTPSA(
        ctx.mol,
        includeSandP=True,
    ),
    "TopoPSA(NO)": _rdkit_descriptor(rdMolDescriptors.CalcTPSA),
    "TopoShapeIndex": _topological_shape_index,
    "VMcGowan": _mcgowan_volume,
    "Vabc": _vabc_volume,
    "WPath": _wiener_path_index,
    "WPol": _wiener_polarity_index,
    "Zagreb1": _zagreb_index_1,
    "Zagreb2": _zagreb_index_2,
    "apol": _atomic_polarizability,
    "bpol": _bond_polarizability,
    "fMF": _framework_molecular_fraction,
    "fragCpx": _fragment_complexity,
    "mZagreb1": _modified_zagreb_index_1,
    "mZagreb2": _modified_zagreb_index_2,
    "nAcid": _smarts_count_descriptor(_ACID_GROUP_SMARTS),
    "nAromAtom": _aromatic_atom_count,
    "nAromBond": lambda ctx: _bond_count_by_predicate(ctx, _is_aromatic_bond),
    "nAtom": lambda ctx: ctx.total_atom_count_including_hydrogen,
    "nBase": _smarts_count_descriptor(_BASE_GROUP_SMARTS),
    "nBridgehead": _rdkit_descriptor(rdMolDescriptors.CalcNumBridgeheadAtoms),
    "nBonds": _bond_count,
    "nBondsA": lambda ctx: _bond_count_by_predicate(ctx, _is_aromatic_bond),
    "nBondsD": lambda ctx: _bond_count_by_predicate(ctx, _is_double_bond),
    "nBondsKD": lambda ctx: _kekulized_bond_count(ctx, rdchem.BondType.DOUBLE),
    "nBondsKS": lambda ctx: _kekulized_bond_count(
        ctx,
        rdchem.BondType.SINGLE,
        include_implicit_hydrogen_bonds=True,
    ),
    "nBondsM": _multiple_bond_count,
    "nBondsO": lambda ctx: len(ctx.bonds),
    "nBondsS": lambda ctx: _bond_count_by_predicate(
        ctx,
        _is_single_bond,
        include_implicit_hydrogen_bonds=True,
    ),
    "nBondsT": lambda ctx: _bond_count_by_predicate(ctx, _is_triple_bond),
    "nHBAcc": lambda ctx: ctx.num_hba,
    "nHBDon": lambda ctx: ctx.num_hbd,
    "nH": _hydrogen_atom_count,
    "nHeavyAtom": _rdkit_descriptor(Descriptors.HeavyAtomCount),
    "nHetero": _rdkit_descriptor(rdMolDescriptors.CalcNumHeteroatoms),
    "nRot": _rdkit_descriptor(rdMolDescriptors.CalcNumRotatableBonds),
    "nSpiro": _rdkit_descriptor(rdMolDescriptors.CalcNumSpiroAtoms),
    "nX": _halogen_atom_count,
}

for _name, _symbol in _ATOM_SYMBOLS_BY_DESCRIPTOR.items():
    _DESCRIPTOR_FUNCTIONS[_name] = _atom_count_by_symbol(_symbol)

for _name in ESTATE_ATOM_TYPE_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _estate_atom_type_count(_name)

for _name in ESTATE_ATOM_TYPE_MAXMIN_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _estate_atom_type_maxmin(_name)

for _name in ESTATE_ATOM_TYPE_SUM_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _estate_atom_type_sum(_name)

for _name in RING_COUNT_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _ring_count_descriptor(_name)

for _name in PATH_COUNT_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _path_count_descriptor(_name)

for _name in WALK_COUNT_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _walk_count_descriptor(_name)

for _name in AUTOCORRELATION_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _autocorrelation_descriptor(_name)

for _name in BCUT_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _bcut_descriptor(_name)

for _name in CONSTITUTIONAL_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _constitutional_descriptor(_name)

for _name in TOPOLOGICAL_CHARGE_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _topological_charge_descriptor(_name)

for _name in CHI_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _chi_descriptor(_name)

for _name in SPECTRAL_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _spectral_descriptor(_name)

for _name in INFORMATION_CONTENT_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _information_content_descriptor(_name)


def _eta_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.eta_values[name]


def _mde_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.mde_values[name]


for _name in ETA_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _eta_descriptor(_name)

for _name in MDE_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = _mde_descriptor(_name)


def _carbon_types_descriptor(name: str) -> DescriptorFunction:
    return lambda ctx: ctx.carbon_types_values[name]


for _name in (*CARBON_TYPES_DESCRIPTORS, "HybRatio"):
    _DESCRIPTOR_FUNCTIONS[_name] = _carbon_types_descriptor(_name)

for _name in ATOMIC_ID_DESCRIPTORS:
    _DESCRIPTOR_FUNCTIONS[_name] = (lambda n: lambda ctx: ctx.atomic_id_values[n])(_name)

for _name in ("RNCG", "RPCG"):
    _DESCRIPTOR_FUNCTIONS[_name] = (lambda n: lambda ctx: ctx.rncg_rpcg_values[n])(_name)


def _vadjmat(ctx: _DescriptorContext) -> float:
    m = sum(
        1 for b in ctx.bonds
        if b.GetBeginAtom().GetAtomicNum() != 1 and b.GetEndAtom().GetAtomicNum() != 1
    )
    return math.log2(m) + 1.0 if m > 0 else float("nan")


def _lipinski(ctx: _DescriptorContext) -> int:
    return int(
        ctx.num_hbd <= 5
        and ctx.num_hba <= 10
        and ctx.exact_molecular_weight <= 500.0
        and ctx.mol_log_p <= 5.0
    )


def _ghose_filter(ctx: _DescriptorContext) -> int:
    n = ctx.explicit_hydrogen_mol.GetNumAtoms()
    return int(
        160 <= ctx.exact_molecular_weight <= 480
        and 20 <= n <= 70
        and -0.4 <= ctx.mol_log_p <= 5.6
        and 40 <= ctx.mol_mr <= 130
    )


def _filter_it_logs(ctx: _DescriptorContext) -> float:
    mol = ctx.mol
    mw = Descriptors.MolWt(mol)
    logS = 0.89823 - 0.10369 * math.sqrt(mw)
    for smarts_mol, coef in _FILTER_IT_LOGS_SMARTS:
        logS += len(mol.GetSubstructMatches(smarts_mol)) * coef
    return logS


def _detour_index(ctx: _DescriptorContext) -> float:
    dt = ctx.detour_matrix
    return int(0.5 * dt.sum()) if dt is not None else float("nan")


def _balaban_j(ctx: _DescriptorContext) -> float:
    from rdkit.Chem import GraphDescriptors
    return float(GraphDescriptors.BalabanJ(ctx.mol, dMat=ctx.distance_matrix))


_DESCRIPTOR_FUNCTIONS["VAdjMat"] = _vadjmat
_DESCRIPTOR_FUNCTIONS["Lipinski"] = _lipinski
_DESCRIPTOR_FUNCTIONS["GhoseFilter"] = _ghose_filter
_DESCRIPTOR_FUNCTIONS["FilterItLogS"] = _filter_it_logs
_DESCRIPTOR_FUNCTIONS["DetourIndex"] = _detour_index
_DESCRIPTOR_FUNCTIONS["BalabanJ"] = _balaban_j
for _name in ("Kier1", "Kier2", "Kier3"):
    _DESCRIPTOR_FUNCTIONS[_name] = (lambda n: lambda ctx: ctx.kier_values[n])(_name)

for _name in SMALL_GRAPH_FORMULA_DESCRIPTORS:
    if _name not in _DESCRIPTOR_FUNCTIONS:
        msg = f"missing small graph/formula implementation: {_name}"
        raise RuntimeError(msg)

for _name in PHYSICAL_PROPERTY_DESCRIPTORS:
    if _name not in _DESCRIPTOR_FUNCTIONS:
        msg = f"missing physical-property implementation: {_name}"
        raise RuntimeError(msg)

for _name in SUPPORTED_MORDRED_2D_DESCRIPTORS:
    if _name not in _DESCRIPTOR_FUNCTIONS and hasattr(Descriptors, _name):
        _DESCRIPTOR_FUNCTIONS[_name] = _rdkit_descriptor(getattr(Descriptors, _name))


def calc_rdkit_mordred_like_2d(
    mol: Chem.Mol,
    names: Iterable[str] | None = None,
) -> dict[str, DescriptorValue]:
    """Return Mordred-name 2D descriptors computed using RDKit only.

    By default every supported descriptor is computed. Pass ``names`` to
    compute only a subset; because each descriptor family is a separately
    cached calculation on the shared per-molecule context, requesting a
    subset skips the work of any family no requested descriptor touches.
    The returned dict preserves the requested order.
    """

    if mol is None:
        msg = "mol must be an RDKit Mol, not None"
        raise ValueError(msg)

    if names is None:
        requested: tuple[str, ...] = SUPPORTED_MORDRED_2D_DESCRIPTORS
    else:
        requested = tuple(names)
        unknown = [name for name in requested if name not in _DESCRIPTOR_FUNCTIONS]
        if unknown:
            msg = f"unsupported descriptor name(s): {sorted(set(unknown))}"
            raise KeyError(msg)

    context = _DescriptorContext(mol)
    return {name: _DESCRIPTOR_FUNCTIONS[name](context) for name in requested}
