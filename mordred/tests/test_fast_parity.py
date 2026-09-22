"""Parity, structural, and parallel tests for the fast RDKit-only descriptor engine.

These tests are the regression net for the fast path (Calculator(use_fast=True))
against the slow path (Calculator(use_fast=False)).  They must stay green whenever:
  - a descriptor is migrated from the fast path into its Descriptor class body, or
  - the vendored engine in mordred/_fast/ is synced from upstream.
"""

import math
import os
import pickle

import pytest
from rdkit.Chem import AllChem as Chem
from numpy.testing import assert_almost_equal

from mordred import Calculator, descriptors
from mordred._fast import FAST_ENABLED, _DESCRIPTOR_FUNCTIONS
from mordred.error import MissingValueBase

_DATA_FILE = os.path.join(os.path.dirname(__file__), "references", "structures.sdf")

_EXTRA_SMILES = [
    # varied chemistry not guaranteed to be in the SDF
    "CCO",
    "CC(=O)O",
    "CCN(CC)CC",
    "c1ccccc1",
    "c1ccncc1",
    "O=C(O)c1ccccc1",
    "CCOC(=O)c1ccccc1",
    "C1CCCCC1",
    "CC(C)(C)c1ccc(O)cc1",
    "O=C(N)c1ccccc1",
    "CCS(=O)(=O)C",
    "C[N+](C)(C)C",
    "O=[N+]([O-])c1ccccc1",
    "C1CCC2CCCCC2C1",
    "C1OC1",
    "C1COCCO1",
    "C.C",  # disconnected — exercises require_connected gate
]


def _load_molecules():
    mols = []
    supplier = Chem.SDMolSupplier(_DATA_FILE, removeHs=False)
    for m in supplier:
        if m is not None:
            mols.append(m)
    for smi in _EXTRA_SMILES:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
    return mols


@pytest.fixture(scope="module")
def molecules():
    return _load_molecules()


@pytest.fixture(scope="module")
def calc_fast():
    return Calculator(descriptors, ignore_3D=True)


@pytest.fixture(scope="module")
def calc_slow():
    return Calculator(descriptors, ignore_3D=True, use_fast=False)


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_fast_structural():
    """FAST_ENABLED must be a subset of both the registry and mordred 2D names."""
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_2d_names = {str(d) for d in calc.descriptors}

    unknown_in_registry = FAST_ENABLED - set(_DESCRIPTOR_FUNCTIONS)
    assert not unknown_in_registry, (
        "FAST_ENABLED contains names absent from _DESCRIPTOR_FUNCTIONS: "
        + str(sorted(unknown_in_registry)[:20])
    )

    unknown_in_mordred = FAST_ENABLED - mordred_2d_names
    assert not unknown_in_mordred, (
        "FAST_ENABLED contains names absent from mordred 2D descriptors: "
        + str(sorted(unknown_in_mordred)[:20])
    )


def test_fast_covers_all_2d():
    """All mordred 2D descriptors should be in FAST_ENABLED (until migrated out)."""
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_2d_names = {str(d) for d in calc.descriptors}
    missing_from_fast = mordred_2d_names - FAST_ENABLED
    assert not missing_from_fast, (
        "These mordred 2D descriptors are not in FAST_ENABLED "
        "(add them or document the omission): "
        + str(sorted(missing_from_fast)[:20])
    )


# ---------------------------------------------------------------------------
# Value parity
# ---------------------------------------------------------------------------

def _compare_results(fast_results, slow_results, desc_names, mol_label):
    mismatches = []
    for name, f, s in zip(desc_names, fast_results, slow_results):
        f_missing = isinstance(f, MissingValueBase)
        s_missing = isinstance(s, MissingValueBase)

        if f_missing != s_missing:
            mismatches.append(
                "{} ({}): fast_missing={} slow_missing={}".format(
                    name, mol_label, f_missing, s_missing
                )
            )
            continue

        if f_missing:
            continue  # both Missing — agree on "undefined"

        try:
            fv = float(f)
            sv = float(s)
        except (TypeError, ValueError):
            continue

        if math.isnan(fv) != math.isnan(sv):
            mismatches.append(
                "{} ({}): NaN mismatch fast={} slow={}".format(name, mol_label, fv, sv)
            )
        elif not math.isnan(fv) and not math.isclose(fv, sv, rel_tol=1e-6, abs_tol=1e-8):
            mismatches.append(
                "{} ({}): value fast={!r} slow={!r}".format(name, mol_label, fv, sv)
            )

    return mismatches


def test_fast_parity(molecules, calc_fast, calc_slow):
    """Fast and slow paths must produce identical values and Missing positions."""
    desc_names = [str(d) for d in calc_fast.descriptors]
    all_mismatches = []

    for mol in molecules:
        label = mol.GetProp("_Name") if mol.HasProp("_Name") else Chem.MolToSmiles(mol)
        r_fast = list(calc_fast(mol))
        r_slow = list(calc_slow(mol))
        mismatches = _compare_results(r_fast, r_slow, desc_names, label)
        all_mismatches.extend(mismatches)

    assert not all_mismatches, (
        "{} parity failures (first 20):\n".format(len(all_mismatches))
        + "\n".join(all_mismatches[:20])
    )


# ---------------------------------------------------------------------------
# Pickle / parallel round-trip
# ---------------------------------------------------------------------------

def test_fast_flag_survives_pickle():
    """use_fast flag must survive a pickle round-trip."""
    calc = Calculator(descriptors, ignore_3D=True, use_fast=True)
    calc2 = pickle.loads(pickle.dumps(calc))
    assert calc2._use_fast is True

    calc_off = Calculator(descriptors, ignore_3D=True, use_fast=False)
    calc_off2 = pickle.loads(pickle.dumps(calc_off))
    assert calc_off2._use_fast is False


def test_parallel_matches_serial(molecules):
    """Fast-path parallel results must match serial (same values and Missing positions)."""
    calc = Calculator(descriptors, ignore_3D=True)
    mols = molecules[:30]  # keep test fast; coverage comes from test_fast_parity

    serial_results = list(calc.map(mols, nproc=1, quiet=True))
    parallel_results = list(calc.map(mols, quiet=True))

    desc_names = [str(d) for d in calc.descriptors]
    for s_row, p_row in zip(serial_results, parallel_results):
        for d, s, p in zip(desc_names, s_row, p_row):
            if isinstance(s, MissingValueBase):
                assert type(s) == type(p), "{}: serial={} parallel={}".format(d, s, p)
            else:
                assert_almost_equal(
                    float(s), float(p), 7,
                    err_msg="{} (serial: {}, parallel: {})".format(d, s, p),
                )


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def test_env_kill_switch(monkeypatch, molecules):
    """MORDRED_DISABLE_FAST=1 must disable the fast path."""
    monkeypatch.setenv("MORDRED_DISABLE_FAST", "1")
    calc = Calculator(descriptors, ignore_3D=True)
    assert calc._use_fast is False
    # spot-check one molecule still computes
    r = list(calc(molecules[0]))
    assert any(not isinstance(v, MissingValueBase) for v in r)
