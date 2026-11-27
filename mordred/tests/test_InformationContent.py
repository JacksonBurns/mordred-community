import pytest
from rdkit import Chem

from mordred import Calculator
from mordred.InformationContent import InformationContent


@pytest.mark.parametrize("smi", ["BrC1CCCC1", "CC1CC1", "C1CC2CCC1C2"])
def test_information_content_input_order(smi):
    # issue #28: the same molecule must give the same values for any input order
    calc = Calculator([InformationContent(order=k) for k in range(6)])
    mol = Chem.MolFromSmiles(smi)
    ref = list(calc(mol).values())
    for s in Chem.MolToRandomSmilesVect(mol, 25, randomSeed=0):
        got = list(calc(Chem.MolFromSmiles(s)).values())
        assert got == pytest.approx(ref), s
