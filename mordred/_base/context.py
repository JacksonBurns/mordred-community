from rdkit import Chem

from .._util import conformer_to_numpy
from ..error import Missing3DCoordinate


class Context(object):
    __slots__ = "_mols", "_coords", "n_frags", "name", "_stack", "config", "_canonical_mol", "_fast_ctx", "_fast_results"

    def __init__(self, mols, coords, n_frags, name, config, canonical_mol=None):
        self._mols = mols
        self._coords = coords
        self.n_frags = n_frags
        self.name = name
        self.config = config
        self._canonical_mol = canonical_mol
        self._fast_ctx = None
        self._fast_results = None

    def __reduce_ex__(self, version):
        # _fast_ctx and _fast_results are not pickled; parallel workers rebuild lazily.
        return (
            self.__class__,
            (self._mols, self._coords, self.n_frags, self.name, self.config, self._canonical_mol),
        )

    def get_fast_ctx(self):
        if self._fast_ctx is None:
            from .._fast import _DescriptorContext
            self._fast_ctx = _DescriptorContext(self._canonical_mol)
        return self._fast_ctx

    def get_fast_results(self):
        """Lazily compute ALL fast-enabled descriptors in one bulk pass and cache."""
        if self._fast_results is None:
            from .._fast import _DESCRIPTOR_FUNCTIONS, FAST_ENABLED
            fast_ctx = self.get_fast_ctx()
            self._fast_results = {
                name: _DESCRIPTOR_FUNCTIONS[name](fast_ctx)
                for name in FAST_ENABLED
            }
        return self._fast_results

    def __str__(self):
        return self.name

    __tf = {True, False}

    @classmethod
    def from_query(cls, mol, require_3D, explicit_hydrogens, kekulizes, id, config):
        if not isinstance(mol, Chem.Mol):
            raise TypeError("{!r} is not rdkit.Chem.Mol instance".format(mol))

        n_frags = len(Chem.GetMolFrags(mol))

        if mol.HasProp("_Name"):
            name = mol.GetProp("_Name")
        else:
            name = Chem.MolToSmiles(Chem.RemoveHs(mol, updateExplicitCount=True))

        canonical_mol = Chem.RemoveHs(mol, updateExplicitCount=True)
        canonical_mol.RemoveAllConformers()

        mols, coords = {}, {}

        for eh, ke in ((eh, ke) for eh in explicit_hydrogens for ke in kekulizes):
            m = Chem.AddHs(mol) if eh else Chem.RemoveHs(mol, updateExplicitCount=True)

            if ke:
                Chem.Kekulize(m)

            if require_3D:
                try:
                    conf = m.GetConformer(id)
                    if conf.Is3D():
                        coords[eh, ke] = conformer_to_numpy(conf)
                except ValueError:
                    pass

            m.RemoveAllConformers()
            mols[eh, ke] = m

        return cls(mols, coords, n_frags, name, config, canonical_mol)

    @classmethod
    def from_calculator(cls, calc, mol, id):
        return cls.from_query(
            mol,
            calc._require_3D,
            calc._explicit_hydrogens,
            calc._kekulizes,
            id,
            calc._config,
        )

    def get_coord(self, desc):
        try:
            return self._coords[desc.explicit_hydrogens, desc.kekulize]
        except KeyError:
            desc.fail(Missing3DCoordinate())

    def get_mol(self, desc):
        return self._mols[desc.explicit_hydrogens, desc.kekulize]

    def reset(self):
        self._stack = []

    def add_stack(self, d):
        self._stack.append(d)

    def get_stack(self):
        return self._stack
