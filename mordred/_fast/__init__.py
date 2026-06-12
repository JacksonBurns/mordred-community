"""Fast RDKit-only descriptor engine vendored from the descriptors package.

Exposes:
- _DescriptorContext: per-molecule cache (distance/adjacency/path/EState data)
- _DESCRIPTOR_FUNCTIONS: flat dict[str, fn(ctx) -> float|int] keyed by Mordred name
- FAST_ENABLED: the set of Mordred 2D descriptor names served by this engine

To disable a descriptor from the fast path (e.g. while migrating its logic into
its Descriptor class body), remove its name from FAST_ENABLED.  Once the class
body is canonical, also remove the name from _DESCRIPTOR_FUNCTIONS and from
mordred_rdkit_registry.SUPPORTED_MORDRED_2D_DESCRIPTORS.
"""

from .rdkit_mordred_like import _DescriptorContext, _DESCRIPTOR_FUNCTIONS
from .mordred_rdkit_registry import SUPPORTED_MORDRED_2D_DESCRIPTORS

FAST_ENABLED: set[str] = set(SUPPORTED_MORDRED_2D_DESCRIPTORS)

__all__ = [
    "_DescriptorContext",
    "_DESCRIPTOR_FUNCTIONS",
    "FAST_ENABLED",
]
