"""Complex Mamba-3 Spinor module with V8.0 extensions."""

from .complex_mamba3 import ComplexMamba3Layer
from .lindblad import LindbladDissipation
from .hybrid_automata import HybridAutomata
from .quaternion import Quaternion, QuaternionicEscape

__all__ = [
    "ComplexMamba3Layer",
    "LindbladDissipation",
    "HybridAutomata",
    "Quaternion",
    "QuaternionicEscape",
]
