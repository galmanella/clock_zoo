"""engine -- limit-cycle solver, perturbations, PTC, and the adaptive-solver reference path."""
import os as _os, sys as _sys
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
