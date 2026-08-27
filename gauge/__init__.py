"""
gauge -- a model's exact unit-rescaling (nondimensionalization) symmetry.

Importing this package puts the repo root on sys.path. Run the entry points as modules from
the repo root:  `python -m gauge.validate almeida`
"""
import os as _os
import sys as _sys

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

# NOTE: deliberately no eager `from gauge.gauge import ...` here -- that would make
# `python -m gauge.gauge` emit a RuntimeWarning about the module already being in sys.modules.
# Import explicitly:  from gauge.gauge import Gauge

__all__ = ['ROOT']
