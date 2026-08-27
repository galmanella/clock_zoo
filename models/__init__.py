"""
models
======
The model zoo. Importing this package registers every model and puts the repo root on
sys.path, so `python -m analysis.<driver>` works from anywhere in the tree.

    from models import get_model, MODEL_REGISTRY
    m = get_model('almeida')

Adding a model: write `models/<name>.py` subclassing `ClockModel` (+ `JaxDictParams` if its
parameters are a flat dict), call `register_model('<name>', TheClass)` at the bottom, and add
one import line below. Then `python -m models.api <name>` checks conformance and
`python -m engine.validate <name>` runs the numerical gate.
"""
import os as _os
import sys as _sys

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

from models.base import ClockModel, JaxDictParams, MODEL_REGISTRY, register_model, get_model  # noqa: E402

# --- registration (import for side effect) --------------------------------- #
from models import goodwin    # noqa: E402,F401  smoke-test control
from models import korencic   # noqa: E402,F401
from models import almeida    # noqa: E402,F401

__all__ = ['ClockModel', 'JaxDictParams', 'MODEL_REGISTRY', 'register_model', 'get_model', 'ROOT']
