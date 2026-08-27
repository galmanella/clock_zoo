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

# Enable float64 BEFORE jax is imported anywhere. Set as an environment variable rather than
# via jax.config so it does not depend on import order: `jax.config.update` only works if it
# runs before the first jax array is created, and a caller that imports `models` and then
# `jax.numpy` directly -- without going through engine/ -- would otherwise silently get
# float32. Measured: that path made the JAX and numpy RHS disagree by 4.3e-07 (exactly
# float32 epsilon) on Goldbeter, which reads as a porting error rather than a dtype problem.
_os.environ.setdefault('JAX_ENABLE_X64', '1')

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)

from models.base import ClockModel, JaxDictParams, MODEL_REGISTRY, register_model, get_model  # noqa: E402

# --- registration (import for side effect) --------------------------------- #
from models import goodwin    # noqa: E402,F401  smoke-test control
from models import korencic   # noqa: E402,F401
from models import goldbeter  # noqa: E402,F401
from models import almeida    # noqa: E402,F401

__all__ = ['ClockModel', 'JaxDictParams', 'MODEL_REGISTRY', 'register_model', 'get_model', 'ROOT']
