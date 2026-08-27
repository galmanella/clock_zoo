"""
models/goodwin.py
=================
3-variable Goodwin negative-feedback oscillator (Gonze et al. 2005 core).

    dX/dt = v1 * K1^n/(K1^n + Z^n) - v2 * X/(K2 + X)      X = clock-gene mRNA
    dY/dt = k3 * X               - v4 * Y/(K4 + Y)        Y = protein
    dZ/dt = k5 * Y               - v6 * Z/(K6 + Z)        Z = transcriptional repressor

Adapted from input_screen/clock_models.py (commit e955873), which already carried the
DIMENSIONS declaration; the JAX / PERTURBATION / OBSERVABLES tiers are added here.

WHY IT IS IN THIS REPO
    Not a science target -- a SMOKE TEST. It is the smallest thing that oscillates, it has a
    structure deliberately unlike the other models (no complexes, Michaelian rather than
    first-order degradation, so both a max-rate and a threshold sit on the same species), and
    its gauge is known-good from input_screen (4 generators, defining identity to 2.5e-15). So
    every tool is validated against it before it is pointed at a model whose answer we do not
    already know.

    Only the CORE 3-D oscillator is kept: the Gonze model's 4th variable V is a decoupled
    downstream readout and its mean-field term is a population feature (zero for a single
    cell), neither of which affects the single-cell PTC.
"""
import numpy as np

from models.base import ClockModel, JaxDictParams, register_model

_STATES = ['X', 'Y', 'Z']

_PARAMS = {
    'v1': 0.7, 'K1': 1.0, 'n': 4.0,
    'v2': 0.35, 'K2': 1.0,
    'k3': 0.7, 'v4': 0.35, 'K4': 1.0,
    'k5': 0.7, 'v6': 0.35, 'K6': 1.0,
}


def _safe_pow_np(x, n):
    return np.where(x > 0.0, np.where(x > 0.0, x, 1.0) ** n, 0.0)


class GoodwinModel(JaxDictParams, ClockModel):
    """Goodwin/Gonze 3-variable oscillator (11 parameters, period ~24 h)."""

    reference_variable = 'X'
    approx_period = 24.0

    def __init__(self):
        self._state_names = list(_STATES)
        self._idx = {s: i for i, s in enumerate(_STATES)}
        self.params = dict(_PARAMS)

    # --- CORE -------------------------------------------------------------- #
    def get_initial_state(self):
        return np.array([0.1, 0.1, 0.1])

    def var_index(self, name):
        try:
            return self._idx[name]
        except KeyError:
            raise KeyError(f"GoodwinModel has no state {name!r}; states are {_STATES}") from None

    def derivatives(self, t, state, params=None):
        p = self.params if params is None else params
        X, Y, Z = np.maximum(np.asarray(state, float), 0.0)
        Kn = p['K1'] ** p['n']
        return np.array([
            p['v1'] * Kn / (Kn + _safe_pow_np(Z, p['n'])) - p['v2'] * X / (p['K2'] + X),
            p['k3'] * X - p['v4'] * Y / (p['K4'] + Y),
            p['k5'] * Y - p['v6'] * Z / (p['K6'] + Z),
        ])

    # --- JAX --------------------------------------------------------------- #
    @staticmethod
    def jax_rhs(y, P):
        import jax.numpy as jnp

        def safe_pow(x, n):
            # x**n with a finite gradient at x = 0. JAX's pow rule contains x**n * log(x);
            # at x = 0 that is 0 * -inf = NaN, and states hit exactly 0 through the clamp
            # below. The double-where evaluates the power on 1.0 where x <= 0 so no -inf
            # leaks into the backward pass, while returning the exact forward value.
            xs = jnp.where(x > 0.0, x, 1.0)
            return jnp.where(x > 0.0, xs ** n, 0.0)

        yy = jnp.maximum(y, 0.0)
        X, Y, Z = yy[0], yy[1], yy[2]
        Kn = P['K1'] ** P['n']
        return jnp.stack([
            P['v1'] * Kn / (Kn + safe_pow(Z, P['n'])) - P['v2'] * X / (P['K2'] + X),
            P['k3'] * X - P['v4'] * Y / (P['K4'] + Y),
            P['k5'] * Y - P['v6'] * Z / (P['K6'] + Z),
        ])

    # --- DIMENSIONS -------------------------------------------------------- #
    def scale_coupled_species(self):
        return []                                  # no shared additive terms: all scale freely

    def parameter_dimensions(self):
        from models.api import TIME
        return {
            'v1': {'X': 1, TIME: 1}, 'K1': {'Z': 1}, 'n': {},      # K1 senses Z, not X
            'v2': {'X': 1, TIME: 1}, 'K2': {'X': 1},
            'k3': {'Y': 1, 'X': -1, TIME: 1}, 'v4': {'Y': 1, TIME: 1}, 'K4': {'Y': 1},
            'k5': {'Z': 1, 'Y': -1, TIME: 1}, 'v6': {'Z': 1, TIME: 1}, 'K6': {'Z': 1},
        }

    # --- PERTURBATION / OBSERVABLES ---------------------------------------- #
    def perturbable_targets(self):
        return list(_STATES)

    def observable_states(self):
        return ['X']                               # only the mRNA is measured

    def observable_pairs(self):
        return [('X', 'Y')]


register_model('goodwin', GoodwinModel)
