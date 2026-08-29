"""
models/base.py
==============
The minimal autonomous-oscillator interface (CORE capability) plus the model registry.

Adapted from input_screen/clock_models.py (commit e955873).

To add a model: subclass `ClockModel`, implement the abstract bits plus whichever optional
tiers you need (see models/api.py), and `register_model('name', TheClass)`.

The framework never touches a model's internal perturbation machinery: perturbations are
applied EXTERNALLY (engine/perturb.py). A model therefore only has to expose its *unforced*
vector field.
"""
from abc import ABC, abstractmethod

import numpy as np


class ClockModel(ABC):
    """Minimal autonomous-oscillator interface.

    Subclasses must define `_state_names`, `params` (dict), `reference_variable`,
    `approx_period`, and implement `derivatives` / `get_initial_state`.
    """

    #: name of the state variable that defines phase 0 and anchors the orbit's phase
    #: condition. Must be cleanly UNIMODAL over the cycle -- see engine/orbit.py.
    reference_variable = None

    #: name of the state variable the PTC reads phase FROM. Defaults to `reference_variable`.
    #: Set it separately when the section species and the best phase carrier differ -- the
    #: section wants unimodality, the readout wants a healthy amplitude and baseline, and a
    #: parameter set that ruins one need not ruin the other.
    readout_variable = None

    #: rough period (time units), used to size transients and the orbit initial guess. The
    #: true period is solved for, not assumed.
    approx_period = None

    # --- parameters -------------------------------------------------------- #
    def get_parameters(self):
        return self.params.copy()

    def set_parameters(self, parameters):
        for key, value in parameters.items():
            if key in self.params:
                self.params[key] = value
        return self

    @property
    def parameter_names(self):
        return list(self.params.keys())

    # --- state ------------------------------------------------------------- #
    @property
    def state_names(self):
        return self._state_names

    @property
    def n_states(self):
        return len(self._state_names)

    def var_index(self, name):
        try:
            return self._state_names.index(name)
        except ValueError:
            raise KeyError(f"{type(self).__name__} has no state {name!r}; "
                           f"states are {self._state_names}") from None

    # --- dynamics ---------------------------------------------------------- #
    @abstractmethod
    def get_initial_state(self):
        """An initial state near the limit cycle (1-D array)."""

    @abstractmethod
    def derivatives(self, t, state, params=None):
        """d(state)/dt for the *unforced* system (1-D array)."""

    # --- optional capabilities (see models/api.py) ------------------------- #
    # Not abstract: a model that skips a tier simply cannot be used by the tools that need
    # it, and api.require() says so by name.

    def scale_coupled_species(self):
        """Groups of species FORCED to share one concentration unit (default: none)."""
        raise NotImplementedError

    def scale_constraints(self):
        """Linear constraints the species scales must satisfy, beyond sharing units.

        Each entry is a dict {species_or_api.TIME: coefficient} meaning

            sum_X  coefficient_X * log(scale_X)  =  0

        Default: none, which is right whenever every term in every equation has a free
        parameter in front of it to absorb a rescaling.

        It is NOT right when a term has a hardcoded coefficient. Korencic's transcription is a
        bare product of Hill factors with no maximal rate, so `d(gene)/dt = 1*(...) - deg*gene`
        forces `scale[gene] * rho = 1` -- the scale is PINNED to the time rescale rather than
        free. Without this the gauge machinery would report generators that are not symmetries
        at all, and `gauge/validate.py identity` would (correctly) fail on them.

        `scale_coupled_species` is the readable special case of this ({A: 1, B: -1} = 0);
        declare shared units there and genuine relations here.
        """
        return []

    def parameter_dimensions(self):
        """param -> {species: exponent, ..., api.TIME: exponent}."""
        raise NotImplementedError

    def perturbable_targets(self):
        """States an experiment can push on. Default: every state."""
        return list(self._state_names)

    def observable_states(self):
        """States an experiment measures. Default: every state."""
        return list(self._state_names)

    def observable_pairs(self):
        """(mRNA, protein) pairs, for protein->mRNA lag observables. [] if not applicable."""
        return []

    # --- convenience ------------------------------------------------------- #
    def param_vector(self, names=None):
        """Current parameters as a flat float array, ordered by `names` (default: all)."""
        names = list(names or self.parameter_names)
        p = self.params
        return np.array([float(p[n]) for n in names])

    def __repr__(self):
        return (f"<{type(self).__name__} {self.n_states} states, "
                f"{len(self.params)} params, ref={self.reference_variable!r}>")


class JaxDictParams:
    """JAX capability for a model whose parameters are a flat dict of scalars.

    The parameter pytree is just `{name: scalar}`. That is slower than Mirsky's packed flat
    arrays, but these models have 18-56 parameters rather than 132, the RHS is written out
    term by term anyway, and the dict makes `jax_apply` a one-liner that is trivially
    differentiable and vmappable -- which is what a parameter sweep and a global search need.

    A subclass provides `jax_rhs(y, P)` as a @staticmethod.
    """

    def jax_params(self, params=None):
        """Current (or given) parameters as the pytree `jax_rhs` consumes."""
        import jax.numpy as jnp
        p = self.params if params is None else params
        return {k: jnp.asarray(float(v), jnp.float64) for k, v in p.items()}

    def jax_apply(self, values, names, base=None):
        """Scatter a FLAT vector of parameter VALUES (ordered by `names`) into the pytree.

        Pure JAX and differentiable, so an optimizer or a vmapped sweep can hold parameters as
        a plain array. Parameters not in `names` keep their base value.
        """
        P = dict(self.jax_params() if base is None else base)
        for i, n in enumerate(names):
            if n not in P:
                raise KeyError(f"{type(self).__name__} has no parameter {n!r}")
            P[n] = values[i]
        return P


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
MODEL_REGISTRY = {}


def register_model(name, cls):
    MODEL_REGISTRY[name] = cls
    return cls


def get_model(name, **kw):
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kw)
