"""
models/api.py
=============
The contract a model must satisfy to be pluggable into the downstream stack (orbit solver,
PTC engine, gauge analysis, sensitivity drivers).

Adapted from input_screen/model_api.py (commit e955873). One substantive change: the
PERTURBATION tier is now just `perturbable_targets()`. In input_screen the perturbation was an
ectopic protein carrying a degron, whose material had to be tracked through every complex it
formed, so a model also had to declare a `coupling_spec`. Here the perturbation is generic
(an additive production pulse, or an instantaneous state displacement -- see engine/perturb.py),
so there is nothing model-specific left to declare beyond which states may be targeted.

CAPABILITY TIERS
----------------
A model does not have to implement everything. Each tier is optional and independently
checkable; a tool that needs a tier fails with a clear message naming the missing method
rather than an AttributeError deep in a numerical routine.

  CORE          (models.base.ClockModel; required by everything)
      derivatives(t, y[, params]) -> dy/dt        get_initial_state() -> y0
      get_parameters() / set_parameters(dict)     parameter_names
      state_names / n_states / var_index(name)
      reference_variable, approx_period

  DIMENSIONS    (required by: gauge/)
      parameter_dimensions() -> {param: {species_or_TIME: exponent}}
      scale_coupled_species() -> [ [species, ...], ... ]
      Lets the unit-rescaling symmetry be DERIVED for any model rather than hand-coded.

  JAX           (required by: engine/orbit, engine/ptc, everything downstream of them)
      jax_params() -> pytree of arrays        jax_rhs(y, P) -> dy/dt   (pure, jittable)
      jax_apply(values, names) -> pytree      differentiable scatter of a FLAT parameter
                                              vector into the pytree, so an optimizer or a
                                              vmapped sweep can hold parameters as an array

  PERTURBATION  (required by: engine/perturb, engine/ptc)
      perturbable_targets() -> [species, ...]   which states an experiment can push on

  OBSERVABLES   (required by: the LC observables used in lc_sens / coupling)
      observable_states() -> [species, ...]        (what an experiment measures)
      observable_pairs()  -> [(mrna, protein), ...] (for protein->mRNA lag terms; [] if the
                                                     model has no mRNA/protein distinction)

DECLARING DIMENSIONS
--------------------
For the gauge (see gauge/METHODS.md) each parameter needs its physical dimensions expressed as
exponents over the species' concentration units and over time. The convention is multiplicative
in log space:

    log(param)  +=  sum_X  e_X * log(scale of species X)  +  e_time * log(rho)

where `rho` multiplies every rate constant (a global time rescale). Examples:

    v1      {'X': +1, TIME: +1}     a production rate:  [X] / time
    K1      {'Z': +1}               a threshold carries the units of the species it SENSES,
                                    not the one it regulates
    k3      {'Y': +1, 'X': -1, TIME: +1}    converts X units into Y units
    a_dimer {'PER1': -1, TIME: +1}  second order: 1 / ([conc] * time)
    km      {TIME: +1}              first-order rate: unit-blind
    n       {}                      dimensionless

`scale_coupled_species()` declares which species are FORCED to share one unit. The rule is
more general than the mass-action case input_screen documented:

    ANY TWO SPECIES WHOSE RATE EQUATIONS SHARE AN ADDITIVE TERM MUST SHARE UNITS.

Mass action is the familiar instance (`A + B <-> C` conserves particle number, so the shared
`flux` forces A, B, C into one unit), but it is not the only one. In Almeida a single promoter
activity `EBOX` is added to five different species' equations, which forces those five into one
unit with no complex involved. Return the raw pairs/groups; overlapping groups are merged by
the consumer, so [['A','C'], ['B','C']] is fine and correct.

A model with no such coupling (Goodwin) returns [] and every species scales freely.
"""
from collections import defaultdict

TIME = '@time'          #: sentinel key for the time exponent in parameter_dimensions()

CAPABILITIES = {
    'core': ['derivatives', 'get_initial_state', 'get_parameters', 'set_parameters',
             'state_names', 'n_states', 'var_index'],
    'dimensions': ['parameter_dimensions', 'scale_coupled_species'],
    'jax': ['jax_params', 'jax_rhs', 'jax_apply'],
    'perturbation': ['perturbable_targets'],
    'observables': ['observable_states', 'observable_pairs'],
}


def has_capability(model, name):
    """True if `model` implements every member of capability `name`."""
    return all(hasattr(model, m) for m in CAPABILITIES[name])


def require(model, name, consumer=''):
    """Raise a clear, actionable error if `model` lacks capability `name`."""
    missing = [m for m in CAPABILITIES[name] if not hasattr(model, m)]
    if missing:
        who = f"{consumer} requires" if consumer else "requires"
        raise NotImplementedError(
            f"{type(model).__name__} does not implement the '{name}' capability; {who} it.\n"
            f"  missing: {', '.join(missing)}\n"
            f"  see models/api.py for the contract and models/almeida.py for a worked example.")


def merge_groups(groups, universe=()):
    """Union-find over `groups`; every member of `universe` not mentioned becomes a singleton.
    Returns a list of sorted member lists, itself sorted -- deterministic across runs."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in groups:
        g = list(g)
        for s in g[1:]:
            ra, rb = find(g[0]), find(s)
            if ra != rb:
                parent[ra] = rb
    for s in universe:
        find(s)
    out = defaultdict(list)
    for s in parent:
        out[find(s)].append(s)
    return sorted([sorted(v) for v in out.values()])


def check_conformance(model, verbose=True):
    """Report which capabilities `model` implements, and sanity-check the ones it does.

    Returns {capability: True/False/'<reason>'}. The first thing to run after writing a new
    model. Deeper numerical validation lives in engine/validate.py.
    """
    res = {}
    for cap in CAPABILITIES:
        res[cap] = has_capability(model, cap)

    states = set(model.state_names)

    # --- DIMENSIONS: the easiest declaration to get subtly wrong ------------- #
    if res['dimensions']:
        try:
            dims = model.parameter_dimensions()
            names = set(model.parameter_names)
            problems = []
            missing = names - set(dims)
            if missing:
                problems.append(f"{len(missing)} parameters undeclared "
                                f"(e.g. {sorted(missing)[:3]})")
            extra = set(dims) - names
            if extra:
                problems.append(f"{len(extra)} declared parameters are not model parameters "
                                f"(e.g. {sorted(extra)[:3]})")
            for p, d in dims.items():
                bad = [k for k in d if k != TIME and k not in states]
                if bad:
                    problems.append(f"{p!r} references unknown species {bad}")
                    break
            for grp in model.scale_coupled_species():
                bad = [s for s in grp if s not in states]
                if bad:
                    problems.append(f"scale_coupled_species references unknown species {bad}")
                    break
            if problems:
                res['dimensions'] = '; '.join(problems)
        except NotImplementedError:                         # base-class stub: simply absent
            res['dimensions'] = False
        except Exception as e:                              # a broken declaration is a failure,
            res['dimensions'] = f"{type(e).__name__}: {e}"  # not a missing capability

    # --- PERTURBATION / OBSERVABLES: names must be real states -------------- #
    for cap, meth in (('perturbation', 'perturbable_targets'), ('observables', 'observable_states')):
        if res[cap]:
            try:
                bad = [s for s in getattr(model, meth)() if s not in states]
                if bad:
                    res[cap] = f"{meth}() references unknown species {bad}"
            except NotImplementedError:
                res[cap] = False
            except Exception as e:
                res[cap] = f"{type(e).__name__}: {e}"
    if res['observables'] is True:
        try:
            bad = [x for pair in model.observable_pairs() for x in pair if x not in states]
            if bad:
                res['observables'] = f"observable_pairs() references unknown species {bad}"
        except Exception as e:
            res['observables'] = f"{type(e).__name__}: {e}"

    # --- CORE: the reference variable must exist and be usable -------------- #
    if res['core']:
        ref = getattr(model, 'reference_variable', None)
        if ref is None or ref not in states:
            res['core'] = f"reference_variable {ref!r} is not a state"
        elif not getattr(model, 'approx_period', None):
            res['core'] = "approx_period is unset (the orbit guess needs it)"

    if verbose:
        print(f"conformance: {type(model).__name__}  "
              f"({model.n_states} states, {len(model.parameter_names)} parameters)")
        for cap, ok in res.items():
            mark = 'yes' if ok is True else ('NO ' if ok is False else 'BAD')
            note = '' if isinstance(ok, bool) else f'  <- {ok}'
            print(f"   [{mark}] {cap:13s} {', '.join(CAPABILITIES[cap])[:56]}{note}")
    return res


def _main():
    import sys
    from models import MODEL_REGISTRY, get_model
    names = sys.argv[1:] or sorted(MODEL_REGISTRY)
    ok = True
    for n in names:
        r = check_conformance(get_model(n))
        ok &= all(v is True for v in r.values())
        print()
    print(f"[{'PASS' if ok else 'FAIL'}] conformance over {len(names)} model(s)")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(_main())
