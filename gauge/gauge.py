"""
gauge.gauge
===========
A model's EXACT unit-rescaling (nondimensionalization) symmetry, and the tools to quotient it
out. Adapted from input_screen/gauge/gauge.py (commit e955873); the only structural change is
that a model is now a REQUIRED argument -- there is no default model in this repo.

THE SYMMETRY
------------
Rescaling each species by a constant -- measuring it in different units -- is an exact
symmetry of the ODE that is absorbed entirely into the parameters. Any observable that does
not independently fix the concentration units is therefore blind to a subspace of parameter
space, whose dimension the model's own declaration determines.

    a max production rate carries the units of what it produces    ->  v *= scale[product]
    a Hill/Michaelis threshold carries the units of what it SENSES  ->  K *= scale[regulator]
    a first-order rate is unit-blind                                ->  k unchanged
    a second-order rate                                             ->  a /= scale[partner]
    a global TIME rescale multiplies every rate                     ->  rate *= rho

THE GROUPING IS FORCED, NOT CHOSEN
    Any two species whose rate equations share an ADDITIVE TERM must be counted in the same
    units -- each equation would otherwise demand a different scale for the same quantity.
    Mass action is the familiar instance (A + B <-> C: the shared flux appears in three
    equations with coefficient +-1, so A, B and C share a unit), but a shared promoter
    activity does it too. Union-find over the model's declared groups gives the components.

WHY IT IS EASY TO REMOVE
    Every rule is multiplicative with integer exponents, so in z = log(theta) the gauge orbit
    through any point is the LINEAR subspace  z + G w,  with G a fixed (n_param x n_gauge)
    integer matrix. Quotienting is an exact orthogonal projection -- no fitting, no
    approximation, computed once from the declaration.

WHAT IS AND IS NOT INVARIANT
    invariant : period; relative amplitude of every species; all phases and phase differences;
                waveform shape; winding number; phi*; the PTC as a function of RESCALED dose.
    NOT       : absolute concentrations; cross-species ratios; S_crit (it is a dose); the PTC
                at a fixed ABSOLUTE dose.

    => Compare parameter sets only after canonicalize(); compare PTCs only after dose_factor()
       alignment, or via invariants (phi*, winding, dose-matched twist).

    What this does NOT mean: absolute doses carry real information and must not be normalised
    away per species. The freedom is one factor per scale GROUP under a single shared dose
    calibration, so probing n groups yields n-1 identifiable ratios. And a single-parameter
    perturbation is generically NOT a gauge motion, so its effect on S_crit is a real physical
    effect -- a fixed dose grid in a sensitivity sweep is correct.

Dependencies: numpy + models.api (deliberately light -- runs on the cluster).
"""
import numpy as np

from models.api import TIME, merge_groups, require


class GaugeSpec:
    """Species-scale bookkeeping derived from a model's DIMENSIONS declaration.

    Coordinates are ordered [one per scale-coupled species group] + [time]. A group is a set
    of species that must share one concentration unit; species not mentioned in
    `scale_coupled_species()` are singletons.
    """

    def __init__(self, model):
        require(model, 'dimensions', consumer='gauge')
        self.model = model
        self.states = list(model.state_names)
        self.groups = merge_groups(model.scale_coupled_species(), self.states)
        self._col = {s: i for i, g in enumerate(self.groups) for s in g}
        self.n_species = len(self.groups)
        self.time_col = self.n_species
        self.n_gauge = self.n_species + 1
        self.names = [self._label(g) for g in self.groups] + ['rho_time']

    @staticmethod
    def _label(group):
        return ('scale_' + group[0]) if len(group) == 1 else ('scale_' + '+'.join(group))

    def col(self, species):
        """Gauge coordinate index for a species."""
        return self._col[species]

    def cols_of(self, species_list):
        """Sorted, de-duplicated coordinate indices for a set of species. Use this instead of
        slicing by index: group order is alphabetical (generic union-find), NOT grouped by
        species type, so positional slices are not portable across models."""
        return sorted({self._col[s] for s in species_list})

    def multi_species_groups(self):
        """Indices of coordinates whose group holds more than one species."""
        return [i for i, g in enumerate(self.groups) if len(g) > 1]


# --------------------------------------------------------------------------- #
#  Parameters
# --------------------------------------------------------------------------- #
def default_param_names(model):
    """Canonical ordered log-space parameter list: the model's own order, with nominal-zero
    entries dropped (log(0) is undefined, and a parameter pinned at 0 is not being fitted)."""
    nom = model.get_parameters()
    return [p for p in model.parameter_names if nom[p] > 0]


def constraint_matrix(model, spec):
    """(n_constraints x n_gauge) matrix C from `model.scale_constraints()`.

    Only the coordinate directions in null(C) are genuine symmetries -- see
    ClockModel.scale_constraints for why a model can have any."""
    rows = []
    for c in (getattr(model, 'scale_constraints', lambda: [])() or []):
        r = np.zeros(spec.n_gauge)
        for key, e in c.items():
            r[spec.time_col if key == TIME else spec.col(key)] += float(e)
        rows.append(r)
    return np.array(rows) if rows else np.zeros((0, spec.n_gauge))


def generators(model, param_names=None):
    """The (n_param x n_free) matrix G spanning the gauge subspace in LOG-parameter space,
    built entirely from the model's declared dimensions and scale constraints.

    Returns (G, param_names, spec, N) where N (n_gauge x n_free) is an orthonormal basis of
    the VALID coordinate directions, i.e. null(C). Without constraints N is the identity and
    the free coordinates are exactly the species groups plus time."""
    spec = GaugeSpec(model)
    names = list(param_names) if param_names is not None else default_param_names(model)
    dims = model.parameter_dimensions()
    D = np.zeros((len(names), spec.n_gauge))          # parameter response to a scale change
    for i, p in enumerate(names):
        if p not in dims:
            raise KeyError(
                f"{type(model).__name__}.parameter_dimensions() does not declare {p!r}. "
                f"Every fitted parameter needs a declaration; use an empty dict for a "
                f"dimensionless one (a missing entry would silently be treated as inert).")
        for key, e in dims[p].items():
            D[i, spec.time_col if key == TIME else spec.col(key)] += float(e)

    C = constraint_matrix(model, spec)
    if C.shape[0] == 0:
        N = np.eye(spec.n_gauge)
    else:                                              # orthonormal basis of null(C) via SVD
        _u, s, vt = np.linalg.svd(C)
        rank = int(np.sum(s > max(C.shape) * np.finfo(float).eps * (s.max() if s.size else 1)))
        N = vt[rank:].T
        if N.shape[1] == 0:
            N = np.zeros((spec.n_gauge, 0))
    return D @ N, names, spec, N


# --------------------------------------------------------------------------- #
#  Projection API  (z = log parameter vector, ordered by param_names)
# --------------------------------------------------------------------------- #
class Gauge:
    """Projection machinery for one model + parameter ordering. Build once, reuse.

    All methods take/return z = log(theta) as (..., n_param) arrays. Dimensionless parameters
    (Hill exponents) are gauge-inert -- their G rows are exactly zero -- so the projection
    never moves them.
    """

    def __init__(self, model, param_names=None, include_time=True):
        self.model = model
        self.G_full, self.names, self.spec, self.N = generators(model, param_names)
        self.include_time = include_time
        # `N` maps free coordinates -> full (species-group, time) coordinates. With no scale
        # constraints it is the identity and the two coincide; with constraints (Korencic) the
        # free coordinates are combinations, so anything that needs a specific species' or the
        # time factor must go through `full_coords`.
        self.G = self.G_full
        self.gauge_names = [f'g{i}' for i in range(self.G.shape[1])] \
            if self.N.shape != (self.spec.n_gauge, self.spec.n_gauge) else list(self.spec.names)
        if not include_time:
            # keep only directions with no time component: project N's rows onto time = 0
            keep = np.abs(self.N[self.spec.time_col]) < 1e-12
            self.G = self.G_full[:, keep]
            self.N = self.N[:, keep]
            self.gauge_names = [n for n, k in zip(self.gauge_names, keep) if k]
        self.rank = int(np.linalg.matrix_rank(self.G)) if self.G.size else 0
        # An orthonormal basis Q of span(G). Use the SVD rather than QR: QR returns as many
        # columns as G has, and if G is rank-deficient (a species that no parameter can
        # rescale -- see Korencic) the extra columns are arbitrary, so canonicalize() would
        # project away a genuinely physical direction.
        if self.G.size:
            u, _s, _vt = np.linalg.svd(self.G, full_matrices=False)
            self.Q = u[:, :self.rank]
        else:
            self.Q = np.zeros((len(self.names), 0))
        self.Gplus = np.linalg.pinv(self.G) if self.G.size else np.zeros((0, len(self.names)))
        nom = model.get_parameters()
        self.z_nominal = np.log(np.array([nom[p] for p in self.names]))

    def full_coords(self, w):
        """Free gauge coordinates -> the (species-group..., time) coordinates, i.e. the actual
        log scale factors. Identity unless the model declares scale constraints."""
        return self.N @ np.asarray(w)

    def species_scales(self, w):
        """{species: multiplicative scale factor} and the time factor rho, for a motion `w`."""
        fc = self.full_coords(w)
        return ({s: float(np.exp(fc[self.spec.col(s)])) for s in self.spec.states},
                float(np.exp(fc[self.spec.time_col])))

    # -- basic ops --------------------------------------------------------- #
    def apply(self, z, w):
        """Move z along the gauge orbit by coordinates w. Exactly cost-preserving."""
        return np.asarray(z) + self.G @ np.asarray(w)

    def gauge_coords(self, z, ref=None):
        """Least-squares gauge motion carrying `ref` toward `z`: argmin_w |z - ref - G w|."""
        ref = self.z_nominal if ref is None else np.asarray(ref)
        return self.Gplus @ (np.asarray(z) - ref)

    def canonicalize(self, z, ref=None):
        """z with its gauge component (relative to `ref`) removed -- the canonical
        representative of z's gauge orbit. Two parameter sets are the SAME model iff their
        canonicalized coordinates agree."""
        ref = self.z_nominal if ref is None else np.asarray(ref)
        d = np.asarray(z) - ref
        return ref + (d - self.Q @ (self.Q.T @ d))

    def project_out(self, V):
        """Remove the gauge component from each COLUMN of V (a matrix of directions)."""
        V = np.asarray(V)
        return V - self.Q @ (self.Q.T @ V)

    def split(self, z1, z2):
        """Decompose the log-distance between two parameter sets into gauge and physical
        parts. Returns (total_rms, gauge_rms, physical_rms) -- a Pythagorean split, since the
        projection is orthogonal. `physical` is the only one that means anything."""
        d = np.asarray(z2) - np.asarray(z1)
        g = self.Q @ (self.Q.T @ d)
        rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))
        return rms(d), rms(g), rms(d - g)

    # -- PTC comparison ---------------------------------------------------- #
    def dose_factor(self, z, ref=None, target=None, mode='pulse'):
        """Dose rescale relating the PTCs of `ref` and `z`: PTC_z(alpha*d) == PTC_ref(d) for
        the gauge part of the displacement.

        A pulse dose is a production RATE ([target]/time), so it picks up the time factor too;
        an instant dose is a concentration displacement and does not."""
        target = target or self.model.perturbable_targets()[0]
        fc = self.full_coords(self.gauge_coords(z, ref))
        e = fc[self.spec.col(target)]
        if mode == 'pulse':
            e = e + fc[self.spec.time_col]
        return float(np.exp(e))

    # -- reporting --------------------------------------------------------- #
    def active_mask(self):
        """Boolean mask of parameters the gauge can move at all."""
        return np.abs(self.G_full).sum(axis=1) > 0

    def summary(self):
        act = self.active_mask()
        constrained = self.N.shape != (self.spec.n_gauge, self.spec.n_gauge)
        kinds = {}
        for p, a in zip(self.names, act):                 # group by leading token of the name
            head = ''.join(c for c in p.split('_')[0] if c.isalpha()) or p.split('_')[0]
            kinds.setdefault(head, [0, 0])[int(a)] += 1
        lines = [f"gauge[{type(self.model).__name__}]: {self.G.shape[1]} generators, "
                 f"rank {self.rank} | {int(act.sum())}/{len(act)} parameters gauge-active",
                 f"  quotient dimension: {len(self.names)} - {self.rank} = "
                 f"{len(self.names) - self.rank} physically meaningful combinations",
                 ("  coords: " + ", ".join(self.gauge_names) +
                  ("   (combinations: the model declares scale constraints)"
                   if constrained else "")),
                 "  scale groups: " + " | ".join('+'.join(g) for g in self.spec.groups),
                 "  by parameter kind (inert / active):"]
        for k in sorted(kinds):
            inert, active = kinds[k]
            lines.append(f"    {k:10s} {inert:3d} inert  {active:3d} active")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  dict <-> z helpers
# --------------------------------------------------------------------------- #
def params_to_z(params, g):
    return np.log(np.array([params[p] for p in g.names]))


def z_to_params(z, g, base=None):
    """z -> full parameter dict (entries outside g.names kept at their base value)."""
    out = dict(base or g.model.get_parameters())
    out.update({p: float(v) for p, v in zip(g.names, np.exp(np.asarray(z)))})
    return out


def random_gauge(g, sigma=0.5, seed=0, include_time=False):
    """A random gauge motion w ~ N(0, sigma^2), in FREE coordinates.

    include_time=False projects out any component that would rescale time -- the time rescale
    changes the period, so it is not a symmetry of a period-constrained cost.

    NB the free coordinates are not the (species-group, time) coordinates once a model
    declares scale constraints, so "drop the time column" is wrong in general: for Korencic
    the single generator IS the time rescale, and its time column index (15) is not even a
    valid index into a length-1 free coordinate vector. Project instead.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, sigma, g.G.shape[1])
    if not include_time and g.G.shape[1]:
        t = g.N[g.spec.time_col]                      # how each free coord contributes to rho
        tt = float(t @ t)
        if tt > 1e-24:
            w = w - t * float(t @ w) / tt             # component with log(rho) = 0
    return w


def gauged_model(model, g, w):
    """A fresh model displaced along the gauge orbit by `w`, with `approx_period` corrected.

    The correction matters: a time rescale by rho divides the period by rho, and
    `approx_period` is what sizes the orbit solver's initial guess. Leaving it stale made a
    rho = 0.36 motion on Almeida (true period 69 h, guess window 62 h) find no clean peak,
    fall back to the nominal period and diverge -- reported as a 1e56 "gauge violation" when
    the symmetry was exact to 1e-13.
    """
    _scales, rho = g.species_scales(w)
    m2 = type(model)()
    m2.set_parameters(z_to_params(g.apply(g.z_nominal, w), g))
    if getattr(model, 'approx_period', None):
        m2.approx_period = model.approx_period / rho
    return m2


def _main():
    import sys
    from models import MODEL_REGISTRY, get_model
    for n in (sys.argv[1:] or sorted(MODEL_REGISTRY)):
        print(Gauge(get_model(n)).summary())
        print()


if __name__ == '__main__':
    _main()
