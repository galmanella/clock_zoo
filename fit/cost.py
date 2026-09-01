"""
fit/cost.py
===========
The objective for fitting a clock model to a PTC surface.

    $PY -m fit.cost --selftest                 # the anti-degeneracy invariant
    $PY -m fit.cost --gradcheck                # autodiff vs finite differences

    C(v) = C_ptc(v) + w_osc * B_osc(v) + w_amp * A(v)

DESIGNED AGAINST ONE SPECIFIC FAILURE
    input_screen/radialize.py fit Mirsky to a radial-isochron target with cost
    `twist_cost + 0.5 * feature_cost` and no limit-cycle term. It converged -- 0.89 -> 0.159 in
    11 L-BFGS iterations -- by walking the clock to a Hopf bifurcation: the phase singularity
    was annihilated, the limit-cycle amplitude collapsed, and the twist "improved" only because
    the oscillation was dying. Parameters moved less than 4%. Both terms of that cost are
    MINIMIZED by a dying oscillator.

    So the arrangement here is deliberate:

    C_ptc   A POINTWISE match over the whole (phase, dose) surface, not a match to extracted
            features. The radial target is defined at every cell, so a dead oscillator cannot
            match it -- but only if unusable cells are scored rather than skipped. They score
            the MAXIMUM (1.0). See fit/target.circ_cost; that one line is what inverts the
            Mirsky degeneracy, and `--selftest` asserts it.
    A       An explicit floor on limit-cycle amplitude, at `amp_frac` of the NOMINAL model's.

            It must be LOOSE. Set at 0.5 it fired on the self-recovery control's own ground
            truth: a displaced-but-perfectly-healthy parameter set whose cycle was 1.32 against
            nominal 4.337 scored 0.153 while its PTC residual was 1e-12 -- so the truth was no
            longer the global minimum and the control was measuring the floor rather than the
            landscape. Worse, it is exactly the LC-shape constraint this experiment is supposed
            to do without: a model with a 3x smaller cycle is a different model, not a dead one.

            At 0.05 the floor sits at 0.217 against a nominal 4.337 -- it means "there is
            essentially no oscillation left", which is all it is for. It still catches the
            degenerate optimum (the collapsed case in `selftest` reaches amp_lc = 0.000), and
            the real anti-degeneracy work is done by C_ptc scoring dead cells at 1.0 anyway.
    B_osc   The Hopf-distance barrier (ported from input_screen/osc_term.py). This is the
            user's stated constraint -- "self-sustained oscillation and nothing more" -- and it
            has a second job that matters just as much: C_ptc SATURATES at 1.0 across the whole
            dead region, so it is flat there and cannot tell an optimizer which way to go.
            Re(lambda_max) at the fixed point is a continuous signed distance to the Hopf
            boundary, defined whether or not the system oscillates, so it supplies the gradient
            that carries a dead candidate back to life. That matters for multistart (T2), where
            random starts routinely land dead.

WHAT IS *NOT* CONSTRAINED
    LC shape. No waveform, amplitude-profile, or period target beyond the floor above. That is
    the point of the experiment: ask whether PTC data ALONE identifies the model. It is also
    what makes the degeneracy live, hence the care above.

COORDINATES
    The optimizer works in `v`, a vector in the GAUGE QUOTIENT: v -> u = B @ v -> theta =
    theta_base * exp(u), with B an orthonormal basis of the complement of the gauge span. For
    Almeida that is 16 free directions out of 18 parameters. Optimizing raw log-parameters
    instead would leave 2 exactly-flat directions for the optimizer to wander along, which
    wastes population in CMA and makes an L-BFGS Hessian singular.
"""
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax, vmap

from engine.orbit import OrbitSolver, make_guess_fn
from engine.ptc import make_ptc, grid_points, DEAD_AMP, NEG_TOL
from fit.target import circ_cost, profile, radial_z
from gauge.gauge import Gauge


# --------------------------------------------------------------------------- #
#  Gauge quotient coordinates
# --------------------------------------------------------------------------- #
def quotient_basis(model, param_names=None, include_time=True):
    """(names, z_base, B) with B an orthonormal basis of the gauge quotient.

    B has one column per PHYSICALLY MEANINGFUL parameter combination: 16 of 18 for Almeida,
    33 of 34 for Korencic, 47 of 52 for Goldbeter. Moving along span(B) can never be undone by
    a change of units, so every direction the optimizer can travel is one that changes the
    model.
    """
    g = Gauge(model, param_names, include_time=include_time)
    n = len(g.names)
    P = np.eye(n) - (g.Q @ g.Q.T if g.Q.size else 0.0)
    u, s, _vt = np.linalg.svd(P)
    B = u[:, s > 1e-8]
    return list(g.names), g.z_nominal.copy(), B, g


def row_dispersion(zu, alive):
    """Per-dose-row circular dispersion of a unit-phase field: 1 - |mean resultant|, in [0, 1].

    THE BRACKETING OBSERVABLE. It says how much a dose row's new phase varies with OLD phase:

        type-1 row -- new phase sweeps the circle    -> resultant ~ 0 -> dispersion ~ 1
        type-0 row -- new phase nearly constant      -> resultant ~ 1 -> dispersion ~ 0

    so the transition sits where it crosses. Three properties make it the right variable for a
    barrier, and each one rules out an alternative that was considered first:

      * DEFINED WHEN THE SINGULARITY IS GONE. `fit.target.soft_singularity` locates the defect
        from the amplitude dip, so it cannot see a defect that has already left the grid -- and
        that is exactly the state we need to penalise. Dispersion is a property of every row
        whether or not a singularity is present anywhere.
      * DIRECTION-AWARE. Dispersion low at the BOTTOM of the window means the transition fell
        below the floor; still high at the TOP means it rose above the ceiling. A barrier on a
        located S* cannot distinguish those once the defect is off-grid; this can, and the
        Aug-30 campaign contains both (13 escaped downward, seed 0 upward).
      * SMOOTH. A mean of unit vectors -- no max, no plaquette search, no grid quantisation, so
        it differentiates cleanly. `analysis.winding.detect_grid` is a staircase and
        `circ_span` is a max.

    VALIDATED against the independent extended-grid S_crit on all 16 Aug-30 fits: every run this
    reads as "type-0 throughout" has a rescanned S* of 0.01-4.0 against a window floor of 12.5,
    and the one it reads as "type-1 throughout" (seed 0) has S* = 363 against a ceiling of 200.
    """
    z = jnp.where(alive, zu, 0.0)
    n = jnp.maximum(jnp.sum(alive.astype(zu.real.dtype), axis=0), 1.0)
    return 1.0 - jnp.abs(jnp.sum(z, axis=0) / n)


def _target_span(zt):
    """Per-dose-row circular span of a target phase field -- how much old-phase structure it has.

    CIRCULAR, and that is not a nicety: phase wraps, and a peak-to-peak about an arithmetic
    mean read 4 informative dose rows for REV against a true 1, because REV's target sits at
    psi = 0.975 and its rows straddle the 0/1 boundary (PROJECT_SUMMARY 5.12). Reuses
    `analysis.winding.circ_span`, which saturates at 0.5 -- here that ceiling IS the meaning
    wanted, since 0.5 already means "sweeps the whole circle".
    """
    from analysis.winding import circ_span
    ph = (np.angle(np.asarray(zt)) / (2 * np.pi)) % 1.0
    return np.array([circ_span(ph[:, j]) for j in range(ph.shape[1])], float)


# --------------------------------------------------------------------------- #
#  Hopf-distance barrier (ported from input_screen/osc_term.py, generalized)
# --------------------------------------------------------------------------- #
def _maxre_val_np(A):
    A = np.asarray(A)
    if not np.isfinite(A).all():             # divergent params -> NaN, not a LAPACK raise
        return np.asarray(np.nan, A.dtype)
    return np.asarray(np.max(np.linalg.eigvals(A).real), A.dtype)


#: |y^H x| below which the leading eigenvalue is treated as DEFECTIVE and its derivative as
#: undefined. See _leading_np.
_DEFECTIVE_TOL = 1e-10


def _leading_np(J):
    """Value and exact gradient of Re(lambda_max): d(Re lam)/dJ_ab = Re(conj(y_a) x_b/(y^H x)),
    with x, y the right and left eigenvectors of the leading eigenvalue.

    THE DENOMINATOR CAN VANISH, AND IT DOES. `y^H x` -> 0 exactly when the leading eigenvalue
    becomes defective -- two eigenvalues colliding, so the left and right eigenvectors become
    orthogonal and the simple-eigenvalue derivative formula stops existing. That is a
    singularity of the FORMULA, not of the cost: Re(lambda_max) itself is perfectly finite
    there, it merely stops being differentiable.

    Left unguarded this returns inf/NaN, and in a fit that is worse than it sounds. MEASURED in
    the first T1 run: 7 of 108 gradient evaluations came back non-finite, the search wrapper
    substituted ZEROS, and L-BFGS read that as convergence and stopped at 58 of 150 iterations.
    The run then reported "NOT RECOVERED" -- a landscape verdict that was really this bug.

    So a defective point returns the correct VALUE and a ZERO gradient, which is the honest
    answer (the derivative does not exist) and is safe because the caller reports how often it
    happens rather than silently averaging it in.
    """
    J = np.asarray(J)
    if not np.isfinite(J).all():
        return np.asarray(np.nan, J.dtype), np.zeros_like(J)
    w, X = np.linalg.eig(J)
    i = int(np.argmax(w.real)); lam = w[i]; x = X[:, i]
    wl, Y = np.linalg.eig(J.T)               # right eigvecs of J^T = left eigvecs of J
    j = int(np.argmin(np.abs(wl - lam.conjugate()))); y = Y[:, j]
    denom = np.vdot(y, x)
    val = np.asarray(np.real(lam), J.dtype)
    if not np.isfinite(denom) or abs(denom) < _DEFECTIVE_TOL:
        return val, np.zeros_like(J)
    G = np.real(np.outer(np.conjugate(y), x) / denom)
    if not np.isfinite(G).all():
        return val, np.zeros_like(J)
    return val, np.asarray(G, J.dtype)


@jax.custom_jvp
def _max_re_eig(J):
    return jax.pure_callback(_maxre_val_np, jax.ShapeDtypeStruct((), J.dtype), J,
                             vmap_method='sequential')


@_max_re_eig.defjvp
def _mre_jvp(primals, tangents):
    """custom_JVP, not custom_VJP, and the difference is load-bearing.

    `jax.custom_vjp` supplies REVERSE mode only. Anything needing forward mode hits

        TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function

    and `fit.search.levenberg_marquardt` needs forward mode by construction: the residual
    Jacobian is built with `jacfwd` (that is how the diffrax flow gets differentiated -- see
    GRAD_MODES in engine/flow). So every LM run died here, in the Hopf barrier, nowhere near
    the flow anyone would have suspected. It killed the whole optimizer benchmark process
    partway through, taking the finished CMA and BOBYQA legs down with it.

    `custom_jvp` gives BOTH modes: JAX obtains reverse mode by transposing the jvp rule, which
    is exact here because the rule is linear in the tangent -- the derivative is the fixed
    matrix G contracted with dJ. The numerics are unchanged; G is the same analytic
    eigenvalue derivative `_leading_np` already returned, zero-guarded at defective points.
    """
    (J,), (dJ,) = primals, tangents
    val, G = jax.pure_callback(
        _leading_np,
        (jax.ShapeDtypeStruct((), J.dtype), jax.ShapeDtypeStruct(J.shape, J.dtype)),
        J, vmap_method='sequential')
    return val, jnp.sum(G * dJ)


def make_growth_fn(model, backend='diffrax', eps=1e-4, k=8, ndir=4, m=256):
    """`growth(P, y0, T) -> r`, the per-period growth of a small deviation from the cycle.

    WHY NOT THE FLOQUET MULTIPLIER. mu is the textbook answer and it is not usable here: it
    needs the monodromy, which integrates the variational equation over a period, so its
    entries grow like exp(lambda*T) and OVERFLOW -- it returned a non-finite matrix on BMAL1
    (killing a completed fit), reported 2.35e6 for a PER optimum, and 0.5146 for the SAME point
    on recompute. It is also a numpy `pure_callback` into `eig`, so it blocks forward-mode
    autodiff and serialises under vmap.

    THIS instead perturbs the cycle and watches, which is power iteration on the monodromy
    without ever forming it: successive periods amplify the leading direction, so
    (d_k/d_0)^(1/k) approaches mu_lead. Distance is measured to the CYCLE (a min over sampled
    points), which quotients out the neutral phase direction that would otherwise floor the
    ratio at 1.

    BENCHMARKED, and the settings are not arbitrary:
      * eps=1e-4 -- at 1e-7 the probe sits AT the integrator's rtol and measures solver noise,
        which produced a 6x spread in the value. Three orders of headroom fixes it.
      * ndir=4 -- one random direction is a noisy lower bound; the geometric mean is stabler.
      * k=8 -- more iterations, better convergence to the leading direction.
    On the known points: base 0.816, REV optimum 0.552 (both attracting), PER optimum 1.437
    (repelling -- confirmed independently by direct perturbation, 204x growth over 8 periods,
    where mu claimed 0.51). Continuity along viable paths: steps of 0.001-0.002 away from
    bifurcations, rising sharply only where the orbit is about to cease to exist.

    Costs ~32 periods of integration against the PTC's ~3080, i.e. about 1%.
    """
    from engine.flow import make_window_sampler
    sampler = make_window_sampler(model, backend=backend)
    solver = OrbitSolver(model)
    g = np.random.default_rng(12345)
    U = g.normal(size=(int(ndir), int(model.n_states)))
    U = jnp.asarray(U / np.linalg.norm(U, axis=1, keepdims=True))

    def growth(P, y0, T):
        cyc = solver.cycle(P, y0, T, m)
        scale = jnp.maximum(cyc.max(0) - cyc.min(0), 1e-12)

        def one(u):
            pert = y0 + eps * scale * u
            d0 = jnp.min(jnp.linalg.norm(cyc - pert, axis=1))
            tr = sampler(pert, 2, k * T / 2, P)
            dk = jnp.min(jnp.linalg.norm(cyc - tr[-1], axis=1))
            return jnp.log(jnp.maximum(dk, 1e-300)) - jnp.log(jnp.maximum(d0, 1e-300))

        return jnp.exp(jnp.mean(vmap(one)(U)) / k)

    return growth


def make_osc_penalty(model, w=0.2, k=200.0, margin=0.015, n_iter=40, reg=1e-9):
    """`pen(P, y_guess) -> (penalty, Re(lambda_max), residual)`.

    w * softplus(k * (margin - Re(lambda_max))): ~0 once comfortably oscillating, so it does
    not distort the data fit, and growing with a strong gradient once the fixed point turns
    stable (i.e. the clock is dead or damped). Newton runs to convergence under stop_gradient,
    then ONE differentiable implicit step, so d y*/dP is the exact implicit-function gradient
    at the cost of a single linear solve.

    `margin` and `k` ARE MODEL-SPECIFIC and the defaults here are input_screen's, tuned for
    Mirsky. Re(lambda_max) has units of 1/time and its natural size differs per model, so a
    fixed margin means different things in different models. Use `scaled_osc_penalty` unless
    you have a reason not to -- the literal defaults fire on a healthy Almeida oscillator.
    """
    rhs = model.jax_rhs
    N = int(model.n_states)
    I = jnp.eye(N)

    def fixed_point(P, y_guess):
        def body(y, _):
            f = rhs(y, P)
            J = jax.jacfwd(rhs)(y, P)
            dy = jnp.linalg.solve(J + reg * I, f)
            return jnp.maximum(y - 0.7 * dy, 0.0), None     # damped + positivity-safeguarded
        yc, _ = lax.scan(body, y_guess, None, length=n_iter)
        yc = lax.stop_gradient(yc)
        f = rhs(yc, P); J = jax.jacfwd(rhs)(yc, P)
        ystar = yc - jnp.linalg.solve(J + reg * I, f)
        res = jnp.linalg.norm(rhs(lax.stop_gradient(ystar), P))
        return ystar, res

    def pen(P, y_guess):
        ystar, res = fixed_point(P, y_guess)
        re = _max_re_eig(jax.jacfwd(rhs)(ystar, P))
        p = w * jax.nn.softplus(k * (margin - re))

        # REJECT THE TRIVIAL FIXED POINT.
        #
        # y = 0 is a genuine equilibrium of a model like Almeida -- every CCE term vanishes when
        # all species are zero -- and `fixed_point` clamps with max(y - 0.7 dy, 0), which makes
        # it an attractor for the damped Newton from some starting points. Its stability is not
        # the quantity this barrier is about: the question is whether the PHYSIOLOGICAL fixed
        # point has crossed the Hopf boundary.
        #
        # Unguarded it produced a flickering penalty. MEASURED along one line in parameter
        # space: osc = 0.0000, 0.0000, 0.0127, 0.0000, 1.0375, 0.0005, 1.0375 -- switching
        # between ~0 and exactly softplus(0.6) = 1.0375, the value for Re(lambda) = 0, at
        # scattered parameter values with every PTC cell alive throughout. That is pure
        # numerical roughness added to an objective an optimizer has to navigate.
        #
        # (It was NOT the cause of the barriers between basins: with w_osc = 0 those measure
        # +0.03623 and +0.57997 against +0.0363 and +0.5800 with it. Two separate problems.)
        trivial = jnp.linalg.norm(ystar) < 1e-3 * jnp.maximum(jnp.linalg.norm(y_guess), 1e-30)
        dead = w * jax.nn.softplus(k * margin + 5.0)
        p = jnp.where(jnp.isfinite(re) & (res < 1e-3) & ~trivial, p, dead)
        return p, jnp.where(trivial, jnp.nan, re), res

    return pen


def scaled_osc_penalty(model, w=0.2, margin_frac=0.02, k_scale=30.0, n_iter=40):
    """The Hopf barrier, with its scale DERIVED from the model instead of hardcoded.

    Returns `(pen, re_nominal)`.

    WHY THIS IS NOT OPTIONAL. Re(lambda_max) at the fixed point has units of 1/time, so its
    natural magnitude is a property of the model. input_screen's constants (margin = 0.015,
    k = 200) were tuned on Mirsky. On Almeida, whose NOMINAL Re(lambda_max) is 0.0627, they
    misfire badly: the self-recovery control's own ground truth sits at Re = 0.0143 -- POSITIVE,
    i.e. a genuinely self-sustained oscillator with a healthy cycle (amplitude 2.31, period
    19.45 h) -- and was charged a penalty of 0.766, which put the global minimum somewhere other
    than the truth and made the control measure the barrier instead of the landscape.

    So both knobs are expressed relative to the model's own nominal value:

        margin = margin_frac * re_nominal      the buffer past the true Hopf boundary (Re = 0)
        k      = k_scale / re_nominal          the steepness, in the same units

    With the defaults, a system at 20% of nominal Re pays ~2e-3, at the bifurcation ~1.0, and
    well past it the penalty grows linearly with a strong gradient -- which is the whole point,
    since C_ptc is flat at its 1.0 ceiling across the dead region and cannot guide anything.
    """
    P0 = model.jax_params()
    # SEED FROM THE LIMIT-CYCLE MEAN, not from a generic initial state.
    #
    # The damped Newton in `fixed_point` clamps at zero, and y = 0 is a genuine equilibrium of a
    # model like Almeida (every CCE term vanishes there). Started from `get_initial_state()` it
    # frequently lands on that trivial point instead of the physiological one, and the trivial
    # point is UNSTABLE, so Re(lambda) > 0 reads as "healthy oscillation" and the barrier
    # silently switches off. At other parameter values Newton fails outright (res >= 1e-3) and
    # the constant fallback softplus(k*margin+5) = 5.6037 switches on.
    #
    # MEASURED before this fix, along one segment: osc took the values 5.6037, 5.6037, ...,
    # 0.0000, 0.0000, 5.6037, ... with every PTC cell alive and the cycle healthy throughout --
    # and because both branches are CONSTANT in the parameters, the term contributed exactly
    # zero gradient (|grad| identical with and without it, to 4 digits, at 0.0 degrees). It was
    # adding discontinuous offsets to the cost while supplying none of the gradient it exists
    # for.
    #
    # The cycle mean lies inside the basin of the interior fixed point, which is the one whose
    # stability the Hopf distance is actually about.
    from engine.orbit import OrbitSolver as _OS
    _s = _OS(model)
    try:
        _y0, _T, _r = jax.jit(_s.solve)(P0, _s.guess(P0))
        y_seed = jnp.mean(_s.cycle(P0, _y0, _T, 64), axis=0)
    except Exception:
        y_seed = jnp.asarray(model.get_initial_state(), jnp.float64)
    probe = make_osc_penalty(model, w=1.0, k=1.0, margin=0.0, n_iter=n_iter)
    _p, re_nom, _res = probe(P0, y_seed)
    re_nom = float(np.asarray(re_nom))
    if not np.isfinite(re_nom) or re_nom <= 0:
        # Nominal is not an unstable spiral -- fall back to the literal constants rather than
        # dividing by a non-positive number, and let the caller see it.
        return make_osc_penalty(model, w=w, n_iter=n_iter), re_nom
    return (make_osc_penalty(model, w=w, k=k_scale / re_nom,
                             margin=margin_frac * re_nom, n_iter=n_iter), re_nom)


# --------------------------------------------------------------------------- #
#  Targets
# --------------------------------------------------------------------------- #
class FixedTarget:
    """A concrete surface to match, in the model's own dose units (the self-recovery control).

    No profiling: the target came from this same model and engine, so its dose scale and its
    phase origin are already the model's. Fitting nuisance parameters here would let the
    optimizer hide a real parameter error inside a dose rescale."""
    name = 'fixed'

    def __init__(self, zt):
        self.zt = jnp.asarray(zt)

    def __call__(self, zm, alive, old, doses):
        return self.zt, {}


class RadialTarget:
    """The analytic radial-isochron surface.

    (k, psi) place the target: k sets its critical dose (S_crit = 1/k) and psi its singular
    phase (phi* = psi + 0.5). They are NOT properties of the model -- the target's dose axis has
    no natural units and a Poincare oscillator has no distinguished phase -- so they have to
    come from somewhere.

    PINNED IS THE DEFAULT, AND PROFILING IS THE OPTION.
        Pass k and psi (see `from_singularity`) and the target is a FIXED picture for the whole
        fit. Pass neither and they are re-profiled at every evaluation to whatever best matches
        the surface being scored.

        Profiling looks attractive -- it asks "is the field radial, wherever the defect sits"
        rather than the stricter "is it radial AND is the defect exactly here" -- but it makes
        the target a function of the model, and that is a feedback loop on top of the ordinary
        one:

        1. `min` over a family is NOT smooth. The profiled cost has kinks wherever the argmin
           jumps to another (k, psi) branch, so the effective target moves discontinuously as
           the parameters move. On a landscape already rugged from the spiral geometry
           (PROJECT_SUMMARY 5.4c) that is the last thing to add.
        2. Costs at different parameter sets are then measured against DIFFERENT targets, so
           they are not strictly comparable, and the singularity is free to drift anywhere --
           a "radialized" model could end up with an S_crit nowhere near the seed's, a large
           physical change that nothing in the objective would flag.

        input_screen/radialize.py pinned it: `make_radial_target(S_crit, phi_sing, ...)` built
        the target at the BASE run's singularity and asked the model to flatten the twist while
        holding the defect where it already was. That is the well-posed question, and it is what
        this class now does by default.
    """
    name = 'radial'

    def __init__(self, k=None, psi=None, n_k=48, n_psi=48, refine=2):
        self.k, self.psi = k, psi
        self.n_k, self.n_psi, self.refine = n_k, n_psi, refine
        self.name = 'radial' if (k is None) else 'radial-pinned'

    @staticmethod
    def from_singularity(s_crit, phi_sing):
        """Target pinned to a measured singularity: k = 1/S_crit, psi = phi* - 0.5."""
        return RadialTarget(k=1.0 / float(s_crit), psi=(float(phi_sing) - 0.5) % 1.0)

    def __call__(self, zm, alive, old, doses):
        if self.k is not None:
            k = jnp.asarray(self.k)
            psi = jnp.asarray(self.psi if self.psi is not None else 0.0)
        else:
            k, psi, _c = profile(zm, alive, old, doses, self.n_k, self.n_psi,
                                 refine=self.refine)
        return radial_z(old, doses, k, psi), {'k': k, 'psi': psi}


# --------------------------------------------------------------------------- #
#  The cost
# --------------------------------------------------------------------------- #
def make_cost(model, target_state, doses, tgt, n_phase=24, mode='instant', backend='diffrax',
              dt=0.02, skip_p=None, w_osc=0.2, w_amp=1.0, amp_frac=0.05, m_amp=64,
              param_names=None, eps=1e-12, grad_mode='rev', ridge=0.0, pulse=8.0,
              readout_ref=None, w_stab=0.0, r_max=0.98, basis=None, amp_ramp=(0.05, 0.20),
              row_weight=False, row_weight_floor=0.1,
              w_brack=0.0, brack_lo=0.25, brack_hi=0.65):
    """Build the objective.

    Returns a dict with
        v0        the origin (all zeros) -- the base parameter set
        n_free    dimension of the search
        parts(v)  -> dict of every term plus diagnostics, as plain floats
        total(v)  -> the scalar to minimize (jit-compiled)
        grad(v)   -> its gradient
        surface(v)-> (z_unit, alive, amp) for plotting

    `basis` pins the gauge-quotient basis instead of re-deriving it -- REQUIRED when
    re-evaluating a `v` saved by an earlier run. See the comment on it below.

    `amp_ramp = (lo, hi)` is the aliveness ramp (P1 of docs/FIT_VALIDITY.md); pass None to
    recover the pre-P1 objective exactly, which is what any comparison against the Aug-30
    campaign must do.

    `row_weight` turns on informativeness weighting (P3). PINNED TARGETS ONLY -- it raises
    otherwise, for the reason given at its definition below.

    `w_brack` weights the BRACKETING BARRIER: a one-sided penalty that keeps the type-1 ->
    type-0 transition inside the dose window, so the window does not have to be widened to
    chase it. Zero at a healthy point by construction; see `row_dispersion`.
    """
    names, z_base, B, g = quotient_basis(model, param_names)
    # A CALLER MAY PIN THE BASIS, AND ANYTHING RE-EVALUATING A SAVED `v` MUST.
    #
    # `quotient_basis` takes the SVD of the projector I - QQ^T, whose nonzero singular values
    # are ALL EXACTLY 1 -- a 16-fold degenerate subspace for Almeida. Any orthonormal basis of
    # it is a valid SVD, so LAPACK's choice is not reproducible across builds or even calls,
    # and `v` is therefore a MACHINE-LOCAL coordinate, not a portable one.
    #
    # MEASURED, not feared: re-deriving the basis here and evaluating the Aug-30 campaign's own
    # stored `v_fit` reconstructed parameter sets up to 3.0 DECADES away from the `theta_fit`
    # those runs recorded, and the resulting "surfaces" were uniformly dead. Every run of that
    # campaign shares one basis, so results within it are sound; re-reading them elsewhere is
    # not. Pass the run's stored `B` and the coordinate means what it meant.
    if basis is not None:
        B = np.asarray(basis, float)
        if B.shape[0] != len(z_base):
            raise ValueError(f"basis has {B.shape[0]} rows, expected {len(z_base)}")
    n_free = B.shape[1]
    Bj, zbj = jnp.asarray(B), jnp.asarray(z_base)

    # `pulse` and `readout_ref` reach the PTC from here so that a run can set them. Before
    # this they were make_ptc defaults that no caller could change: a pulse experiment was
    # locked to 8 h, and the phase observable was whatever the model happened to declare.
    f, solver = make_ptc(model, target_state, mode=mode, readout='raw', skip_p=skip_p,
                         dt=dt, track_min=True, backend=backend, grad_mode=grad_mode,
                         pulse=pulse, readout_ref=readout_ref)
    guess = make_guess_fn(model)
    ph, dz = grid_points(n_phase, doses)
    nd, npz = len(doses), n_phase
    old = jnp.arange(n_phase) / n_phase
    ref_idx = int(model.var_index(model.reference_variable))
    # The ORBIT relaxation seed. Distinct from the fixed-point seed below, and deliberately
    # left alone: `guess(P, y_seed)` feeds Newton's basin for the periodic orbit, every batch-1
    # result was produced with this value, and repurposing it would silently change orbit
    # finding everywhere.
    y_seed = jnp.asarray(model.get_initial_state(), jnp.float64)

    # base limit-cycle amplitude, the reference the floor is expressed against
    Pb = model.jax_params()
    y0b, Tb, _rb = jax.jit(solver.solve)(Pb, solver.guess(Pb))
    cyc_b = solver.cycle(Pb, y0b, Tb, m_amp)
    cb = np.asarray(cyc_b)[:, ref_idx]
    amp_base = float((cb.max() - cb.min()) / max(abs(cb.mean()), 1e-12))
    amp_floor = amp_frac * amp_base

    # The FIXED-POINT seed for the Hopf barrier: the limit-cycle mean, which lies inside the
    # basin of the interior equilibrium. See scaled_osc_penalty for what starting from
    # `get_initial_state()` did instead (it landed on the trivial y = 0 point, or failed to
    # converge, and the term contributed constants with zero gradient).
    y_fp_seed = jnp.mean(cyc_b, axis=0)
    osc, re_nom = scaled_osc_penalty(model, w=1.0) if w_osc else (None, float('nan'))

    def _theta(v):
        return jnp.exp(zbj + Bj @ v)

    growth_fn = make_growth_fn(model, backend=backend) if w_stab else None

    # INFORMATIVENESS WEIGHTING (P3): weight each dose row by how much old-phase structure the
    # TARGET still has there. A Poincare target goes phase-blind above its own singularity --
    # measured span 0.5 below S*, 0.05 at 6 x S* (REPO_MAP hazard 17) -- and on a log window
    # most rows are up there, so most of the cost is nearly free to satisfy. Split at S* on the
    # Aug-30 campaign: the residual BELOW went 0.199 -> 0.181-0.221 (no better) while ABOVE it
    # went 0.276 -> 0.035-0.057. All of the apparent progress was in the blind region.
    #
    # ONLY LEGAL FOR A PINNED TARGET, AND THIS IS NOT A DETAIL. `w` is safe in circ_cost
    # precisely because the optimizer cannot move it. A PROFILED target is re-registered to the
    # candidate at every evaluation, so its span would become a function of the model -- and a
    # model-dependent WEIGHT is the down-weighting that circ_cost's docstring forbids: the
    # optimizer could discount the informative rows by moving the target. So a profiled target
    # gets no weighting, and says so rather than silently ignoring the request.
    row_w = None
    if row_weight:
        if getattr(tgt, 'k', None) is None:
            raise ValueError(
                "row_weight requires a PINNED target. With a profiled target the weights would "
                "depend on the candidate, which turns a safe weight into a down-weighting the "
                "optimizer can exploit (see fit.target.circ_cost). Pin the target, or pass "
                "row_weight=False.")
        _zt_fixed = radial_z(old, doses, float(tgt.k), float(tgt.psi))
        _sp = _target_span(np.asarray(_zt_fixed))
        # a floor, not a bare span: a row with zero weight contributes nothing at all, and a
        # surface is still expected to be a PTC there. `row_weight_floor` keeps every row in the
        # cost while concentrating it where the target can discriminate.
        w1 = row_weight_floor + (1.0 - row_weight_floor) * (_sp / max(_sp.max(), 1e-30))
        row_w = jnp.asarray(np.broadcast_to(w1[None, :], (n_phase, len(doses))))

    def _surface(v):
        P = model.jax_apply(_theta(v), names)
        x0 = guess(P, y_seed)
        y0, T, res = solver.solve(P, x0)
        full = solver.cycle(P, y0, T, m_amp)
        cyc = full[:, ref_idx]
        amp_lc = (jnp.max(cyc) - jnp.min(cyc)) / jnp.maximum(jnp.abs(jnp.mean(cyc)), 1e-12)

        # IS THIS EVEN A LIMIT CYCLE?
        #
        # REPO_MAP hazard 2: "a small BVP residual does not mean you have a limit cycle...
        # make_orbit_finder tests amplitude, positivity and period band as well; do not bypass
        # it." This function bypassed it, and a radial CMA fit walked straight through the gap:
        # it returned a "cycle" with period 0.159 h, concentrations of order 1e11 AND NEGATIVE,
        # and Floquet mu = 1.0 -- then scored c_ptc = 0.011 on it, better than anything legitimate,
        # and the driver reported RADIALIZED.
        #
        # An unphysical orbit must cost the MAXIMUM, for the same reason a dead cell does (see
        # fit/target.circ_cost): anything else makes running away from the model class the
        # cheapest move available.
        T_nom = float(getattr(model, 'approx_period', None) or 24.0)
        ok_orbit = ((res < 1e-4)
                    & jnp.isfinite(T) & (T > 0.25 * T_nom) & (T < 4.0 * T_nom)
                    & (jnp.min(full) >= NEG_TOL)          # concentrations stay non-negative
                    & jnp.all(jnp.isfinite(full))
                    & (amp_lc > 1e-3))                    # not an equilibrium

        z, ymin = f(P, jnp.concatenate([y0, T[None]]), ph, dz)
        z = z.reshape(nd, npz).T
        ymin = ymin.reshape(nd, npz).T
        fin = jnp.isfinite(z.real) & jnp.isfinite(z.imag) & jnp.isfinite(ymin)
        zs = jnp.where(fin, z, 1.0 + 0j)
        amp = jnp.abs(zs)
        # a bad orbit kills every cell, so the whole surface scores the maximum
        alive = fin & (amp > DEAD_AMP) & (ymin >= NEG_TOL) & ok_orbit
        zu = zs / (amp + eps)
        # per-period growth of a deviation from the cycle; > 1 means the orbit REPELS, so the
        # asymptotic phase the whole PTC is built on does not exist there
        r = growth_fn(P, y0, T) if growth_fn is not None else jnp.array(jnp.nan)
        return P, zu, alive, amp, amp_lc, T, r

    def _parts(v):
        P, zu, alive, amp, amp_lc, T, r_grow = _surface(v)
        zt, aux = tgt(zu, alive, old, doses)
        # ALIVENESS RAMP (P1). `DEAD_AMP = 0.01` means "not literally zero"; it does not mean
        # "carries phase information". At |z| = 0.05 the post-perturbation oscillation is 5% of
        # the intact clock and its Fourier phase is noise -- measured, 4 of the 16 Aug-30 fits
        # had a median |z| of 0.05-0.09 across the whole window and were scored as PTCs anyway
        # (PROJECT_SUMMARY 5.12).
        #
        # It blends TOWARD THE MAXIMUM, never down-weights -- see fit.target.circ_cost. A weight
        # would let the optimizer discount a cell by killing it.
        #
        # WHY A PER-CELL RAMP IS SAFE HERE, WHICH IS NOT OBVIOUS. |z| also falls at a genuine
        # phase singularity, which is the most informative cell on the surface, so a ramp could
        # in principle blank exactly what the fit is for. MEASURED on the rescanned campaign:
        # at seed 9's singularity |z| = 0.997 and at seed 6's the deepest cell is 0.490, while
        # the dead surfaces sit at 0.016-0.09. The band below is 2.5x clear of the deepest
        # genuine dip. `fit.contract` C5 is the backstop if a future model breaks that margin.
        soft = None
        if amp_ramp is not None:
            lo, hi = amp_ramp
            t = jnp.clip((amp - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
            soft = t * t * (3.0 - 2.0 * t)          # smoothstep: C1 at both knees
        c_ptc = circ_cost(zu, zt, alive, soft=soft, w=row_w)
        # one-sided quadratic floor: 0 when healthy, rising as the cycle shrinks. Quadratic
        # rather than linear so it is gentle near the floor and firm well below it.
        a = jnp.maximum(0.0, 1.0 - amp_lc / amp_floor) ** 2
        if osc is not None:
            b, re, res = osc(P, y_fp_seed)
        else:
            b = jnp.array(0.0); re = jnp.array(jnp.nan); res = jnp.array(jnp.nan)
        # STABILITY. Softplus on log r so the term is exactly 0 for a comfortably attracting
        # cycle and rises smoothly as the fit approaches r = 1. Penalising the APPROACH rather
        # than only the crossing is the point: a hard gate at r > 1 gives the optimizer no
        # gradient telling it which way is safe, which is how the PER fit walked onto a
        # repelling orbit while every existing check reported healthy.
        # SQUARED HINGE, NOT SOFTPLUS. Softplus has an exponential tail that never reaches
        # zero: at the base point (growth 0.857, r_max 0.95) it still charged 0.006, which is
        # FIFTEEN TIMES the T1 recovery optimum of 0.000392. A penalty whose tail dwarfs the
        # signal would have made the regression test meaningless. The hinge is exactly 0 below
        # r_max, C1 at the knee, and quadratic above -- the same shape the amplitude floor
        # uses a few lines up.
        if w_stab and growth_fn is not None:
            st = jnp.maximum(0.0, jnp.log(jnp.maximum(r_grow, 1e-12))
                             - jnp.log(r_max)) ** 2
        else:
            st = jnp.array(0.0)
        # BRACKETING BARRIER. Keep the type-1 -> type-0 transition inside the dose window
        # instead of widening the window to chase it. 16 of 16 Aug-30 fits ended with it
        # outside (contract C6), which is what made every c_ptc they reported a statement about
        # a surface with no transition in it.
        #
        # ONE-SIDED AND SQUARED, the same shape as the amplitude floor and the stability term:
        # exactly 0 while the transition is comfortably inside, C1 at the knee, quadratic
        # beyond. It must be exactly 0 at a healthy point or it moves the global minimum, which
        # is the mistake 5.3 records twice (the amplitude floor and the Hopf barrier both
        # charged a perfectly good truth). MEASURED at base: dispersion is 0.700 at the bottom
        # row and 0.146 at the top, so the thresholds below clear it by 0.2 and 0.15.
        #
        # `j_lo` skips a dose-0 row if the grid carries one: at zero dose the PTC is the
        # identity, whose dispersion is 1 by construction, so it would mask a collapse.
        # GATED ON THE ROW BEING ALIVE, and that is not a detail. With every cell dead the
        # resultant is 0 and the dispersion reads 1.0 on EVERY row, so an all-dead surface
        # looks exactly like "the transition is above the ceiling" and the barrier charged it
        # 0.49 for a reason that is not true. A dead surface already costs the maximum through
        # c_ptc; letting a second term charge it again, for the wrong reason, muddies both the
        # value and the gradient. Scaling by the row's alive fraction keeps each term
        # responsible for exactly one failure -- the same separation the aliveness ramp makes.
        # AGGREGATED OVER THE WHOLE WINDOW, NOT READ OFF THE TWO EDGE ROWS.
        #
        # The first version used the bottom and top rows only. It separated the cases correctly
        # but AMPLIFIED THE KNOWN GRADIENT PATHOLOGY: at seed 0, |grad| went 3.0e10 -> 6.1e11
        # when the barrier was switched on. The reason is structural -- `c_ptc` averages over
        # all 280 cells, so per-cell non-smoothness (hazard 11 / 5.4: the coexisting fixed
        # point's basin boundary) averages down, while two rows of 20 cells give it 14x less
        # room to cancel. Meaning the barrier is smooth exactly where the map is, and inherits
        # the map's roughness with less damping everywhere else.
        #
        # The mean dispersion over the window separates the cases just as cleanly -- base
        # 0.512, the fourteen collapsed fits 0.000-0.162, seed 0's upward escape 0.764 -- while
        # averaging over every row.
        if w_brack:
            disp = row_dispersion(zu, alive)
            af = jnp.mean(alive.astype(jnp.float64), axis=0)
            lo_j = 1 if float(doses[0]) == 0.0 else 0   # a dose-0 row is the identity: skip it
            wgt = af[lo_j:]
            D = jnp.sum(wgt * disp[lo_j:]) / jnp.maximum(jnp.sum(wgt), 1e-12)
            # gated on the surface being alive at all: with every cell dead the resultant is 0
            # and D reads 1.0, which looks exactly like "the transition is above the ceiling".
            # A dead surface already costs the maximum through c_ptc; charging it again here,
            # for a reason that is not true, muddies both the value and the gradient.
            gate = jnp.mean(alive.astype(jnp.float64))
            br = gate * (jnp.maximum(0.0, brack_lo - D) ** 2
                         + jnp.maximum(0.0, D - brack_hi) ** 2)
        else:
            br = jnp.array(0.0)
        total = c_ptc + w_osc * b + w_amp * a + w_stab * st + w_brack * br
        return dict(total=total, c_ptc=c_ptc, osc=b, amp_pen=a, amp_lc=amp_lc, period=T,
                    alive_frac=jnp.mean(alive.astype(jnp.float64)), re_lambda=re,
                    fp_res=res, stab=st, growth=r_grow, brack=br, **aux)

    def _residual(v):
        """The cost as a RESIDUAL VECTOR r with |r|^2 == total.

        WHY THIS EXISTS. The objective is a sum of squares wearing a scalar's clothing:
        (1 - cos 2*pi*d)/2 = sin^2(pi*d), so `total` is mean(r_i^2) over ~100 cells. Handing
        optimizers only the scalar throws away the per-cell structure, and that structure is
        exactly what Gauss-Newton methods use to SEE the anisotropy of a sloppy landscape
        instead of having to infer it.

        It matters here specifically. L-BFGS builds curvature by accumulating gradient
        differences, so a single pathological evaluation contaminates its Hessian estimate for
        the rest of the run -- and gradients in this landscape spike to 1e25 where a trajectory
        grazes the coexisting equilibrium's basin boundary. Levenberg-Marquardt rebuilds J^T J
        from scratch at every step, so one bad point costs one rejected step and nothing more.

        Layout, with |r|^2 reproducing `total` exactly:
            cells    (zu - zt) / (2 sqrt(N)), real and imaginary parts. |zu - zt|^2 =
                     2(1 - cos d), so summing gives mean(1 - cos d)/2 = c_ptc.
            dead     magnitude 2 (antipodal, the maximum), so a dead cell contributes 1/N --
                     the same maximum penalty the scalar form assigns.
            osc/amp  sqrt(w * term), one entry each.
            ridge    sqrt(ridge) * v, one entry per free direction (see `ridge` below).
        """
        P, zu, alive, amp, amp_lc, T, _r = _surface(v)
        zt, aux = tgt(zu, alive, old, doses)
        fin = jnp.isfinite(zu.real) & jnp.isfinite(zu.imag)
        zs = jnp.where(fin, zu, 1.0 + 0j)
        d = zs - zt
        # dead cells: the maximally-wrong unit-phase difference, magnitude 2
        d = jnp.where(alive, d, 2.0 + 0j)
        N = d.size
        scale = 1.0 / (2.0 * jnp.sqrt(N))
        r = jnp.concatenate([jnp.real(d).ravel() * scale, jnp.imag(d).ravel() * scale])

        a = jnp.maximum(0.0, 1.0 - amp_lc / amp_floor) ** 2
        if osc is not None:
            b, _re, _res = osc(P, y_fp_seed)
        else:
            b = jnp.array(0.0)
        extra = [jnp.sqrt(jnp.maximum(w_osc * b, 0.0))[None],
                 jnp.sqrt(jnp.maximum(w_amp * a, 0.0))[None]]
        if ridge:
            extra.append(jnp.sqrt(ridge) * v)
        return jnp.concatenate([r] + extra)

    # TWO compiles, not three. `total` is just a field of `parts`, and this graph is expensive
    # to trace -- compiling a separate scalar version of it cost about a third of the startup
    # time for nothing. The dict lookup is free next to a 160-cell ODE solve.
    # jacfwd of a scalar returns the gradient. Forward mode integrates the variational equation
    # ALONGSIDE the state, so it never runs the dynamics backward -- see engine/flow.GRAD_MODES.
    _grad = jax.jit((jax.jacfwd if grad_mode == 'fwd' else jax.grad)(
        lambda v: _parts(v)['total']))
    _parts_j = jax.jit(_parts)
    _resid_j = jax.jit(_residual)
    # forward-mode: the residual has many outputs and only n_free inputs, so jacfwd costs
    # n_free JVPs -- about one reverse-mode gradient, for the whole Jacobian.
    _jac_j = jax.jit(jax.jacfwd(_residual))

    def _total(v):
        return _parts_j(v)['total']

    def parts(v):
        d = _parts_j(jnp.asarray(v, jnp.float64))
        return {k: float(np.asarray(x)) for k, x in d.items()}

    # Jitted: a conditioning study finite-differences this 2*n_free times, and in eager mode
    # each call re-dispatches the whole vmapped solve one primitive at a time.
    _surface_j = jax.jit(_surface)

    def surface(v):
        P, zu, alive, amp, amp_lc, T, _r = _surface_j(jnp.asarray(v, jnp.float64))
        return np.asarray(zu), np.asarray(alive), np.asarray(amp)

    def residual(v):
        return np.asarray(_resid_j(jnp.asarray(v, jnp.float64)))

    def jac(v):
        return np.asarray(_jac_j(jnp.asarray(v, jnp.float64)))

    return dict(names=names, z_base=z_base, B=B, gauge=g, n_free=n_free,
                grad_mode=grad_mode, backend=backend, re_nominal=re_nom, ridge=ridge,
                residual=residual, jac=jac,
                v0=np.zeros(n_free), doses=np.asarray(doses), old=np.asarray(old),
                amp_base=amp_base, amp_floor=amp_floor, target=tgt.name,
                theta=lambda v: np.asarray(_theta(jnp.asarray(v, jnp.float64))),
                total=lambda v: float(_total(jnp.asarray(v, jnp.float64))),
                # WHOLE POPULATION IN ONE LAUNCH. CMA has a parallel axis that a quasi-Newton
                # method does not -- popsize mutually independent parameter vectors per
                # generation -- and evaluating them in a Python loop throws it away. Batched,
                # one generation is popsize * n_cells integrations in a single call (12 * 280 =
                # 3360) instead of popsize calls of n_cells, which is the difference between
                # poor and good GPU occupancy. Returns a jnp array, NOT floats: the caller
                # converts, because `float()` on a traced value is what stopped `total` itself
                # from being vmappable.
                #
                # NOT YET FIT TO OPTIMISE AGAINST. MEASURED, 8 x 5 grid, population 4: the
                # batched values differ from the sequential ones by up to 9.955e-04, on a cost
                # whose working scale is 0.1-0.5. That is far too large to be rounding, and it
                # lands exactly where it does the most damage -- CMA RANKS its population, and
                # late in a run the gaps between good members are within an order of magnitude
                # of 1e-3, so a batched generation could reorder itself relative to the true
                # costs.
                #
                # Leading suspicion: vmapped lanes share the diffrax `while_loop` termination
                # and `max_steps` accounting, so a lane's effective step budget depends on its
                # NEIGHBOURS -- i.e. a member's cost would depend on who else was sampled in
                # the same generation. Explain the 1e-3 before wiring this into `search.cma`;
                # a self-consistency assert (batched == looped to ~1e-12) is the gate.
                #
                # It also runs 2.2x SLOWER than the loop on CPU, as expected: the CPU cannot
                # exploit the population axis, and lockstep stepping makes every lane pay for
                # the slowest. The value here is purely that the graph TRACES, which is what a
                # GPU would need.
                total_batch=jax.jit(jax.vmap(_total)),
                grad=lambda v: np.asarray(_grad(jnp.asarray(v, jnp.float64))),
                parts=parts, surface=surface, solver=solver, ptc_fn=f)


# --------------------------------------------------------------------------- #
#  Self-tests
# --------------------------------------------------------------------------- #
def _build(name='almeida', target='BMAL1', n_phase=12, n_dose=6, backend='diffrax',
           dt=0.02, w_osc=0.2, max_factor=6.0):
    """The configuration a FIT actually uses -- so the selftest and gradcheck test that.

    Deliberately `fit.doses.fit_dose_grid` and not the wide characterization grid. The two
    differ by design (see fit/doses.py): the characterization grid runs to ~18x S_crit because
    the twist lives above S_crit, while a fit is capped at `max_factor * S_crit`. Checking the
    gradient on a grid the fit will never use would answer the wrong question in both
    directions -- it could fail on doses that are never fitted, or pass on a grid that omits
    ones that are."""
    from models import get_model
    from fit.doses import fit_dose_grid
    model = get_model(name)
    doses, s_crit = fit_dose_grid(name, target, 'instant', max_factor, n_dose)
    C = make_cost(model, target, doses, RadialTarget(), n_phase=n_phase, mode='instant',
                  backend=backend, dt=dt, w_osc=w_osc)
    C['s_crit'] = s_crit
    return model, C


def selftest(name='almeida', target='BMAL1'):
    """THE load-bearing assertion: a Hopf-collapsed parameter set must score NEAR-MAXIMAL.

    If this ever fails, the Mirsky degeneracy has been rebuilt -- a cost whose global optimum
    is a dead oscillator -- and no result from this pipeline should be believed.
    """
    model, C = _build(name, target)
    print("=" * 76)
    print(f"COST SELF-TEST -- {name}/{target}, {C['n_free']} free directions "
          f"(of {len(C['names'])} parameters)")
    print("=" * 76)

    base = C['parts'](C['v0'])
    print(f"  base            total={base['total']:.4f} c_ptc={base['c_ptc']:.4f} "
          f"osc={base['osc']:.4f} amp={base['amp_pen']:.4f} "
          f"amp_lc={base['amp_lc']:.3f} alive={base['alive_frac']:.2f}")
    ok = np.isfinite(base['total']) and base['alive_frac'] > 0.8

    # Drive the clock toward Hopf by scanning one direction until the cycle collapses, then
    # assert the cost went UP. Which parameter does it is model-specific; what matters is that
    # SOME collapse exists and that it is punished.
    worst, best_v = None, None
    for i in range(C['n_free']):
        for s in (+1.0, -1.0):
            v = np.zeros(C['n_free']); v[i] = s * 1.2
            p = C['parts'](v)
            if p['amp_lc'] < 0.25 * C['amp_base']:
                if worst is None or p['amp_lc'] < worst['amp_lc']:
                    worst, best_v = p, v.copy()
    if worst is None:
        print("  no collapsing direction found at |v|=1.2 -- widening")
        for i in range(C['n_free']):
            v = np.zeros(C['n_free']); v[i] = 3.0
            p = C['parts'](v)
            if worst is None or p['amp_lc'] < worst['amp_lc']:
                worst, best_v = p, v.copy()
    print(f"  most collapsed  total={worst['total']:.4f} c_ptc={worst['c_ptc']:.4f} "
          f"osc={worst['osc']:.4f} amp={worst['amp_pen']:.4f} "
          f"amp_lc={worst['amp_lc']:.3f} alive={worst['alive_frac']:.2f}")
    print(f"  base amp_lc = {C['amp_base']:.3f}, floor = {C['amp_floor']:.3f}")

    collapsed_worse = worst['total'] > base['total']
    print(f"  collapsed cost > base cost: {collapsed_worse}  "
          f"({worst['total']:.4f} vs {base['total']:.4f})   <- THE invariant")
    ok &= bool(collapsed_worse)

    # and an all-dead surface must sit at the ceiling
    print(f"  a fully dead surface scores c_ptc = 1.0 by construction "
          f"(fit.target.circ_cost, asserted in its own selftest)")
    print(f"  [{'PASS' if ok else 'FAIL'}] cost punishes the degenerate optimum")
    return ok


def gradcheck(name='almeida', target='BMAL1', n=6, h=1e-4, seed=0, backend='diffrax'):
    """T0.5: autodiff vs central finite differences on the ADAPTIVE backend.

    REPO_MAP hazard 1 is that autodiffing the fixed-step RK4 PTC at nontrivial dose produced
    |J| up to 3.95e83 against a true response of 0.006 -- and it was found only after it had
    invalidated a table of results. This check would have caught it in a minute. It gates every
    gradient-driven run; if it fails, use --optimizer cma and say so in the writeup.
    """
    model, C = _build(name, target, backend=backend)
    rng = np.random.default_rng(seed)
    v0 = np.zeros(C['n_free'])
    g = C['grad'](v0)
    idx = rng.choice(C['n_free'], size=min(n, C['n_free']), replace=False)
    print("=" * 76)
    print(f"GRADIENT CHECK -- {name}/{target}, backend={backend}, h={h}")
    print("=" * 76)
    print(f"  |grad| = {np.linalg.norm(g):.4e}   max|g_i| = {np.max(np.abs(g)):.4e}")
    if not np.all(np.isfinite(g)):
        print("  FAIL: gradient is not finite")
        return False
    if np.max(np.abs(g)) > 1e6:
        print(f"  FAIL: gradient magnitude {np.max(np.abs(g)):.3e} is not credible "
              f"(hazard 1 signature)")
        return False
    ok = True
    print(f"  {'i':>3s} {'autodiff':>14s} {'central diff':>14s} {'rel err':>10s}")
    for i in idx:
        vp = v0.copy(); vp[i] += h
        vm = v0.copy(); vm[i] -= h
        fd = (C['total'](vp) - C['total'](vm)) / (2 * h)
        rel = abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-8)
        print(f"  {i:3d} {g[i]:14.6e} {fd:14.6e} {rel:10.2e}")
        ok &= rel < 5e-2
    print(f"  [{'PASS' if ok else 'FAIL'}] autodiff agrees with finite differences")
    return ok


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='PTC fitting objective')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--gradcheck', action='store_true')
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--target', default='BMAL1')
    ap.add_argument('--backend', default='diffrax', choices=('rk4', 'diffrax'))
    ap.add_argument('-n', type=int, default=6)
    a = ap.parse_args(argv)
    ok = True
    if a.selftest or not a.gradcheck:
        ok &= selftest(a.model, a.target)
    if a.gradcheck:
        ok &= gradcheck(a.model, a.target, n=a.n, backend=a.backend)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
