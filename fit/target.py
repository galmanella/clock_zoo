"""
fit/target.py
=============
The analytic radial-isochron PTC -- the phase-transition surface of a Poincare oscillator whose
isochrons are exact rays -- and the machinery to align it to a model surface for free.

    python -m fit.target --selftest

THE TARGET
    A Poincare oscillator sits on the unit circle at exp(2i*pi*theta). An INSTANT kick of size
    `d` along a fixed direction `psi` lands it at

        z = exp(2i*pi*theta) + d * exp(2i*pi*psi)

    and because the isochrons are radial, the new asymptotic phase is just arg(z). So

        new_phase(theta, dose) = arg( exp(2i*pi*theta) + k*dose*exp(2i*pi*psi) ) / 2pi

    This is exact, not simulated: it is the ideal that "radial isochrons" means, and it is
    already validated in this repo -- analysis/winding._selftest builds the psi=0 case, confirms
    winding +1 below k*dose=1 and 0 above, and recovers the singularity at (1.0, 0.5) to within
    the grid (detected 0.947 / 0.510).

    Its structure: winding +1 for k*dose < 1, winding 0 for k*dose > 1, a single phase
    singularity at dose = 1/k and old phase psi + 0.5, and ZERO twist (the stable fixed point
    sits at the same phase for every dose, which is what makes the isochrons radial).

THE TWO NUISANCE PARAMETERS, AND WHY THEY ARE FREE TO PROFILE OUT
    k    the dose scale. The target's dose axis has no natural units; only the MODEL knows what
         a dose of 25 means. Equivalently k sets the target's S_crit to 1/k.
    psi  the kick direction, which places the singularity at old phase psi + 0.5. A Poincare
         oscillator has no distinguished phase, so nothing physical picks psi out.

    Neither is a property of the model, so fitting them is not fitting the model, and pinning
    them would ask the wrong question -- "is the model radial AND oriented exactly so", rather
    than "is the model radial". They are therefore profiled out.

    Doing so is FREE. Both only relabel the TARGET's axes, and the target is a closed-form
    expression, so scanning them costs a few array evaluations and NOT ONE extra ODE solve. A
    48 x 48 scan over a 32 x 24 grid is ~1.8M flops against ~10^9 for a single PTC evaluation.

    Crucially the identity at dose 0 survives for any (k, psi): at dose 0 the expression is
    arg(exp(2i*pi*theta)) = theta exactly. That matters because engine/ptc.py calibrates the
    model PTC to be exactly the identity at dose 0, so the two agree there by construction and
    any residual is real signal rather than a convention mismatch.

GRADIENTS THROUGH THE PROFILE (Danskin)
    The profiled cost is min over (k, psi) of an inner objective. By Danskin's theorem its
    gradient with respect to the MODEL parameters equals the partial gradient at the inner
    optimum, holding (k*, psi*) fixed. So `profile` returns stop_gradient-ed optima and the
    caller re-evaluates at them -- correct, and no differentiation through the argmin.
"""
import numpy as np

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import lax


def radial_z(old, doses, k, psi):
    """Unit complex phase of the radial target on the [n_phase, n_dose] grid.

    Returned as a complex exponential rather than a phase in [0,1) because every consumer wants
    a circular comparison, and doing it on unit vectors avoids a wrap-around branch cut.
    """
    old = jnp.asarray(old)[:, None]
    d = jnp.asarray(doses)[None, :]
    z = jnp.exp(2j * jnp.pi * old) + k * d * jnp.exp(2j * jnp.pi * psi)
    return z / (jnp.abs(z) + 1e-300)


def radial_phase(old, doses, k, psi):
    """The radial target as new_phase in [0, 1), shape [n_phase, n_dose]."""
    return (jnp.angle(radial_z(old, doses, k, psi)) / (2 * jnp.pi)) % 1.0


def circ_cost(zm, zt, alive):
    """Mean circular residual between two unit-phase fields, with dead cells at the MAXIMUM.

        per cell = (1 - cos(2*pi*(phi_model - phi_target))) / 2   in [0, 1]

    which for unit vectors is (1 - Re(zm * conj(zt)))/2 -- no angles, no branch cut.

    DEAD CELLS SCORE 1.0. Not dropped, not down-weighted, not NaN. This is the single most
    important line in the fit: a cost that excludes or neutralizes unusable cells is minimized
    by killing the oscillator, which is exactly how the Mirsky radialization run reached 0.159
    while destroying the clock. Here, degrading the oscillation makes cells dead and each dead
    cell costs the maximum, so the collapse is the worst point in the space rather than the
    best.
    """
    # Sanitize BEFORE the select: jnp.where propagates NaN through the UNSELECTED branch in
    # reverse mode, so a NaN left in `zm` would poison the gradient even where alive is False.
    fin = jnp.isfinite(zm.real) & jnp.isfinite(zm.imag)
    zs = jnp.where(fin, zm, 1.0 + 0j)
    per = 0.5 * (1.0 - jnp.real(zs * jnp.conj(zt)))
    return jnp.mean(jnp.where(alive, per, 1.0))


def profile(zm, alive, old, doses, n_k=48, n_psi=48, k_lo=None, k_hi=None, refine=2):
    """Best (k, psi) for a model surface, and the cost there.

    `zm` is the model's unit phase field [n_phase, n_dose] (complex), `alive` its boolean
    usability mask. Returns (k, psi, cost) with k and psi stop_gradient-ed, per Danskin.

    Two-stage grid: a coarse sweep, then `refine` rounds zooming on the winner. A grid rather
    than a solver because the objective is periodic in psi and multimodal in k near the
    singularity, so a local method started badly would find the wrong branch -- and the whole
    scan is analytic, so buying robustness with brute force costs nothing measurable.

    The default k range brackets every dose on the grid: k = 1/dose puts the target's
    singularity exactly at that dose, so [1/max(dose), 1/min(dose)] covers "the transition is
    anywhere in the sampled window" plus a decade of margin either side.
    """
    d = np.asarray(doses, float)
    dpos = d[d > 0]
    lo = k_lo if k_lo is not None else 0.1 / dpos.max()
    hi = k_hi if k_hi is not None else 10.0 / dpos.min()

    def cost_at(k, psi):
        return circ_cost(zm, radial_z(old, doses, k, psi), alive)

    vcost = jax.vmap(jax.vmap(cost_at, in_axes=(None, 0)), in_axes=(0, None))

    lk_lo, lk_hi = jnp.log(lo), jnp.log(hi)
    p_lo, p_hi = 0.0, 1.0
    best_k, best_psi, best_c = None, None, None
    for r in range(refine + 1):
        ks = jnp.exp(jnp.linspace(lk_lo, lk_hi, n_k))
        ps = jnp.linspace(p_lo, p_hi, n_psi, endpoint=(r > 0))
        C = vcost(ks, ps)
        i, j = jnp.unravel_index(jnp.argmin(C), C.shape)
        best_k, best_psi, best_c = ks[i], ps[j], C[i, j]
        # zoom to +/- 2 cells, so the bracket always contains the winner
        dk = (lk_hi - lk_lo) / max(n_k - 1, 1)
        dp = (p_hi - p_lo) / max(n_psi - 1, 1)
        lk_lo, lk_hi = jnp.log(best_k) - 2 * dk, jnp.log(best_k) + 2 * dk
        p_lo, p_hi = best_psi - 2 * dp, best_psi + 2 * dp
    return (lax.stop_gradient(best_k), lax.stop_gradient(best_psi % 1.0),
            lax.stop_gradient(best_c))


def target_features(k, psi):
    """(S_crit, phi_sing) the target is asking the model to produce."""
    return float(1.0 / k), float((psi + 0.5) % 1.0)


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #
def selftest():
    """The target must reproduce the analytic facts winding._selftest already checks, and the
    profile must recover a known (k, psi) exactly from a synthetic surface."""
    from analysis import winding as W

    ok = True
    print("=" * 76)
    print("RADIAL TARGET SELF-TEST")
    print("=" * 76)

    M, J = 48, 30
    old = np.arange(M) / M
    doses = np.geomspace(0.05, 5.0, J)

    # 1. structure: winding +1 below 1/k, 0 above, one singularity at (1/k, psi+0.5)
    k_true, psi_true = 1.0, 0.0
    ptc = np.asarray(radial_phase(old, doses, k_true, psi_true))
    w = W.winding_vs_dose(ptc)
    s = W.find_singularities(old, doses, ptc)
    lo = doses[w == 1].max(); hi = doses[w == 0].min()
    print(f"  winding transition bracketed by [{lo:.3f}, {hi:.3f}] (expect 1/k = 1.0)")
    print(f"  {len(s)} singularity (expect 1)"
          + (f" at dose {s[0]['dose']:.3f}, phi {s[0]['phi']:.3f} (expect 1.000, 0.500)"
             if s else ""))
    ok &= (lo < 1.0 < hi) and len(s) == 1

    # 2. identity at dose 0, for ANY (k, psi) -- the convention that must match engine/ptc.py
    for kk, pp in ((0.04, 0.3), (7.0, 0.85)):
        z0 = np.asarray(radial_phase(old, np.array([0.0]), kk, pp))[:, 0]
        err = float(np.max(np.abs(((z0 - old + 0.5) % 1.0) - 0.5)))
        print(f"  dose 0 is the identity for (k={kk}, psi={pp}): max err {err:.2e}")
        ok &= err < 1e-12

    # 3. twist is exactly zero -- the defining property of radial isochrons
    tw = W.twist_curve(old, doses, ptc)
    print(f"  total twist = {W.total_twist(tw):.2e} cyc (expect 0 -- isochrons are rays)")
    ok &= W.total_twist(tw) < 1e-6

    # 4. profile recovers a planted (k, psi) from a synthetic surface
    for k_t, psi_t in ((0.04, 0.31), (1.7, 0.86)):
        zm = radial_z(old, doses, k_t, psi_t)
        alive = jnp.ones(zm.shape, bool)
        k, psi, c = profile(zm, alive, old, doses)
        print(f"  profile: planted (k={k_t:.4g}, psi={psi_t:.3f}) -> "
              f"recovered ({float(k):.4g}, {float(psi):.3f}), cost {float(c):.2e}")
        ok &= abs(float(k) / k_t - 1) < 0.02 and abs((float(psi) - psi_t + 0.5) % 1 - 0.5) < 0.02
        ok &= float(c) < 1e-6

    # 5. dead cells cost the maximum -- the anti-degeneracy invariant
    zm = radial_z(old, doses, 1.0, 0.0)
    c_live = float(circ_cost(zm, radial_z(old, doses, 1.0, 0.0), jnp.ones(zm.shape, bool)))
    c_dead = float(circ_cost(zm, radial_z(old, doses, 1.0, 0.0), jnp.zeros(zm.shape, bool)))
    print(f"  perfect match alive -> {c_live:.3f} (expect 0); all-dead -> {c_dead:.3f} "
          f"(expect 1, NOT 0)")
    ok &= c_live < 1e-12 and abs(c_dead - 1.0) < 1e-12

    print(f"  [{'PASS' if ok else 'FAIL'}] radial target")
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.parse_args()
    raise SystemExit(0 if selftest() else 1)


# --------------------------------------------------------------------------- #
#  Soft singularity location -- a FEATURE that does not suffer the spiral problem
# --------------------------------------------------------------------------- #
def soft_singularity(z_unit, amp, old, doses, eps=0.25):
    """Smooth (phi, log-dose) location of the phase singularity, from the amplitude field.

    WHY A FEATURE TERM IS NEEDED AT ALL
        The pointwise cost is not monotone in singularity displacement. MEASURED: a parameter
        set whose singularity sits at EXACTLY the truth's (S_crit, phi*) = (5.458, 0.604) scores
        c_ptc = 0.0616, worse than one sitting at (6.979, 0.521) which scores 0.0260. Moving the
        defect onto its target made the cost go up.

        The reason is geometric. Around a singularity the PTC is a spiral, so comparing two
        surfaces is like cross-correlating two spirals: alignment oscillates with displacement
        rather than improving monotonically, and every half-turn of relative rotation is a local
        minimum. With a denser spiral -- a more twisted isochron field, or a finer grid -- there
        are more such minima, and the surface becomes rugged in a way no local method handles.
        The same argument applies to the strongly twisted type-0 region at high dose, which is
        where the L-BFGS solution got stuck.

    WHY THIS ESTIMATOR
        `analysis.winding.detect_grid` gives (S_crit, phi*) but it is a discrete plaquette
        search: quantized to the dose grid (on a 16-dose grid it returns one of two values) and
        not differentiable. As a cost term that is a staircase, which trades a rugged landscape
        for a flat one.

        The amplitude field is the smooth alternative. |z| -> 0 AT the singularity and nowhere
        else, so a centroid weighted by exp(-(|z|/eps)^2) locates it continuously, without a
        grid, and differentiably. `old` is circular, so the phi centroid is taken on the unit
        circle rather than as an arithmetic mean.

    THE TEMPERATURE MUST ADAPT TO THE AMPLITUDE RANGE, NOT BE FIXED.
        A fixed eps = 0.06 fails outright. MEASURED on the BMAL1/instant grid: |z| never falls
        below 0.79, because the singularity sits BETWEEN grid cells and the amplitude dip is
        never resolved. Every weight exp(-(0.79/0.06)^2) then underflows to zero and the
        centroid degenerates to the grid's own log-mean -- the estimator returned dose 23.690 at
        every point along a path where the singularity was demonstrably moving.

        So the scale is taken from the surface: the softmin is measured from the MINIMUM
        amplitude present, with a temperature proportional to the observed spread. That tracks
        the dip wherever it is and however shallow it is.

    Returns (phi, log_dose, contrast). `contrast` = (max - min)/max of the amplitude field: how
    pronounced the dip is at all. Near zero means there is no singularity on this grid and the
    location is meaningless, so a caller must check it rather than trusting the centroid.
    """
    a = jnp.asarray(amp)
    a_min, a_max = jnp.min(a), jnp.max(a)
    spread = jnp.maximum(a_max - a_min, 1e-12)
    temp = jnp.maximum(eps * spread, 1e-9)
    w = jnp.exp(-((a - a_min) / temp) ** 2)
    w = w / (jnp.sum(w) + 1e-300)
    zc = jnp.sum(w * jnp.exp(2j * jnp.pi * jnp.asarray(old)[:, None]))
    phi = (jnp.angle(zc) / (2 * jnp.pi)) % 1.0
    ld = jnp.sum(w * jnp.log(jnp.asarray(doses))[None, :])
    return phi, ld, spread / jnp.maximum(a_max, 1e-12)


def singularity_cost(amp, amp_target, old, doses, eps=0.25):
    """Squared distance between two surfaces' soft singularity locations.

    Circular in phi, log-scaled in dose (a factor-of-two error at dose 5 and at dose 100 are
    the same error). Returns 0 when either surface has no singularity to speak of, so this term
    stays silent rather than inventing a target when there is nothing to match.
    """
    p1, l1, m1 = soft_singularity(None, amp, old, doses, eps)
    p2, l2, m2 = soft_singularity(None, amp_target, old, doses, eps)
    dphi = jnp.abs(((p1 - p2 + 0.5) % 1.0) - 0.5)
    dld = (l1 - l2) / jnp.maximum(jnp.abs(l2), 1e-9)
    # `contrast` now, not raw mass: a surface with no amplitude dip has no singularity
    # to locate, and matching one that is not there is worse than staying silent.
    live = (m1 > 1e-3) & (m2 > 1e-3)
    return jnp.where(live, dphi ** 2 + dld ** 2, 0.0)
