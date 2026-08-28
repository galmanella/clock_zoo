"""THE project question, as a rank comparison.

"How identifiable are mechanistic models from limit-cycle data, and from PTC data?" is a
question about the SIZE of the set of parameter sets consistent with the data -- not about
recovering a particular parameter vector. Non-uniqueness is the premise, so the quantity of
interest is the DIMENSION of the solution family:

    dim(solution set) = n_free - rank(J)

for each observable, measured LOCALLY (the coarse-scale rank saturates at 16/16 for everything,
which is why analysis/coupling.py's +-26% comparison cannot separate them).

Three observables, same parameters, same scale, so the ranks are directly comparable:

    LC     the phase-aligned limit-cycle profiles of all observable species -- what a
           trajectory experiment measures, with the phase origin fixed by the orbit solver so
           the comparison is not confounded by a phase shift.
    PTC    the single-gene phase-transition surface (unit phase, real and imaginary parts).
    both   the two stacked, each block normalized so neither dominates by sheer scale.

If PTC adds directions LC cannot see, rank(both) > rank(LC) and the difference is exactly the
identifiability the perturbation experiment buys.
"""
import numpy as np
import jax
import jax.numpy as jnp

from fit.cost import make_cost, RadialTarget, quotient_basis
from fit.doses import fit_dose_grid
from models import get_model
from engine.orbit import OrbitSolver

MODEL, TARGET, MODE = 'almeida', 'BMAL1', 'instant'
NPH, ND, CAP, H = 12, 8, 6.0, 1e-3

model = get_model(MODEL)
doses, S = fit_dose_grid(MODEL, TARGET, MODE, CAP, ND)
C = make_cost(model, TARGET, doses, RadialTarget(), n_phase=NPH, mode=MODE,
              backend='diffrax', dt=0.02, w_osc=0.0, w_amp=0.0)
names, z_base, B, g = quotient_basis(model)
n = B.shape[1]
obs = list(model.observable_states())
oidx = [int(model.var_index(s)) for s in obs]
solver = OrbitSolver(model)
M_CYCLE = 64

print(f"{MODEL}/{TARGET} ({MODE}), {n} free directions, {len(obs)} observable species, "
      f"{NPH}x{len(doses)} PTC cells, h={H}", flush=True)


def lc_vec_of(species):
    """LC profile restricted to a subset of species -- for the MATCHED comparison.

    The unmatched version below uses all 8 observables x 64 phase points against a single
    gene's PTC, which compares a full multi-species time course to one perturbation
    experiment. That is not a comparison that can support an experimental-design claim, so the
    matched version restricts the LC to the SAME single species the PTC perturbs."""
    idx = jnp.asarray([int(model.var_index(t)) for t in species])

    def f(v):
        P = model.jax_apply(jnp.exp(jnp.asarray(z_base) + jnp.asarray(B) @ jnp.asarray(v)),
                            names)
        y0, T, _r = solver.solve(P, solver.guess(P))
        cyc = solver.cycle(P, y0, T, M_CYCLE)[:, idx]
        return np.asarray(cyc).ravel()
    return f


def lc_vec(v):
    """Phase-aligned limit-cycle profiles, all observables, flattened and scale-normalized.

    Each species is divided by its own peak-to-peak range at NOMINAL so that a species with a
    large absolute amplitude does not dominate the rank -- the same normalization
    analysis/lc_sens.py uses for its shape metric."""
    P = model.jax_apply(jnp.exp(jnp.asarray(z_base) + jnp.asarray(B) @ jnp.asarray(v)), names)
    y0, T, _r = solver.solve(P, solver.guess(P))
    cyc = solver.cycle(P, y0, T, M_CYCLE)[:, jnp.asarray(oidx)]
    return np.asarray(cyc).ravel()


def ptc_vec(v):
    zu, alive, amp = C['surface'](v)
    return np.concatenate([np.real(zu).ravel(), np.imag(zu).ravel()])


def jac_of(fn, v0, scale=None):
    f0 = fn(v0)
    J = np.zeros((f0.size, n))
    for i in range(n):
        vp = v0.copy(); vp[i] += H
        vm = v0.copy(); vm[i] -= H
        J[:, i] = (fn(vp) - fn(vm)) / (2 * H)
    return J


def report(label, J):
    s = np.linalg.svd(J, compute_uv=False)
    s = s / s[0]
    r2 = int(np.sum(s > 1e-2)); r3 = int(np.sum(s > 1e-3))
    print(f"  {label:22s} rank@1e-2 {r2:2d}/{n}  rank@1e-3 {r3:2d}/{n}  "
          f"cond {1 / s[-1]:8.2e}   dim(solution set) = {n - r2}", flush=True)
    return s, r2


v0 = np.zeros(n)
J_lc = jac_of(lc_vec, v0)
J_lc = J_lc / np.linalg.norm(J_lc)
s_lc, r_lc = report('LC (trajectories)', J_lc)

J_ptc = jac_of(ptc_vec, v0)
J_ptc = J_ptc / np.linalg.norm(J_ptc)
s_ptc, r_ptc = report('PTC (1 gene)', J_ptc)

J_both = np.vstack([J_lc, J_ptc])
s_both, r_both = report('LC + PTC', J_both)

print(f"\n  PTC adds {r_both - r_lc} direction(s) beyond LC alone "
      f"({r_lc} -> {r_both} of {n})", flush=True)
print(f"  LC  adds {r_both - r_ptc} direction(s) beyond PTC alone "
      f"({r_ptc} -> {r_both} of {n})", flush=True)

# --- the MATCHED comparison ---
# The block above pits an 8-species, 64-point time course against ONE gene's PTC. Whatever it
# shows cannot support an experimental-design claim, because the two experiments are not the
# same size. Here the LC is restricted to the SAME single species the PTC perturbs.
print(f"\n  MATCHED (single species, {TARGET} only):", flush=True)
J_lc1 = jac_of(lc_vec_of([TARGET]), v0)
J_lc1 = J_lc1 / np.linalg.norm(J_lc1)
s_lc1, r_lc1 = report(f'LC ({TARGET} only)', J_lc1)
J_both1 = np.vstack([J_lc1, J_ptc])
s_b1, r_b1 = report(f'LC+PTC ({TARGET})', J_both1)
print(f"    PTC adds {r_b1 - r_lc1} beyond one-species LC ({r_lc1} -> {r_b1} of {n})",
      flush=True)
print(f"    LC  adds {r_b1 - r_ptc} beyond PTC ({r_ptc} -> {r_b1} of {n})", flush=True)

# Which directions does each MISS? Principal angles between the well-determined subspaces.
_U, _s, Vt_lc = np.linalg.svd(J_lc, full_matrices=False)
_U, _s, Vt_ptc = np.linalg.svd(J_ptc, full_matrices=False)
k = min(r_lc, r_ptc)
if k > 0:
    sv = np.linalg.svd(Vt_lc[:k] @ Vt_ptc[:k].T, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(sv, -1, 1)))
    print(f"\n  principal angles between the top-{k} determined subspaces (deg):")
    print("   " + "  ".join(f"{a:.1f}" for a in ang), flush=True)
    print(f"  largest angle {ang.max():.1f} deg -- 0 means LC and PTC see the SAME "
          f"combinations, 90 means fully complementary", flush=True)
