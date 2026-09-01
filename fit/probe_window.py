"""
fit/probe_window.py
===================
DOES A RELATIVE DOSE WINDOW GIVE A CONTINUOUS COST? -- the decision probe for direction 2.

    $PY -m fit.probe_window

WHAT DIRECTION 2 PROPOSES
    Instead of a dose window fixed in absolute units, define it RELATIVE to each candidate's own
    transition: [a, b] x S*(v). That removes the escape route entirely -- moving S* no longer
    changes what you are scored on -- and it matches what a radialization is about, since twist
    lives above S*. It is legitimate for a SHAPE question and not for fitting real data, where
    dose is a measured physical quantity.

    The stated risk was that it makes the cost discontinuous, because the window's position
    would jump as the located S* jumps. This measures that, and it measures ONLY that: the cost
    itself is not built here, because if S*(v) is discontinuous nothing downstream can be
    continuous either, and if it is smooth the rest follows.

WHY NOT `detect_grid` OR `soft_singularity` TO LOCATE IT
    `detect_grid` is quantised to the dose grid -- a staircase by construction. `soft_singularity`
    reads the amplitude dip, so it cannot see a defect that has already left the grid, which is
    the state most of the Aug-30 fits are in. This uses the dose at which per-row circular
    DISPERSION crosses 0.5, interpolated in log dose: smooth, defined everywhere, and the same
    observable the bracketing barrier uses.

    IT IS A PROXY, NOT S_crit, AND THE OFFSET IS REAL. At the base point it reads 71.8 where the
    singularity-based S_crit is 32.79 -- a factor 2.2. Consistent, so it works as the anchor of
    a relative window; it must never be reported as S_crit.

THE PROBE RANGE STOPS AT 12 x S_crit ON PURPOSE
    An earlier version ran to 100x and every surface came back dead: that is hazard 11's
    non-smooth, invalid-cell regime, not a property of the candidates.

RESULT (2026-08-31, base -> seed 6, 33 points)
    Where an orbit exists, S*(v) is SMOOTH. Over the contiguous stretch t = 0.500..1.000 it
    slides 29.93 -> 9.93 with per-step |dlog10 S*| median 0.026 and max 0.085, ratio 3.3.

    14 of 33 points could not be located AT ALL, and the orbit residual column says why: 362,
    670, 3000, and several degenerate solves. THE STRAIGHT LINE BETWEEN THE BASE POINT AND A
    HEALTHY FITTED OPTIMUM PASSES THROUGH PARAMETER SETS WITH NO LIMIT CYCLE. That breaks the
    fixed-window cost exactly as badly -- it scores 1.0 there -- so it is a property of the
    landscape, not of the relative window. It is also independent evidence for the multimodality
    of 5.10a: the optima are separated by a region with no clock in it.

    So the continuity objection to direction 2 is NOT supported by measurement.
"""
import os, sys, time
ROOT = r"C:\Users\galma\OneDrive\postdoc\Python\ModelingAndDataProcessingCodes (1)\ModelingAndDataProcessingCodes\Modeling\clock_zoo"
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, json, jax, jax.numpy as jnp
from models import get_model
from fit.cost import make_cost, RadialTarget, row_dispersion
from engine.ptc import grid_points, DEAD_AMP, NEG_TOL
from engine.orbit import make_guess_fn

z = dict(np.load('out/almeida/fit_radial/bmal1seeds/seeds_BMAL1_instant.npz', allow_pickle=True))
cfg = json.loads(str(z['cfg_json'])); B = np.asarray(z['B'])
m = get_model('almeida'); m.reference_variable = str(z['section'])
S0 = 1.0 / float(z['k_used'][0]); NPH = 12; ND = 14
# 0.003x to 12x S_crit: wide enough to follow a collapsing transition, and it STOPS BELOW the
# 18x S_crit regime where hazard 11's non-smoothness and dead cells begin
probe = np.geomspace(S0 * 3e-3, S0 * 12.0, ND)
C = make_cost(m, 'BMAL1', probe, RadialTarget(), n_phase=NPH, mode='instant', backend='diffrax',
              dt=float(cfg['dt']), w_osc=0.0, w_amp=0.0, pulse=float(cfg['pulse']),
              skip_p=cfg['skip_p'], readout_ref=str(z['readout']), basis=B)
f, solver = C['ptc_fn'], C['solver']
guess = make_guess_fn(m); yseed = jnp.asarray(m.get_initial_state(), jnp.float64)
names = list(C['names']); solve = jax.jit(solver.solve)
ph, dz = grid_points(NPH, probe)

def locate(v):
    P = m.jax_apply(np.asarray(C['theta'](v)), names)
    y0, T, res = solve(P, guess(P, yseed))
    zz, ymin = f(P, jnp.concatenate([y0, T[None]]), ph, dz)
    zz = np.asarray(zz).reshape(ND, NPH).T; ymin = np.asarray(ymin).reshape(ND, NPH).T
    fin = np.isfinite(zz.real) & np.isfinite(zz.imag) & np.isfinite(ymin)
    zs = np.where(fin, zz, 1.0 + 0j); amp = np.abs(zs)
    alive = fin & (amp > DEAD_AMP) & (ymin >= NEG_TOL)
    d = np.asarray(row_dispersion(jnp.asarray(zs / (amp + 1e-12)), jnp.asarray(alive)))
    a = np.where(d > 0.5)[0]
    if len(a) == 0:
        return -np.inf, float(res), d          # transition already below the probe floor
    if a[-1] >= ND - 1:
        return np.inf, float(res), d           # still above the probe ceiling
    i = a[-1]; ld = np.log(probe)
    fr = (d[i] - 0.5) / max(d[i] - d[i + 1], 1e-12)
    return float(np.exp(ld[i] + fr * (ld[i + 1] - ld[i]))), float(res), d

i6 = [k for k, x in enumerate(z['labels']) if str(x) == '6'][0]
v0, v1 = np.zeros(B.shape[1]), np.asarray(z['v'][i6])
ts = np.linspace(0.0, 1.0, 33)
print(f"probe {probe[0]:.4g} .. {probe[-1]:.4g}   (S_crit at base = {S0:.4g})", flush=True)
print(f"{'t':>6s}{'S*(v)':>12s}{'log10 S*':>10s}{'orbit res':>11s}{'alive':>7s}", flush=True)
out = []; t0 = time.time()
for t in ts:
    v = (1 - t) * v0 + t * v1
    S, res, d = locate(v)
    ls = np.log10(S) if np.isfinite(S) else (np.nan if S > 0 else np.nan)
    print(f"{t:6.3f}{S:12.4g}{ls:10.4f}{res:11.1e}{np.mean(np.isfinite(d)):7.2f}", flush=True)
    out.append((t, S))
print(f"\nelapsed {time.time()-t0:.0f}s")
a = np.array(out, float); ok = np.isfinite(a[:, 1]) & (a[:, 1] > 0)
ls = np.log10(a[ok, 1]); st = np.abs(np.diff(ls))
print(f"S* located at {ok.sum()}/{len(ts)} path points; S* {a[ok,1].min():.4g} -> {a[ok,1].max():.4g}")
print(f"per-step |dlog10 S*|: median {np.median(st):.4f}  max {st.max():.4f}  "
      f"max/median {st.max()/max(np.median(st),1e-12):.1f}")
print("A smooth S*(v) gives max/median of order a few; a jump gives tens or more.")
