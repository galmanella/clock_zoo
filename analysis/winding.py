"""
analysis/winding.py
===================
The pure-numpy PTC surface analysis: winding, phase singularities, the fixed-point / twist
curve, and hole imputation. Model-agnostic, engine-agnostic -- it takes a PTC grid and returns
features.

Lifted from input_screen/sensitivity_full.py and screen_core.py (commit e955873), which are
the parts of that codebase that survived every solver correction, because they never touch a
solver. Kept together here so `analysis/` has one place to ask "what is in this surface".

CONVENTIONS
    A PTC grid is `ptc[n_phase, n_dose]` of new_phase in [0, 1), with old_phase uniform on
    [0, 1) (PERIODIC, so the first and last rows are adjacent) and dose increasing along
    columns (NOT periodic).

WHAT THE FEATURES ARE, AND WHICH ONES TO TRUST
    winding W    per dose column: how many cycles new_phase advances over one cycle of
                 old_phase. |W| = 1 is type-1 (weak) resetting, |W| = 0 is type-0 (strong).
                 The 1 -> 0 transition is a topological event and passes through a phase
                 SINGULARITY, where the perturbed state is phaseless.

    S_crit, phi* dose and old-phase of that singularity. TOPOLOGICAL DEFECTS, so they are
                 intrinsically discontinuous and quantised by the grid. Report them, but do
                 not read a smooth sensitivity off them -- input_screen learned this the hard
                 way (its singularity panels are quantised to 1/32 in phi).

    twist        the stable fixed point's phase as a function of dose. SMOOTH and
                 gauge-invariant, and therefore the feature to read isochron change from.

WHY THE DIPOLE FILTER EXISTS
    A raw winding field puts a +1 and a -1 plaquette next to each other wherever the phase
    turns quickly -- a grid-resolution artifact, not a pair of singularities. Counting those
    badly inflates both the singularity count and the S_crit span. `find_singularities`
    cancels +/- pairs that are close in both phase and dose before reporting anything.
"""
import numpy as np


# --------------------------------------------------------------------------- #
#  Winding
# --------------------------------------------------------------------------- #
def winding_curve(new_phase):
    """Signed winding number of ONE PTC curve, or None if any point is NaN.

    None rather than a number: a winding number is a property of a CLOSED loop, so a curve
    with a hole in it does not have one, and computing an integer from the surviving points
    would silently invent a topology. Impute first (see `impute`) if the hole is a glitch."""
    new = np.asarray(new_phase, float)
    if not np.all(np.isfinite(new)):
        return None
    z = np.exp(2j * np.pi * new)
    d = np.angle(z[1:] * np.conj(z[:-1])).sum() + np.angle(z[0] * np.conj(z[-1]))
    return int(np.round(d / (2 * np.pi)))


def winding_vs_dose(ptc):
    """W for each dose column; np.nan where the column has a hole."""
    out = np.full(ptc.shape[1], np.nan)
    for k in range(ptc.shape[1]):
        w = winding_curve(ptc[:, k])
        if w is not None:
            out[k] = w
    return out


def winding_field(ptc):
    """Signed winding per plaquette, W[n_phase, n_dose-1] (old periodic, dose linear).

    The circulation of the phase around each grid cell. A cell containing a phase singularity
    has |W| = 1; everywhere else it is 0."""
    z = np.exp(2j * np.pi * ptc)
    zr = np.roll(z, -1, 0)
    d = lambda a, b: np.angle(a * np.conj(b))
    return (d(zr[:, :-1], z[:, :-1]) + d(zr[:, 1:], zr[:, :-1]) +
            d(z[:, 1:], zr[:, 1:]) + d(z[:, :-1], z[:, 1:])) / (2 * np.pi)


# --------------------------------------------------------------------------- #
#  Hole imputation
# --------------------------------------------------------------------------- #
def fill_circular_gaps(line, x, max_gap=4):
    """Fill interior NaN runs of length <= max_gap by circular (shortest-arc) linear
    interpolation between the bracketing finite values. Endpoint gaps and long runs stay NaN."""
    y = np.asarray(line, float).copy()
    n = len(y)
    k = 0
    while k < n:
        if np.isfinite(y[k]):
            k += 1
            continue
        a, b = k - 1, k
        while b < n and not np.isfinite(y[b]):
            b += 1
        if a >= 0 and b < n and (b - a - 1) <= max_gap:          # interior gap, short enough
            dwrap = ((y[b] - y[a] + 0.5) % 1.0) - 0.5            # shortest circular diff
            for j in range(a + 1, b):
                y[j] = (y[a] + (x[j] - x[a]) / (x[b] - x[a]) * dwrap) % 1.0
        k = b
    return y


def impute(ptc, max_gap=2):
    """Fill ISOLATED holes in a PTC grid, circularly along the periodic old-phase axis.

    Fixes the problem at the source: a lone failed point would otherwise break the winding
    (which needs a closed loop) and the fixed-point extraction, forcing every downstream
    routine to special-case NaN. Each dose column is tiled 3x so a hole at the wrap counts as
    interior. Runs longer than `max_gap`, all-NaN columns, and columns with < 2 finite points
    are left alone -- those are genuine gaps (a dead oscillator, an unstable integration), not
    glitches, and imputing them would manufacture data.

    input_screen used max_gap=2 for analysis and 6 for display; keep that split.
    """
    ptc = np.asarray(ptc, float).copy()
    M = ptc.shape[0]
    xext = np.arange(-M, 2 * M, dtype=float)
    for k in range(ptc.shape[1]):
        fin = np.isfinite(ptc[:, k])
        if fin.sum() < 2 or fin.all():
            continue
        ptc[:, k] = fill_circular_gaps(np.tile(ptc[:, k], 3), xext, max_gap=max_gap)[M:2 * M]
    return ptc


# --------------------------------------------------------------------------- #
#  Singularities
# --------------------------------------------------------------------------- #
def find_singularities(old, doses, ptc, d_phi=0.15, d_dose_ratio=2.5, max_gap=2):
    """Real singularities of a PTC grid, with the DIPOLE FILTER.

    A +1 and a -1 plaquette within `d_phi` (circular old-phase) AND within a factor
    `d_dose_ratio` in dose are a resolution artifact of a fast phase change, not two
    singularities, and are removed as a pair by greedy nearest matching. Returns the surviving
    [{phi, dose, sign}].
    """
    ptc = impute(ptc, max_gap=max_gap)
    W = winding_field(ptc)
    M = ptc.shape[0]
    oc = (np.asarray(old) + 0.5 / M) % 1.0                  # plaquette centres
    dc = 0.5 * (np.asarray(doses)[:-1] + np.asarray(doses)[1:])
    OC, DC = np.meshgrid(oc, dc, indexing='ij')
    finite = np.isfinite(W)
    pos = list(zip(OC[finite & (W > 0.5)], DC[finite & (W > 0.5)]))
    neg = list(zip(OC[finite & (W < -0.5)], DC[finite & (W < -0.5)]))

    lr = np.log(d_dose_ratio)
    used, surv = set(), []
    for (po, pdose) in pos:                                 # cancel each + against a nearby -
        best, bd = None, np.inf
        for j, (no, ndose) in enumerate(neg):
            if j in used:
                continue
            dphi = abs(po - no) % 1.0
            dphi = min(dphi, 1 - dphi)
            dr = abs(np.log(pdose / ndose)) if (pdose > 0 and ndose > 0) else np.inf
            if dphi < d_phi and dr < lr and dphi + 0.3 * dr < bd:
                best, bd = j, dphi + 0.3 * dr
        if best is None:
            surv.append({'phi': float(po), 'dose': float(pdose), 'sign': 1})
        else:
            used.add(best)
    surv += [{'phi': float(no), 'dose': float(nd), 'sign': -1}
             for j, (no, nd) in enumerate(neg) if j not in used]
    return surv


def track_base_singularity(sings_by_setting, base_idx):
    """Follow the singularity connected to the BASE across a parameter sweep.

    At each step take the survivor nearest the previous pick (circular old-phase + log-dose),
    walking outward from `base_idx` in both directions. Without this the reported S_crit jumps
    between distinct singularities as a parameter moves, which reads as an enormous
    sensitivity when nothing has actually moved far.

    Returns (S[n], phi[n]) following that one singularity.
    """
    nf = len(sings_by_setting)
    S = np.full(nf, np.nan)
    P = np.full(nf, np.nan)

    def nearest(sings, ref):
        if not sings:
            return None
        if ref is None:
            return min(sings, key=lambda s: s['dose'])     # base: the lowest-dose one
        def dist(s):
            dphi = abs(s['phi'] - ref[0]) % 1.0
            return min(dphi, 1 - dphi) + 0.3 * abs(np.log(s['dose'] / ref[1]))
        return min(sings, key=dist)

    sb = nearest(sings_by_setting[base_idx], None)
    ref0 = (sb['phi'], sb['dose']) if sb else None
    if sb:
        S[base_idx], P[base_idx] = sb['dose'], sb['phi']
    for direction in (range(base_idx + 1, nf), range(base_idx - 1, -1, -1)):
        ref = ref0
        for i in direction:
            s = nearest(sings_by_setting[i], ref)
            if s:
                S[i], P[i] = s['dose'], s['phi']
                ref = (s['phi'], s['dose'])
    return S, P


def detect_grid(old, doses, ptc):
    """(S_crit, phi_sing, n_sing) for one PTC grid: the LOWEST-dose surviving singularity is
    the type-1 -> type-0 transition. For a smooth S_crit across a parameter sweep use
    `track_base_singularity` instead -- this one can jump."""
    sings = find_singularities(old, doses, ptc)
    if not sings:
        return np.nan, np.nan, 0.0
    s = min(sings, key=lambda s: s['dose'])
    return float(s['dose']), float(s['phi']), float(len(sings))


# --------------------------------------------------------------------------- #
#  Fixed point / twist
# --------------------------------------------------------------------------- #
def stable_fixed_points(old_phase, new_phase, fine=720, min_amplitude=0.02):
    """Attracting phase(s) of the once-per-cycle map new = f(old).

    A fixed point is where the wrapped shift g(phi) = wrap(new - old) crosses zero. Larger
    phase means more advanced, so the STABLE crossing has advance on the left and delay on the
    right: g going + -> -. Usually one; several in extreme PTCs.

    `min_amplitude`: if the PTC is within this of the identity everywhere, it is treated as
    NON-resetting and no fixed point is returned. Without the guard a near-identity PTC yields
    spurious noise-driven crossings that jump around, which then dominate a twist curve.

    Robust to wrap-arounds: interpolates on the unwrapped lift of new_phase using the winding
    number before locating crossings.
    """
    ph = np.asarray(old_phase, float)
    nw = np.asarray(new_phase, float)
    good = np.isfinite(nw)
    ph, nw = ph[good], nw[good]
    if len(ph) < 4:
        return np.array([])
    order = np.argsort(ph)
    ph, nw = ph[order], nw[order]

    if np.max(np.abs(((nw - ph + 0.5) % 1.0) - 0.5)) < min_amplitude:
        return np.array([])

    d = winding_curve(nw)
    if d is None:
        return np.array([])
    lift = np.unwrap(2 * np.pi * nw) / (2 * np.pi)
    ph_ext = np.concatenate([ph, [ph[0] + 1.0]])
    lift_ext = np.concatenate([lift, [lift[0] + d]])

    phi = np.linspace(0.0, 1.0, fine, endpoint=False)
    new_fine = np.interp(phi, ph_ext, lift_ext)
    g = ((new_fine - phi + 0.5) % 1.0) - 0.5

    fps = []
    for i in range(fine):
        j = (i + 1) % fine
        g0, g1 = g[i], g[j]
        if g0 > 0 and g1 <= 0 and (g0 - g1) < 0.5:
            frac = g0 / (g0 - g1) if (g0 - g1) != 0 else 0.0
            fps.append((phi[i] + frac / fine) % 1.0)
    return np.array(sorted(fps))


def _primary_fixed_point(fps, prev):
    """One representative stable FP for a dose-response line: nearest (circularly) to the
    previous dose's pick, else the first."""
    if len(fps) == 0:
        return np.nan
    if len(fps) == 1 or not np.isfinite(prev):
        return float(fps[0])
    return float(fps[int(np.argmin(np.abs(((fps - prev + 0.5) % 1.0) - 0.5)))])


def twist_curve(old, doses, ptc, max_gap=2):
    """Stable-fixed-point phase vs dose -- the TWIST curve.

    This is the feature to read isochron change from: smooth, and gauge-invariant (unlike
    S_crit, which is a dose and therefore rescales with the units). The surface is imputed
    first so a lone hole cannot break a column's diagonal crossing; a whole-column dropout
    still leaves a gap, which is circularly interpolated along dose if short.
    """
    ptc = impute(ptc, max_gap=max_gap)
    doses = np.asarray(doses, float)
    line, last = [], np.nan
    for k in range(ptc.shape[1]):
        col = ptc[:, k]
        if np.isfinite(col).sum() < 4:
            line.append(np.nan)
            continue
        fp = _primary_fixed_point(stable_fixed_points(old, col), last)
        line.append(fp)
        if np.isfinite(fp):
            last = fp                      # keep the reference across gaps
    return fill_circular_gaps(np.array(line), doses)


# --------------------------------------------------------------------------- #
#  Scalar summaries
# --------------------------------------------------------------------------- #
def circ_span(phi):
    """Largest pairwise circular distance among finite values -- the spread of a phase set."""
    a = np.asarray(phi, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return 0.0
    d = np.abs(a[:, None] - a[None, :]) % 1.0
    return float(np.max(np.minimum(d, 1.0 - d)))


def circ_rms(a, b):
    """RMS circular difference between two phase arrays, ignoring NaN."""
    d = np.abs(((np.asarray(a, float) - np.asarray(b, float) + 0.5) % 1.0) - 0.5)
    return float(np.sqrt(np.nanmean(d ** 2))) if np.any(np.isfinite(d)) else np.nan


def total_twist(tw):
    """Total isochron shear across the dose axis: the circular span of the twist curve."""
    return circ_span(tw)


def _selftest():
    """Synthetic surfaces with known answers."""
    M, J = 48, 30
    old = np.arange(M) / M
    doses = np.geomspace(0.05, 5.0, J)
    ok = True

    # 1. identity map -> type-1 everywhere, no singularity
    ptc = np.tile(old[:, None], (1, J))
    w = winding_vs_dose(ptc)
    s = find_singularities(old, doses, ptc)
    print(f"  identity map      : W = {set(w.astype(int))} (expect {{1}}), "
          f"{len(s)} singularities (expect 0)")
    ok &= set(w.astype(int)) == {1} and len(s) == 0

    # 2. a Winfree-style surface that genuinely changes type at a known dose.
    #    new = arg( exp(2 pi i old) + r(dose) ) traces winding 1 for r < 1 and 0 for r > 1,
    #    with the singularity exactly at r = 1 -- an analytic answer to check against.
    r = doses / 1.0
    z = np.exp(2j * np.pi * old)[:, None] + r[None, :]
    ptc2 = (np.angle(z) / (2 * np.pi)) % 1.0
    w2 = winding_vs_dose(ptc2)
    s2 = find_singularities(old, doses, ptc2)
    lo = doses[w2 == 1].max() if np.any(w2 == 1) else np.nan
    hi = doses[w2 == 0].min() if np.any(w2 == 0) else np.nan
    print(f"  type-1 -> type-0  : transition bracketed by [{lo:.3f}, {hi:.3f}] (expect 1.0), "
          f"{len(s2)} singularity (expect 1)")
    ok &= (lo < 1.0 < hi) and len(s2) == 1
    if s2:
        print(f"                      detected at dose {s2[0]['dose']:.3f}, "
              f"phi {s2[0]['phi']:.3f}")

    # 3. imputation restores a punched hole without touching a long run
    ptc3 = ptc2.copy(); ptc3[7, 5] = np.nan; ptc3[10:20, 6] = np.nan
    imp = impute(ptc3)
    err = abs(imp[7, 5] - ptc2[7, 5])
    print(f"  imputation        : lone hole recovered to {err:.2e}; "
          f"long run left as NaN: {not np.isfinite(imp[10, 6])}")
    ok &= err < 0.02 and not np.isfinite(imp[10, 6])

    # 4. twist curve is finite and smooth on the type-0 side
    tw = twist_curve(old, doses, ptc2)
    print(f"  twist curve       : {np.sum(np.isfinite(tw))}/{J} finite, "
          f"total twist {total_twist(tw):.3f} cyc")
    ok &= np.sum(np.isfinite(tw)) > J // 2

    print(f"  [{'PASS' if ok else 'FAIL'}] winding kernels")
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if _selftest() else 1)
