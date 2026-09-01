"""
analysis/phaseref.py
====================
A COMMON PHASE ORIGIN, so old phase means the same thing in every model.

    $PY -m analysis.phaseref                      # measure and save the offsets
    $PY -m analysis.phaseref --anchor Per --all   # ... and show every species' peak

THE PROBLEM THIS FIXES
    `engine.orbit` anchors phase 0 at a MAXIMUM of each model's `reference_variable`, and
    those are three different biological events:

        almeida    phase 0 = peak of PER      (protein)
        korencic   phase 0 = peak of Reverbx  (Rev-erb mRNA)
        goldbeter  phase 0 = peak of MP       (Per mRNA)

    So a singularity at phi* = 0.5 in two models is not the same time of day, and the two
    numbers must not be plotted on one axis. MEASURED at the base parameters: Almeida's PER
    and Goldbeter's MP already sit at 0.000, but Korencic's Perx peaks at 0.445 -- its whole
    phase axis is out of register by nearly half a cycle (about 11.5 h).

    The fix is a RELABELLING, not a re-run. Both PTC axes carry the same origin (the readout
    is calibrated so dose 0 is the identity), so subtracting one constant from old phase AND
    new phase moves the origin and leaves every difference-based feature -- total,
    accumulated and signed twist -- exactly invariant.

WHY NOT JUST CHANGE `reference_variable`
    Because that attribute is the ORBIT SOLVER'S POINCARE SECTION, `rhs(y0)[ref] = 0`, and it
    is chosen for UNIMODALITY, not for biology. REPO_MAP hazard 14 is the record of what
    happens when the two roles are conflated: Almeida's section on BMAL1 stopped being
    unimodal at a fitted optimum and the SAME parameter set returned periods of 0.159 h and
    43.1 h from two different guesses. The section is a numerical choice and stays where it
    is; the origin is a presentation choice and moves here, in the analysis layer, where it
    is reversible and costs nothing.

THE RESIDUAL CAVEAT, WHICH IS REAL
    The anchor is each model's IN-SCOPE species for the anchor gene (analysis.genemap), so
    Korencic and Goldbeter are anchored on Per mRNA while Almeida -- which has no mRNA at all
    -- is anchored on PER protein. Those are not the same event. The mRNA-to-protein lag is
    measured here and reported alongside the offsets, so the size of the mismatch is on the
    record rather than assumed away: Korencic 0.088 cyc, Goldbeter ~0.03 cyc.
"""
import argparse
import os

import numpy as np

import analysis  # noqa: F401  -- repo root on sys.path
import paths
from analysis import genemap as GM
from analysis.features import ZOO

#: The gene whose peak defines phase 0 for every model.
ANCHOR_GENE = 'Per'
#: Samples per cycle used to locate the peak. 512 -> a resolution of 0.002 cyc (~3 min).
M_CYCLE = 512


def wrap_half(x):
    """A phase difference wrapped to [-0.5, 0.5) -- the signed shortest arc."""
    return float(((np.asarray(x, float) + 0.5) % 1.0) - 0.5)


def _peak_phase(profile):
    """Phase of the maximum, by parabolic interpolation through the three samples around it.

    Sub-sample, because the raw argmax quantises the offset to 1/M and that quantisation would
    show up as a systematic shift between models sampled at the same M."""
    c = np.asarray(profile, float)
    m = len(c)
    k = int(np.argmax(c))
    y0, y1, y2 = c[(k - 1) % m], c[k], c[(k + 1) % m]
    den = y0 - 2 * y1 + y2
    d = 0.0 if abs(den) < 1e-300 else 0.5 * (y0 - y2) / den
    d = float(np.clip(d, -0.5, 0.5))
    return ((k + d) / m) % 1.0


def measure(model_name, anchor_gene=ANCHOR_GENE, m_cycle=M_CYCLE):
    """Everything about one model's phase convention, at the BASE parameters."""
    import jax
    from models import get_model
    from engine.orbit import OrbitSolver

    model = get_model(model_name)
    s = OrbitSolver(model, n_steps=1024)
    P = model.jax_params()
    y0, T, resid = jax.jit(s.solve)(P, s.guess(P))
    mu, _ev = s.floquet(P, y0, T)
    cyc = np.asarray(s.cycle(P, y0, T, m=m_cycle))
    names = list(model.state_names)
    peaks = {n: _peak_phase(cyc[:, i]) for i, n in enumerate(names)}

    scope = GM.SCOPE_LEVELS.get(model_name, GM.LEVELS)
    anch = [st for (m, st, lv) in GM.pairs(anchor_gene) if m == model_name and lv in scope]
    if not anch:
        raise SystemExit(f"{model_name} has no in-scope {anchor_gene!r} species to anchor on "
                         f"(scope levels {scope}); pass a different --anchor")
    anchor = anch[0]

    # the mRNA -> protein lag for the anchor gene, where the model resolves both. This is the
    # size of the residual mismatch against a protein-anchored model like Almeida.
    lag = np.nan
    mrna = [st for (m, st, lv) in GM.pairs(anchor_gene, 'mrna') if m == model_name]
    prot = [st for (m, st, lv) in GM.pairs(anchor_gene, 'protein') if m == model_name]
    if mrna and prot:
        lag = float((peaks[prot[0]] - peaks[mrna[0]]) % 1.0)
        if lag > 0.5:
            lag -= 1.0

    return dict(model=model_name, anchor=anchor, anchor_gene=anchor_gene,
                offset=float(peaks[anchor]), section=s.ref_name,
                readout=(getattr(model, 'readout_variable', None) or s.ref_name),
                period=float(T), residual=float(np.max(np.abs(np.asarray(resid)))),
                mu=float(mu), mrna_protein_lag=lag, peaks=peaks, names=names,
                cycle=cyc)


def build(models=GM.MODELS, anchor_gene=ANCHOR_GENE, m_cycle=M_CYCLE, verbose=True):
    rows = [measure(m, anchor_gene, m_cycle) for m in models]
    if verbose:
        report(rows)
    return rows


def save(rows, tag=None):
    tag = paths.run_tag(tag)
    blob = dict(models=np.array([r['model'] for r in rows]),
                anchor=np.array([r['anchor'] for r in rows]),
                anchor_gene=str(rows[0]['anchor_gene']),
                offset=np.array([r['offset'] for r in rows]),
                section=np.array([r['section'] for r in rows]),
                readout=np.array([r['readout'] for r in rows]),
                period=np.array([r['period'] for r in rows]),
                residual=np.array([r['residual'] for r in rows]),
                mu=np.array([r['mu'] for r in rows]),
                mrna_protein_lag=np.array([r['mrna_protein_lag'] for r in rows]))
    for r in rows:                                   # raw cycles: the figures may want them
        blob[f"cycle__{r['model']}"] = np.asarray(r['cycle'])
        blob[f"states__{r['model']}"] = np.array(r['names'])
        blob[f"peaks__{r['model']}"] = np.array([r['peaks'][n] for n in r['names']])
    out = paths.out_path(ZOO, 'phaseref', 'phaseref.npz', tag)
    paths.savez(out, **blob)
    print(f"[phaseref] -> {out}", flush=True)
    return out


def load(tag=None):
    """{model: offset}, plus the blob. Returns ({}, None) if never measured -- callers then
    fall back to the raw model-local origin and SAY SO on the figure."""
    tag = tag or paths.latest_run(ZOO, 'phaseref')
    fp = None if tag is None else paths.out_path(ZOO, 'phaseref', 'phaseref.npz', tag)
    if fp is None or not os.path.exists(fp):
        return {}, None
    z = dict(np.load(fp, allow_pickle=True))
    return {str(m): float(o) for m, o in zip(z['models'], z['offset'])}, z


def report(rows):
    print(f"\n{'=' * 92}\nCOMMON PHASE ORIGIN -- anchored on the peak of "
          f"{rows[0]['anchor_gene']}\n{'=' * 92}")
    print(f"  {'model':10s} {'section (phase 0 now)':22s} {'readout':10s} {'anchor':9s} "
          f"{'offset':>8s} {'period':>8s} {'mu':>6s} {'mRNA->prot lag':>15s}")
    for r in rows:
        lag = ('--' if not np.isfinite(r['mrna_protein_lag'])
               else f"{r['mrna_protein_lag']:+.3f} cyc")
        print(f"  {r['model']:10s} peak of {r['section']:<14s} {r['readout']:10s} "
              f"{r['anchor']:9s} {r['offset']:8.3f} {r['period']:8.3f} {r['mu']:6.3f} "
              f"{lag:>15s}")
    print("\n  offset = where the anchor peaks in the model's CURRENT coordinate; the figures")
    print("  subtract it from BOTH phase axes, which leaves every twist measure invariant.")
    # Report the SIGNED shift, wrapped to [-0.5, 0.5). An anchor sitting a hair below sample 0
    # comes back from `% 1.0` as 0.999998, which is the same shift as -0.000002 but reads as
    # "displaced by a whole cycle" -- a scary number for a quantity that is exactly zero.
    for r in rows:
        w = wrap_half(r['offset'])
        if abs(w) > 0.01:
            print(f"  {r['model']}: shifted by {w:+.3f} cyc ({w * r['period']:+.1f} h) -- its "
                  f"phi* values in earlier figures were out of register by that much.")
        else:
            print(f"  {r['model']}: already on the anchor ({w:+.1e} cyc); nothing moves.")
    lags = [r['mrna_protein_lag'] for r in rows if np.isfinite(r['mrna_protein_lag'])]
    if lags:
        print(f"  CAVEAT: models with no mRNA (Almeida) are anchored on protein instead. The "
              f"lag that costs is {min(lags):+.3f}..{max(lags):+.3f} cyc where measurable.")


def show_peaks(rows):
    for r in rows:
        print(f"\n  {r['model']} (T={r['period']:.3f} h) -- peak phase of every state, "
              f"AFTER the shift to the {r['anchor_gene']} anchor:")
        for n in r['names']:
            p0 = r['peaks'][n]
            print(f"      {n:9s} {p0:6.3f}  ->  {((p0 - r['offset']) % 1.0):6.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description='a common phase origin across models')
    ap.add_argument('--anchor', default=ANCHOR_GENE, help='gene whose peak is phase 0')
    ap.add_argument('--models', default=','.join(GM.MODELS))
    ap.add_argument('--m-cycle', type=int, default=M_CYCLE)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--all', action='store_true', help='also list every state peak')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    rows = build([m.strip() for m in a.models.split(',') if m.strip()], a.anchor, a.m_cycle)
    if a.all:
        show_peaks(rows)
    if not a.dry_run:
        save(rows, a.tag)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
