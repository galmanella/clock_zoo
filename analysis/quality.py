"""
analysis/quality.py
===================
A PASS/FAIL gate on a PTC surface, run BEFORE any feature (S_crit, phi*, twist, rho) is quoted
from it.

    python -m analysis.quality --model almeida --mode instant [--tag T] [--targets A,B]

WHY THIS EXISTS
    The CRY surface in the pulse-mode batch-1 run was reported with an S_crit of 138.1, a twist
    of 0.285 and a decoupling ratio of 1923.5 -- the largest numbers in the table, and all of
    them meaningless. The surface was phase-scrambled: its winding sequence over dose ran
    +1,+1,...,0,0,+1,+2,0,...,-1,...,+1,0, its phase changed by up to 0.499 cyc between
    ADJACENT phase samples (0.5 is the maximum possible, i.e. no correlation between
    neighbours), and it carried 13 raw winding plaquettes that the dipole filter reduced to one
    arbitrary survivor. Nothing downstream noticed, because every consumer took the
    `detect_grid` answer at face value.

    The failure was caught by eye, by the user, from a figure. This module makes it mechanical.

WHAT IT CHECKS, AND WHY EACH ONE
    winding set     A PTC winding number is +1 (type 1) or 0 (type 0). Values of -1 or +2 are
                    not resetting behaviour, they are evidence the phase readout is noise.
    scramble        The fraction of cells whose phase differs from its neighbour in old-phase
                    by more than `D_THR`. A REAL singularity produces exactly this, but only in
                    the few cells near it, so a clean surface scores a small fraction and a
                    scrambled one scores a large one. The most diagnostic single number here.
    dipole pairs    How much work the dipole filter had to do: (raw plaquettes - survivors)/2.
                    A clean transition needs a pair or two; needing six means the filter is
                    picking an arbitrary survivor out of noise. WARN, not FAIL, since a
                    legitimately fast phase change also generates pairs.
    NaN fraction    Cells with no usable phase at all.

    The two HARD checks are the winding set and the scramble fraction, because both are
    unambiguous. The rest are warnings, so this gate does not silently discard a surface that
    is merely under-resolved.

THRESHOLDS
    Deliberately loose, and always reported next to the raw numbers so a borderline call stays
    visible instead of hiding inside a boolean. The measured separation on the pulse-mode
    batch-1 run was wide (BMAL1 scramble 0.006 against CRY 0.30), so nothing here is finely
    tuned to the two cases that motivated it.
"""
import argparse
import glob
import os

import numpy as np

import analysis  # noqa: F401
import paths
from analysis import winding as W

#: Phase change between adjacent old-phase samples above which a cell is called discontinuous.
#: 0.5 is the theoretical maximum (antipodal); this is half of it.
D_THR = 0.25
#: Fraction of discontinuous cells above which the surface is called scrambled. A single
#: singularity on a 32 x 20 grid touches a handful of cells, i.e. well under 1%.
SCRAMBLE_MAX = 0.05
#: NaN fraction above which the surface is too holey to read.
NAN_MAX = 0.20
#: Dipole pairs above which the filter is doing suspicious amounts of work (WARN only).
DIPOLE_WARN = 3


def circ_absdiff(a, b):
    """|a - b| on the circle, in cycles, in [0, 0.5]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 1.0
    return np.minimum(d, 1.0 - d)


def score(old, doses, ptc):
    """Metrics + verdict for one PTC surface. `ptc` is [n_phase, n_dose]."""
    ptc = np.asarray(ptc, float)
    n_phase, n_dose = ptc.shape

    nan_frac = float(np.mean(~np.isfinite(ptc)))

    # Judge the IMPUTED surface, the same one detect_grid uses, so the gate sees what the
    # consumers actually see rather than a stricter private version.
    imp = W.impute(ptc)
    Wv = W.winding_vs_dose(imp)
    wfin = Wv[np.isfinite(Wv)].astype(int)
    wset = sorted(set(wfin.tolist()))
    wset_ok = set(wset) <= {0, 1}

    d = circ_absdiff(imp, np.roll(imp, -1, axis=0))
    fin = np.isfinite(d)
    scramble = float(np.mean(d[fin] > D_THR)) if fin.any() else 1.0
    max_d = float(np.max(d[fin])) if fin.any() else np.nan
    # per-dose worst: shows whether badness is localized at the transition or spread everywhere
    col_max = np.full(n_dose, np.nan)
    for k in range(n_dose):
        c = d[:, k][np.isfinite(d[:, k])]
        if c.size:
            col_max[k] = c.max()
    bad_cols = int(np.sum(col_max > D_THR))

    fieldW = W.winding_field(imp)
    raw = int(np.sum(np.isfinite(fieldW) & (np.abs(fieldW) > 0.5)))
    sings = W.find_singularities(old, doses, ptc)
    surviving = len(sings)
    pairs = max(0, raw - surviving) // 2

    hard = []
    if not wset_ok:
        hard.append(f"winding set {wset} is not a subset of [0, 1]")
    if scramble > SCRAMBLE_MAX:
        hard.append(f"scramble fraction {scramble:.3f} > {SCRAMBLE_MAX}")
    if nan_frac > NAN_MAX:
        hard.append(f"NaN fraction {nan_frac:.3f} > {NAN_MAX}")
    warn = []
    if pairs > DIPOLE_WARN:
        warn.append(f"{pairs} dipole pairs removed to reach {surviving} singularity(ies)")
    if surviving > 1:
        warn.append(f"{surviving} surviving singularities (a clean transition has 1)")

    return dict(n_phase=n_phase, n_dose=n_dose, nan_frac=nan_frac,
                winding_set=wset, winding_ok=bool(wset_ok), winding=Wv,
                scramble=scramble, max_dphase=max_d, bad_cols=bad_cols, col_max=col_max,
                raw_plaquettes=raw, surviving=surviving, dipole_pairs=pairs,
                passed=not hard, failures=hard, warnings=warn)


def report(name, s, verbose=True):
    tag = 'PASS' if s['passed'] else 'FAIL'
    print(f"  [{tag}] {name:9s} scramble={s['scramble']:.4f} max|dphi|={s['max_dphase']:.3f} "
          f"({s['bad_cols']}/{s['n_dose']} cols) W={s['winding_set']} "
          f"nan={s['nan_frac']:.3f} plaq={s['raw_plaquettes']}->{s['surviving']}")
    for f in s['failures']:
        print(f"           FAIL: {f}")
    for w in s['warnings']:
        print(f"           warn: {w}")
    if verbose and not s['passed']:
        seq = ' '.join('x' if not np.isfinite(v) else f'{int(v):+d}' for v in s['winding'])
        print(f"           W(dose): {seq}")
    return s['passed']


def load_surfaces(model_name, mode, tag=None):
    """{target: (old, doses, ptc)} from a characterize run (one file per target)."""
    tag = tag or paths.latest_run(model_name, 'characterize')
    if tag is None:
        raise SystemExit(f"run `python -m analysis.characterize --model {model_name} "
                         f"--mode {mode}` first")
    d = paths.out_dir(model_name, 'characterize', tag, create=False)
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, f'char_{mode}*.npz'))):
        z = np.load(fp, allow_pickle=True)
        for t in [str(x) for x in z['targets']]:
            if f'ptc__{t}' in z:
                out[t] = (np.asarray(z[f'old__{t}']), np.asarray(z[f'doses__{t}']),
                          np.asarray(z[f'ptc__{t}']))
    return out, tag


def run(model_name, mode='pulse', targets=None, tag=None):
    surf, tag = load_surfaces(model_name, mode, tag)
    if targets:
        surf = {k: v for k, v in surf.items() if k in targets}
    if not surf:
        raise SystemExit(f"no surfaces for {model_name}/{mode} in tag {tag}")
    print(f"\n{'=' * 78}\nPTC QUALITY GATE -- {model_name} ({mode}), tag {tag}\n{'=' * 78}")
    results, allok = {}, True
    for t, (old, doses, ptc) in surf.items():
        s = score(old, doses, ptc)
        results[t] = s
        allok &= report(t, s)
    good = [t for t, s in results.items() if s['passed']]
    print(f"\n  {len(good)}/{len(results)} surfaces usable: {good or 'NONE'}")
    print("  Features (S_crit, twist, rho) must not be quoted from a FAILed surface.")

    blob = {'model': model_name, 'mode': mode,
            'targets': np.array(list(results)),
            'passed': np.array([results[t]['passed'] for t in results])}
    for t, s in results.items():
        for k in ('scramble', 'max_dphase', 'nan_frac', 'raw_plaquettes', 'surviving',
                  'dipole_pairs', 'bad_cols'):
            blob[f'{k}__{t}'] = np.asarray(s[k])
        blob[f'winding__{t}'] = np.asarray(s['winding'])
        blob[f'col_max__{t}'] = np.asarray(s['col_max'])
    out = paths.out_path(model_name, 'quality', f'quality_{mode}.npz', tag)
    paths.savez(out, **blob)
    print(f"\n[quality] -> {out}")
    return results, allok


def main(argv=None):
    ap = argparse.ArgumentParser(description='PASS/FAIL gate on PTC surfaces')
    ap.add_argument('--model', default=os.environ.get('MODEL', 'almeida'))
    ap.add_argument('--mode', default=os.environ.get('MODE', 'pulse'),
                    choices=('pulse', 'instant'))
    ap.add_argument('--targets', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--strict', action='store_true',
                    help='exit non-zero if any surface fails')
    a = ap.parse_args(argv)
    _r, ok = run(a.model, a.mode, a.targets.split(',') if a.targets else None, a.tag)
    return 0 if (ok or not a.strict) else 1


if __name__ == '__main__':
    raise SystemExit(main())
