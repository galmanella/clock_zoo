"""
fit/rescan.py
=============
Re-render a campaign's fitted PTC surfaces on a BIGGER, FINER dose grid -- reaching down to 0.

    python -m fit.rescan --model almeida --tag bmal1seeds
    python -m fit.rescan --model almeida --tag genes --kind genes --n-phase 48 --n-dose 48

WHY THE FIT WINDOW IS THE WRONG WINDOW TO LOOK AT THE ANSWER IN
    `fit/doses.fit_dose_grid` spans `0.5x` to `max_factor x S_crit`, and that is the right
    window to FIT in: it keeps the gradient finite (hazard 11) and it is where the target has
    anything to say. It is the wrong window to READ a result in, for two reasons that the
    Aug-30 campaign ran straight into (PROJECT_SUMMARY 5.10c-d):

      * eleven of twelve usable fits ended with NO singularity anywhere in the window. A
        surface with the transition pushed below the floor and a surface with no transition at
        all are the same picture there, and they are completely different clocks.
      * five ended with a PTC that does not vary with old phase. On a window that starts at
        0.38 x S_crit that could mean "type-0 everywhere in this window" or "phaseless" -- and
        the first is ordinary while the second is a degeneracy.

    Extending DOWN settles both. As dose -> 0 the PTC must approach the identity, so a healthy
    clock's surface has to become type-1 somewhere below its own S_crit no matter where that
    sits. A surface that stays flat all the way to dose 0 is not a clock with a low S_crit; it
    is a surface with no phase information, and the identity row at dose 0 is the control that
    proves the machinery was working.

WHY IT ALSO GOES FINER
    PROJECT_SUMMARY 5.9b: the twist on BMAL1 is aliased below ~48 dose samples, and an
    under-sampled spiral is smooth, plausible and wrong rather than obviously noisy. The
    campaign ran 14. No twist number off that grid is a magnitude.

THIS RE-RUNS THE PRODUCTION PATH, NOT A COPY OF IT
    The surfaces come from `fit.cost.make_cost(...)['surface']` at the run's own stored
    `v_fit`, with the run's own section, readout, mode, backend and dt. That is the same code
    that produced the fitted values, which is the point -- a diagnostic that reimplements the
    thing it is checking cannot check it (hazard 15).
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import paths


#: how far the dose-0 row may sit from the identity before the readout is called uncalibrated.
#: NOT machine epsilon: the phase comes from a Fourier fundamental on a finite window, so a
#: healthy run lands at 1e-4 - 1e-3 cyc. MEASURED separation on bmal1seeds -- the calibrated
#: runs sit at 1.0e-04 to 9.4e-04 and the broken ones at 4.1e-02 to 5.0e-01, two orders clear,
#: so the threshold is not a judgement call.
ID_TOL = 1e-2


def dose_grid(d_max, n_dose, decades=4.0, include_zero=True):
    """Log-spaced up to `d_max` over `decades`, with an explicit dose-0 row in front.

    Zero cannot be on a log axis, and it is not there to be plotted -- it is the CONTROL.
    `engine.ptc` calibrates the phase origin so that dose 0 returns new phase == old phase
    exactly; if that row is not the identity, the readout or the section is wrong and nothing
    else on the surface means anything. It costs one row out of fifty to keep that check
    attached to every rescan instead of trusting it.
    """
    lo = d_max / (10.0 ** decades)
    d = np.geomspace(lo, d_max, int(n_dose) - (1 if include_zero else 0))
    return np.concatenate([[0.0], d]) if include_zero else d


def rescan(model_name, tag, analysis='fit_radial', kind='seeds', n_phase=48, n_dose=48,
           decades=4.0, verbose=True):
    """Every run of an aggregated campaign, re-rendered. Returns the blob to save."""
    from models import get_model
    from fit.cost import make_cost, RadialTarget
    from fit.target import radial_z as _radial_z
    from analysis import winding as W
    from analysis import quality as Q

    d = paths.out_dir(model_name, analysis, tag, create=False)
    pat = 'seeds_*.npz' if kind == 'seeds' else 'genes_*.npz'
    fs = sorted(glob.glob(os.path.join(d, pat)))
    if not fs:
        raise SystemExit(f"no aggregated {pat} under {d}. Run `python -m fit.aggregate "
                         f"--model {model_name} --tag {tag}"
                         + ('' if kind == 'seeds' else ' --kind genes') + "` first.")
    z = dict(np.load(fs[-1], allow_pickle=True))
    cfg = json.loads(str(z['cfg_json'])) if 'cfg_json' in z else {}
    labels = [str(x) for x in z['labels']]
    names = [str(x) for x in z['names']]
    model = get_model(model_name)
    if 'section' in z:
        model.reference_variable = str(z['section'])
    readout = str(z['readout']) if 'readout' in z else None
    mode = str(z['mode'])
    n = len(labels)

    targets = ([str(z['target'])] * n if np.asarray(z['target']).ndim == 0
               else [str(t) for t in z['target']])
    basis = np.asarray(z['B']) if 'B' in z else None
    if basis is None:
        raise SystemExit(
            "the aggregated npz has no gauge-quotient basis `B`. Re-run `python -m "
            "fit.aggregate` with the current code: without the run's own basis, its `v` "
            "cannot be turned back into the parameters it was fitted at.")
    print(f"[rescan] {n} run(s) of {tag} at {n_phase} phase x {n_dose} dose, "
          f"{decades:g} decades below each window's top, plus a dose-0 identity row",
          flush=True)

    surf, alive, amps, doses_all, tw, feats = [], [], [], [], [], []
    t0 = time.time()
    for i, lab in enumerate(labels):
        dmax = float(np.max(np.asarray(z['doses'])[i]))
        doses = dose_grid(dmax, n_dose, decades)
        # PIN THE BASIS THE RUN USED. `v` only means anything in its own gauge-quotient
        # basis, and `quotient_basis` does not return a reproducible one (see fit/cost's note
        # on `basis`). Without this the surfaces come back uniformly DEAD, because the
        # reconstructed parameters are decades away from the ones that were fitted -- which is
        # exactly how this was found.
        C = make_cost(model, targets[i], doses, RadialTarget(), n_phase=n_phase, mode=mode,
                      backend=str(cfg.get('backend', 'diffrax')), dt=float(cfg.get('dt', 0.02)),
                      w_osc=0.0, w_amp=0.0, pulse=float(cfg.get('pulse', 8.0)),
                      skip_p=cfg.get('skip_p'), readout_ref=readout, basis=basis)
        # and PROVE it round-trips before trusting the surface
        th = np.asarray(C['theta'](np.asarray(z['v'][i])))
        drift = float(np.max(np.abs(np.log10(th / np.asarray(z['theta_fit'][i])))))
        if drift > 1e-8:
            raise SystemExit(
                f"basis mismatch for {lab}: reconstructing theta from the stored v differs "
                f"from the stored theta_fit by {drift:.3g} decades. Refusing to rescan a "
                f"parameter set the run never evaluated.")
        zz, aa, am = C['surface'](np.asarray(z['v'][i]))
        ptc = np.where(np.asarray(aa), (np.angle(np.asarray(zz)) / (2 * np.pi)) % 1.0, np.nan)
        old = np.asarray(C['old'])
        t = W.twist_curve(old, doses, ptc)
        S, phi, ns = W.detect_grid(old, doses, ptc)
        q = Q.score(old, doses, ptc)
        surf.append(ptc); alive.append(np.asarray(aa)); amps.append(np.asarray(am))
        doses_all.append(doses); tw.append(t)
        feats.append(dict(S_crit=S, phi_sing=phi, n_sing=ns, scramble=q['scramble'],
                          quality=bool(q['passed']),
                          accum=W.accumulated_twist(t), signed=W.signed_twist(t),
                          span_total=W.total_twist(t)))
        if verbose:
            el = time.time() - t0
            # the dose-0 row IS the control: report how far it is from the identity
            id_err = float(np.nanmax(np.abs(((ptc[:, 0] - old) + 0.5) % 1.0 - 0.5)))
            print(f"  [{i + 1:2d}/{n}] {lab:>6s}  S_crit={S if np.isfinite(S) else float('nan'):9.4g}"
                  f"  n_sing={int(ns):2d}  accum={feats[-1]['accum']:6.3f}"
                  f"  dose-0 identity err={id_err:.2e}"
                  f"  ({el:.0f}s, ~{el / (i + 1) * (n - i - 1):.0f}s left)", flush=True)

    old = np.asarray(C['old'])
    blob = dict(labels=np.array(labels), key=np.asarray(str(z['key'])),
                target=np.asarray(z['target']), model=np.asarray(model_name),
                mode=np.asarray(mode), campaign_tag=np.asarray(str(tag)),
                section=np.asarray(str(z.get('section', ''))), readout=np.asarray(str(readout)),
                n_phase=np.asarray(n_phase), n_dose=np.asarray(n_dose),
                decades=np.asarray(decades),
                old=old, doses=np.array(doses_all),
                # --- RAW (hazard 9) ------------------------------------------------------ #
                ptc=np.array(surf), alive=np.array(alive), amp=np.array(amps),
                twist=np.array(tw),
                # the FIT window, so a figure can shade what the optimizer actually saw
                fit_doses=np.asarray(z['doses']),
                k_used=np.asarray(z['k_used']), psi_used=np.asarray(z['psi_used']),
                v=np.asarray(z['v']), theta_fit=np.asarray(z['theta_fit']),
                names=np.array(names),
                )
    for k in ('S_crit', 'phi_sing', 'n_sing', 'scramble', 'accum', 'signed', 'span_total'):
        blob[k] = np.array([f[k] for f in feats], float)
    blob['quality'] = np.array([f['quality'] for f in feats])
    # the identity control, per run
    blob['identity_err'] = np.array([
        float(np.nanmax(np.abs(((np.asarray(s)[:, 0] - old) + 0.5) % 1.0 - 0.5))) for s in surf])
    # the radial target on the SAME extended grid, for comparison
    blob['ptc_target'] = np.array([
        (np.angle(np.asarray(_radial_z(old, doses_all[i], float(blob['k_used'][i]),
                                       float(blob['psi_used'][i])))) / (2 * np.pi)) % 1.0
        if np.isfinite(blob['k_used'][i]) else np.full((n_phase, n_dose), np.nan)
        for i in range(n)])
    return blob


def report(b):
    n = len(b['labels'])
    print(f"\n{'=' * 96}\nRESCAN -- {b['campaign_tag']}   {int(b['n_phase'])} phase x "
          f"{int(b['n_dose'])} dose, down to dose 0\n{'=' * 96}")
    print(f"  {str(b['key']):>6s} {'dose range':>22s} {'S_crit':>10s} {'n_sing':>7s} "
          f"{'accum':>8s} {'signed':>8s} {'span':>7s} {'scram':>7s} {'id_err':>9s}  quality")
    for i in np.argsort(b['labels']):
        d = np.asarray(b['doses'])[i]
        pos = d[d > 0]
        print(f"  {str(b['labels'][i]):>6s} {pos.min():10.3g} - {pos.max():9.3g} "
              f"{b['S_crit'][i]:10.4g} {int(b['n_sing'][i]):7d} {b['accum'][i]:8.3f} "
              f"{b['signed'][i]:+8.3f} {b['span_total'][i]:7.3f} {b['scramble'][i]:7.4f} "
              f"{b['identity_err'][i]:9.1e}  {'PASS' if b['quality'][i] else 'FAIL'}")
    ok = np.isfinite(b['identity_err']) & (b['identity_err'] <= ID_TOL)
    bad = int(np.sum(~ok))
    if ok.any():
        print(f"\n  dose-0 identity: {int(ok.sum())}/{n} within {ID_TOL:g} cyc, clustered at "
              f"{np.nanmin(b['identity_err'][ok]):.0e}-{np.nanmax(b['identity_err'][ok]):.0e} "
              f"-- the readout's own resolution, not an error")
    if bad:
        w = np.where(~ok)[0]
        print("  WARNING: " + ", ".join(f"{b['labels'][i]} ({b['identity_err'][i]:.2g})"
                                        for i in w)
              + " do NOT. At dose 0 the perturbation is ZERO, so new phase MUST equal old "
                "phase; where it does not, the phase readout is not calibrated at that "
                "parameter set and nothing else on that surface is a measurement.")
    ns0 = int(np.sum(b['n_sing'] == 0))
    print(f"  {n - ns0}/{n} surfaces have a phase singularity somewhere in the EXTENDED "
          f"window; {ns0} still have none even down to dose 0.")
    return b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY THE FIT')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--kind', default='seeds', choices=('seeds', 'genes'))
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--n-phase', type=int, default=48)
    ap.add_argument('--n-dose', type=int, default=48,
                    help='INCLUDING the dose-0 identity row')
    ap.add_argument('--decades', type=float, default=4.0,
                    help='how far below the top of the fit window to reach (default 4)')
    a = ap.parse_args(argv)
    b = rescan(a.model, a.tag, a.analysis, a.kind, a.n_phase, a.n_dose, a.decades)
    report(b)
    out = paths.out_path(a.model, a.analysis, f"rescan_{b['mode']}.npz", a.tag)
    paths.savez(out, **b)
    print(f"\n[rescan] -> {os.path.relpath(out, paths.HERE)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
