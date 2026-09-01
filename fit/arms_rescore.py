"""
fit/arms_rescore.py
===================
The EXPENSIVE half of an arm comparison, run once and saved, so the figures stay pure-read.

    $PY -m fit.arms_rescore --model almeida --arms arm0_ctl arm1_ramp arm2_brack \
        arm3_inform arm4_floor --control arm0_ctl --tag arms_batch1 --check

WHAT IT SAVES
    * the common-yardstick score and the six contract checks for every endpoint
      (`fit.compare_arms.compare`, which is where that logic lives -- this does not restate it);
    * THE OBJECTIVE THAT WAS ACTUALLY SEARCHED, evaluated at each arm's start point.

WHY THE SECOND ONE EXISTS
    A campaign's npz records the objective the run DECLARED: `trace_f[0]` is the parent's own
    evaluation of the start under the arm's own cost. Every other entry in the trace comes from
    the worker pool. If the two are not the same function then the tag says one thing and the
    search did another, and nothing downstream can tell -- the batch-1 arms are exactly that
    case (PROJECT_SUMMARY 5.15).

    So the searched objective is rebuilt HERE, EXPLICITLY, as `make_cost` with no arm options at
    all -- which is what the pre-fix `fit.parallel.build_cost` produced in absolute mode, since
    it forwarded none of them and `make_cost` defaults `amp_ramp=(0.05, 0.20)`. It is written
    longhand rather than obtained by calling `build_cost`, so that FIXING `build_cost` cannot
    silently change this audit. `--check` asserts it against the arms' own stored numbers.
"""
import argparse
import glob
import json
import os

import numpy as np

import paths


def worker_objective_pre_fix(model_name, cfg, z):
    """`build_cost`'s absolute branch AS IT WAS: no amp_ramp, no row_weight, no w_brack and no
    basis forwarded, so `make_cost`'s own defaults apply -- and its default ramp is (0.05, 0.20).

    FROZEN ON PURPOSE. This function must keep reproducing the PRE-FIX behaviour after
    `build_cost` is corrected; that is the whole point of not calling `build_cost` here.
    """
    from models import get_model
    from fit.cost import make_cost, RadialTarget
    m = get_model(model_name)
    if cfg.get('section'):
        m.reference_variable = cfg['section']
    if cfg.get('readout'):
        m.readout_variable = cfg['readout']
    tgt = RadialTarget(k=float(z['k_used']), psi=float(z['psi_used']))
    return make_cost(m, cfg['target'], np.asarray(z['doses'], float), tgt,
                     n_phase=int(cfg['n_phase']), mode=cfg['mode'], backend=cfg['backend'],
                     dt=float(cfg['dt']), w_osc=float(cfg['w_osc']), w_amp=float(cfg['w_amp']),
                     pulse=float(cfg['pulse']), skip_p=cfg.get('skip_p'),
                     readout_ref=cfg.get('readout'))


def start_vectors(model_name, runs):
    """Each run's start point, in the basis THIS build derives.

    That is the basis the run's own `v0` / fixture projection used and the one the pre-fix
    worker rebuilt for itself, so a start scored here is the start the pool was handed. It is
    NOT in general the stored `B` (hazard 18): passing the npz's basis to a locally built cost
    scores a different parameter set, and measured on this batch that alone moves the fixture
    start from 0.1220 to 3.1207.

    THE START NAME COMES FROM THE RUN'S OWN `cfg_json`, NOT FROM ITS DIRECTORY NAME.
    `compare_arms._start_of` splits the directory on '_', so the fixture start `fixture_seed2`
    reads back as `fixture` and the fixture cannot be loaded. The directory name is a label; the
    config is the record.
    """
    from fit.config import load_start_fixture
    out = {}
    for r in runs:
        n_free = int(np.asarray(r['z']['B']).shape[1])
        start = str(json.loads(str(r['z']['cfg_json'])).get('start', 'base'))
        if not start.startswith('fixture_'):
            out[r['dirname']] = np.zeros(n_free)
        else:
            out[r['dirname']] = load_start_fixture(model_name, start.split('_', 1)[1], n_free)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHAT IT SAVES')[0].strip())
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--control', required=True)
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--tag', default='arms_batch1')
    ap.add_argument('--check', action='store_true',
                    help="assert the rebuilt searched objective against the stored trace[0]")
    ap.add_argument('--tol', type=float, default=1e-9,
                    help="relative tolerance for 'declared == searched'. NOT zero: the stored "
                         "value was computed on the CLUSTER and the rebuild here is on another "
                         "machine, so identical objectives still differ in the last bits "
                         "(measured ~1e-16 relative). A real objective mismatch is 6x, not "
                         "1e-16, so nothing is hidden by this.")
    a = ap.parse_args(argv)

    from fit.compare_arms import compare, report, _arm_dirs, _start_of, _seed_of

    # THE CHEAP HALF FIRST. `compare` is ~10 minutes of surface evaluations; resolving the
    # start fixtures is a file read. Doing it second once cost exactly that ten minutes to a
    # missing fixture, so the order is deliberate.
    runs = []
    for arm in a.arms:
        for dirname, f in _arm_dirs(a.model, arm, a.analysis):
            z = dict(np.load(f, allow_pickle=True))
            runs.append(dict(arm=arm, dirname=dirname, start=_start_of(dirname),
                             seed=_seed_of(z), z=z))
    v0s = start_vectors(a.model, runs)

    # ONE REBUILD PER DISTINCT DOSE GRID. The pre-fix worker cost is a function of the dose grid
    # and the pinned target only; nothing else that varies across these arms reaches it, which
    # is precisely the defect being audited.
    cache = {}
    for r in runs:
        cfg = json.loads(str(r['z']['cfg_json']))
        key = (round(float(r['z']['doses'][0]), 12), round(float(r['z']['doses'][-1]), 12),
               round(float(r['z']['k_used']), 12))
        if key not in cache:
            cache[key] = worker_objective_pre_fix(a.model, cfg, r['z'])
        r['searched_f0'] = float(cache[key]['total'](v0s[r['dirname']]))
        r['declared_f0'] = float(np.asarray(r['z']['trace_f'])[0])
        r['f0_reldiff'] = (abs(r['searched_f0'] - r['declared_f0'])
                           / max(abs(r['declared_f0']), 1e-30))
        r['same_objective'] = bool(r['f0_reldiff'] <= a.tol)

    rows, cfg0 = compare(a.model, a.arms, a.control, a.analysis, with_contract=True)
    if not rows:
        raise SystemExit("no runs found for any arm -- has the array finished, and has out/ "
                         "been rsynced back? (out/ is gitignored.)")

    if a.check:
        # arm1_ramp DECLARES exactly the ramp the pre-fix worker used by default, so wherever
        # the ramp is the only difference its stored start value must equal the rebuilt
        # searched one to the last bit. Any other arm agreeing is evidence of the same defect.
        for arm in sorted({r['arm'] for r in runs}):
            sel = [r for r in runs if r['arm'] == arm]
            same = sum(1 for r in sel if r['same_objective'])
            worst = max(r['f0_reldiff'] for r in sel)
            print(f"  check  {arm:<12s} declared == searched at the start in "
                  f"{same}/{len(sel)} runs   (worst relative difference {worst:.2e})")

    by_dir = {r['dirname']: r for r in runs}
    for row in rows:
        row['searched_f0'] = by_dir[row['dirname']]['searched_f0']
        row['declared_f0'] = by_dir[row['dirname']]['declared_f0']
        row['f0_reldiff'] = by_dir[row['dirname']]['f0_reldiff']
        row['same_objective'] = by_dir[row['dirname']]['same_objective']

    def _plain(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return v

    d = paths.out_dir(a.model, a.analysis, a.tag)
    p = os.path.join(d, 'yardstick.json')
    with open(p, 'w') as fh:
        json.dump(dict(control=a.control, arms=list(a.arms), analysis=a.analysis,
                       rows=[{k: _plain(v) for k, v in row.items()} for row in rows],
                       provenance=paths.provenance()), fh, indent=2, default=str)
    print(f"\n[rescore] -> {os.path.relpath(p, paths.HERE)}  ({len(rows)} endpoints)")

    report(rows, a.control)

    print(f"  {'arm':<12s}{'start':<16s}{'declared f(v0)':>16s}{'SEARCHED f(v0)':>16s}"
          f"{'rel diff':>11s}   verdict")
    seen = set()
    for r in sorted(runs, key=lambda x: (x['start'], x['arm'])):
        k = (r['arm'], r['start'])
        if k in seen:
            continue
        seen.add(k)
        agree = 'same objective' if r['same_objective'] else 'DIFFERENT OBJECTIVE'
        print(f"  {r['arm']:<12s}{r['start']:<16s}{r['declared_f0']:16.6f}"
              f"{r['searched_f0']:16.6f}{r['f0_reldiff']:11.1e}   {agree}")
    print("\n  'declared' is the arm's own cost at its start point (the npz's trace[0], the one\n"
          "  evaluation the parent makes); 'SEARCHED' is what the worker pool minimised for the\n"
          "  other 8000. Where they differ, the tag names an objective the run did not use.\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
