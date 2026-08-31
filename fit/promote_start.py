"""
fit/promote_start.py
====================
Promote a finished fit's parameters into `fixtures/starts/` so a later run can START there.

    python -m fit.promote_start --model almeida --tag bmal1seeds --label 2 --name seed2 \
        --why "attracting orbit, but fails contract C5 and C6 -- the pathological arm start"
    git add fixtures/starts/almeida && git commit

WHY A FIXTURE AND NOT A PATH INTO out/
    REPO_MAP hazard 16: `out/` is output and is never an input. A run that reads its starting
    point out of `out/` is a run whose starting point depends on whichever machine last
    produced one, and `out/` is gitignored so a fresh clone has none of it. Promotion is
    deliberate, reviewable, and shows up as a diff -- "the experiment's starting point changed"
    should be a commit.

WHAT IT STORES, AND WHY MORE THAN `v`
    `v`, `B`, `z_base` AND `theta`. The gauge-quotient basis is NOT reproducible -- it is the
    SVD of a projector whose nonzero singular values are all exactly 1, so any orthonormal
    basis of that subspace is a valid answer (hazard 18). MEASURED: the same `v` under a
    re-derived basis reconstructed parameters up to 3.0 DECADES away. So the basis travels with
    the coordinate, and `fit.config.load_start_fixture` verifies the round trip before any run
    uses it.
"""
import argparse
import glob
import json
import os

import numpy as np

import paths


def promote(model_name, tag, label, name, why='', analysis='fit_radial', dry_run=False):
    d = paths.out_dir(model_name, analysis, tag, create=False)
    fs = sorted(glob.glob(os.path.join(d, 'seeds_*.npz')))
    if not fs:
        raise SystemExit(f"no aggregated seeds_*.npz under {d}; run `python -m fit.aggregate "
                         f"--model {model_name} --tag {tag}` first.")
    z = dict(np.load(fs[-1], allow_pickle=True))
    labels = [str(x) for x in z['labels']]
    if str(label) not in labels:
        raise SystemExit(f"no run labelled {label!r} in {os.path.basename(fs[-1])}; "
                         f"have {labels}")
    i = labels.index(str(label))
    for k in ('B', 'z_base'):
        if k not in z:
            raise SystemExit(f"the aggregate has no {k!r} -- re-run `fit.aggregate` on current "
                             f"code, or the start cannot be reconstructed (hazard 18).")
    v = np.asarray(z['v'][i], float)
    B = np.asarray(z['B'], float)
    zb = np.asarray(z['z_base'], float)
    theta = np.asarray(z['theta_fit'][i], float)
    drift = float(np.max(np.abs(np.log10(np.exp(zb + B @ v) / theta))))
    if drift > 1e-8:
        raise SystemExit(f"refusing: exp(z_base + B v) disagrees with the recorded theta_fit "
                         f"by {drift:.3g} decades in the SOURCE aggregate.")
    blob = dict(v=v, B=B, z_base=zb, theta=theta, names=np.asarray(z['names']),
                model=np.asarray(model_name), source_tag=np.asarray(str(tag)),
                source_label=np.asarray(str(label)), why=np.asarray(str(why)),
                target=np.asarray(z['target']), mode=np.asarray(z['mode']),
                section=np.asarray(z.get('section', '')),
                readout=np.asarray(z.get('readout', '')))
    out = os.path.join(paths.HERE, 'fixtures', 'starts', model_name, name + '.npz')
    print(f"  {name}: from {tag}/{label}   |v| = {np.linalg.norm(v):.3f}   "
          f"round-trip drift {drift:.1e} decades")
    print(f"  why: {why or '(not given -- record one)'}")
    if dry_run:
        print(f"  [dry-run] would write {os.path.relpath(out, paths.HERE)}")
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    paths.savez(out, **blob)
    print(f"  -> {os.path.relpath(out, paths.HERE)}   (git add it, and commit)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY A FIXTURE')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='almeida')
    ap.add_argument('--tag', required=True, help='aggregated campaign tag to take it from')
    ap.add_argument('--label', required=True, help='which run in that campaign (seed or gene)')
    ap.add_argument('--name', required=True, help='fixture name, used as start="fixture:<name>"')
    ap.add_argument('--why', default='', help='why THIS start -- recorded in the fixture')
    ap.add_argument('--analysis', default='fit_radial')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    promote(a.model, a.tag, a.label, a.name, a.why, a.analysis, a.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
