"""
fit/campaign.py
===============
Turn ONE config file with list-valued fields into a matrix of runs, and execute or enumerate it.

    python -m fit.campaign --config campaigns/genes.json --list      # what would run
    python -m fit.campaign --config campaigns/genes.json --dry-run   # + validate every one
    python -m fit.campaign --config campaigns/genes.json --index 3   # run one (SLURM array)
    python -m fit.campaign --config campaigns/genes.json --all       # run all, in order

WHY AN INDEX RATHER THAN A LOOP
    A campaign is a set of INDEPENDENT experiments, and that maps onto a SLURM array exactly:
    `--array=0-N` with `--index $SLURM_ARRAY_TASK_ID` gives one task per configuration, each
    with its own wall clock, its own failure, and its own output directory. `--all` exists for
    running the same matrix on one machine, but it is the fallback, not the design.

    The index is defined by the SORTED cartesian product, so it is STABLE: index 3 means the
    same configuration today and next month, on the cluster and on a laptop. Do not reorder the
    lists in a config file that a submitted array is still using.

WHAT `--list` IS FOR, AND WHY IT VALIDATES
    A 24-task array that dies at task 0 because of a typo has cost a queue slot and an hour.
    `--dry-run` constructs and validates every configuration in the matrix without integrating
    anything, so the failure happens at submission time, in front of you.
"""
import argparse
import dataclasses

from fit.config import RunConfig


def load_matrix(path):
    """(configs, base, axes) for a campaign file, deduplicated and validated.

    `axes` are the fields actually given as lists -- what this campaign varies -- and they name
    the runs. Entries whose `identity()` collides are DROPPED with a message: sweeping
    mode x pulse produces `instant` twice, since an instant kick ignores pulse duration, and a
    silently duplicated array task is a wasted queue slot and a misleading directory.
    """
    base = RunConfig.load(path)
    # `seeds` counts as an axis ONLY when given as a list of lists (seed SETS). Excluding it
    # unconditionally meant a seed-set sweep produced no axes, so every entry got the same tag
    # and the array tasks silently overwrote one another's output.
    axes = sorted(f.name for f in dataclasses.fields(RunConfig)
                  if f.name != 'seeds'
                  and isinstance(getattr(base, f.name), (list, tuple)))
    sv = getattr(base, 'seeds')
    if isinstance(sv, (list, tuple)) and sv and all(
            isinstance(x, (list, tuple)) for x in sv):
        axes = sorted(axes + ['seeds'])
    seen, configs, dropped = {}, [], 0
    for c in base.expand():
        c.validate()
        key = c.identity()
        if key in seen:
            dropped += 1
            continue
        seen[key] = True
        configs.append(c)
    if dropped:
        print(f"  [campaign] dropped {dropped} duplicate configuration(s): the swept fields "
              f"did not change the experiment (e.g. pulse duration under mode=instant)")
    return configs, base, axes


def _tag_for(base, cfg, axes):
    """Output tag: the campaign name, then only what this entry varies.

    Named rather than numbered, because `out/almeida/fit_radial/genes__target-PER/`
    says what it is and `.../job_3/` does not. Flat, because paths.py requires a
    tag to be a plain name -- the structure lives in the name, not in nesting.
    """
    root = base.tag or 'campaign'
    return f"{root}__{cfg.label_over(axes)}" if axes else root


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('WHY AN INDEX')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='JSON with list-valued fields to sweep')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--list', action='store_true', help='print the matrix and exit')
    g.add_argument('--dry-run', action='store_true',
                   help='build and VALIDATE every configuration without running anything')
    g.add_argument('--index', type=int, help='run one entry (use $SLURM_ARRAY_TASK_ID)')
    g.add_argument('--all', action='store_true', help='run every entry, sequentially')
    ap.add_argument('--workers', type=int, default=None,
                    help='override workers for this invocation (e.g. from $SLURM_CPUS_PER_TASK)')
    a = ap.parse_args(argv)

    configs, base, axes = load_matrix(a.config)
    if a.workers is not None:
        for c in configs:
            c.workers = a.workers

    if a.list or a.dry_run:
        print(f"{len(configs)} run(s) from {a.config}")
        total_seeds = 0
        for i, c in enumerate(configs):
            print(f"  [{i:3d}] {_tag_for(base, c, axes):<52s} seeds={c.seed_list} "
                  f"workers={c.workers} maxfev={c.maxfev}")
            total_seeds += len(c.seed_list)
        print(f"\n  {total_seeds} searches in total "
              f"({len(configs)} configurations x their seeds)")
        if a.dry_run:
            print("\n  all configurations validated; nothing was run")
            print(f"\n  SLURM: sbatch --array=0-{len(configs) - 1} slurm/campaign_cpu.sh "
                  f"{a.config}")
        return 0

    from fit.radial import run_seeds
    todo = list(range(len(configs))) if a.all else [a.index]
    if a.index is not None and not (0 <= a.index < len(configs)):
        raise SystemExit(f"--index {a.index} is out of range: this campaign has "
                         f"{len(configs)} entries (0-{len(configs) - 1})")
    for i in todo:
        c = configs[i]
        c.tag = _tag_for(base, c, axes)
        print(f"\n{'#' * 78}\n# campaign entry {i} of {len(configs) - 1}: {c.tag}\n{'#' * 78}",
              flush=True)
        run_seeds(c)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
