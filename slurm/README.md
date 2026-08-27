# `slurm/` — cluster job templates

Every driver in `analysis/` is already headless and shardable, so these scripts do only three
things: pin threads, set the environment, and map `SLURM_ARRAY_TASK_ID` onto `--shard`.

```bash
sbatch slurm/sens_cpu.sh almeida BMAL1 pulse     # array over parameters, then merge
sbatch slurm/scrit_cpu.sh almeida pulse          # array over targets
sbatch slurm/ptc_gpu.sh   goldbeter MP pulse     # same work, one GPU task
```

## The rules these encode

- **`CLOCKZOO_STRICT_PROVENANCE=1`.** An untracked entry script or a dirty tree becomes a hard
  error instead of a warning. Nobody reads stderr on a cluster, and a result whose recorded
  commit cannot reproduce it is worse than no result.
- **One thread per worker.** Without pinning, every one of `NJOBS` processes spawns its own
  BLAS pool and the node thrashes.
- **All array tasks share `SLURM_ARRAY_JOB_ID`**, so they write into ONE output directory and
  a `--merge` pass collects them. `paths.py` implements that convention.
- **Never pipe a job's output through `tail`.** It buffers, so a healthy job looks hung.

## Sizing

`analysis.scrit` refines `dt` until `S_crit` is trustworthy, so its cost varies a lot by
target (Almeida: 376 s for 8 targets, but the hard ones needed dt=0.00125). `analysis.ptc_sens`
is the expensive one: parameters x 9 factors x a phase-dose grid. Shard over parameters and
give each task a whole core.
