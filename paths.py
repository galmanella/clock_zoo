"""
paths.py
========
Output-path protocol. One place that decides where results go, so every driver writes
consistently and results never clobber.

Layout:
    out/<model>/<analysis>/[<tag>/]<name>

  model    : any registered model name ('almeida', 'korencic', 'goldbeter', 'goodwin', ...)
  analysis : 'scrit' | 'characterize' | 'lc_sens' | 'ptc_sens' | 'coupling' | 'validate'
  tag      : optional run label (SLURM job id / timestamp) for things that re-run; omit for
             single-latest artifacts (a figure you overwrite).

ARRAY-JOB SEMANTICS (why `tag` and the filename are decided separately)
    All tasks of one SLURM array share `SLURM_ARRAY_JOB_ID`, so `run_tag()` gives them ONE
    output directory; the per-task `SLURM_ARRAY_TASK_ID` only distinguishes the FILE inside it
    (`shard_filename`). A `--merge` pass then globs `shard*.npz` in that directory.

Helpers:
  out_dir(model, analysis, tag=None)      -> ensured directory path
  out_path(model, analysis, name, tag)    -> full file path (dir ensured)
  run_tag(explicit=None)                  -> SLURM (array) job id / $RUN_TAG / timestamp
  shard_filename(prefix)                  -> 'shard<taskid>.npz' in an array job, else '<prefix>.npz'
  latest_run(model, analysis)             -> newest tag dir (for readers), or None
  write_meta(model, analysis, tag, **cfg) -> meta.json (git commit, host, time, cfg)
  savez(path, **arrays)                   -> npz + sidecar .meta.json (provenance)
  load_shards(model, analysis, tag)       -> [dict] of every shard npz in a run dir
  provenance() / check_provenance()       -> is this output reproducible? (see below)

Provenance: copied verbatim in spirit from input_screen/paths.py (commit e955873). The
failure it prevents actually happened there -- see the guard's own comment.
"""
import os
import sys
import glob
import json
import socket
import pickle
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

#: `analysis` values the drivers use. Not enforced (a new analysis should not need a code
#: change here) but listed so the layout stays discoverable.
ANALYSES = ('validate', 'scrit', 'characterize', 'lc_sens', 'ptc_sens', 'coupling')


def _safe(part, what):
    """Reject path separators / traversal in a name that becomes a directory component."""
    s = str(part)
    if not s or s != os.path.basename(s) or s in ('.', '..'):
        raise ValueError(f"{what} must be a plain name, got {part!r}")
    return s


def out_dir(model, analysis, tag=None, create=True):
    """out/<model>/<analysis>[/<tag>].

    `model` is any registered model name -- deliberately NOT validated against a hardcoded
    tuple (input_screen's `MODELS = ('original','reduced')` had to be edited for every new
    model, which is exactly the coupling this repo exists to remove)."""
    parts = [OUT, _safe(model, 'model'), _safe(analysis, 'analysis')]
    if tag:
        parts.append(_safe(tag, 'tag'))
    p = os.path.join(*parts)
    if create:
        os.makedirs(p, exist_ok=True)
    return p


def out_path(model, analysis, name, tag=None):
    return os.path.join(out_dir(model, analysis, tag), name)


def run_tag(explicit=None):
    """SHARED tag for a run: explicit > SLURM (array) job id > $RUN_TAG > timestamp.

    For array jobs all tasks see the same SLURM_ARRAY_JOB_ID -> one tag dir; the per-task id
    only distinguishes the FILE (see `shard_filename`)."""
    if explicit:
        return str(explicit)
    jid = os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID')
    if jid:
        return str(jid)
    return os.environ.get('RUN_TAG') or datetime.now().strftime('%Y%m%d_%H%M%S')


def shard_id():
    """(index, count) of this array task, or (0, 1) outside an array job. Honours explicit
    $SHARD / $NSHARDS so a shard can be reproduced locally without SLURM."""
    i = os.environ.get('SHARD', os.environ.get('SLURM_ARRAY_TASK_ID'))
    n = os.environ.get('NSHARDS', os.environ.get('SLURM_ARRAY_TASK_COUNT'))
    return (int(i) if i is not None else 0), (int(n) if n is not None else 1)


def shard_filename(prefix='result', idx=None):
    """'<prefix>_shard<i>.npz' inside an array job (or when SHARD is set), else '<prefix>.npz'."""
    if idx is None:
        idx = os.environ.get('SHARD', os.environ.get('SLURM_ARRAY_TASK_ID'))
    return f"{prefix}.npz" if idx is None else f"{prefix}_shard{int(idx)}.npz"


def latest_run(model, analysis):
    """Newest tag subdirectory under out/<model>/<analysis>, or None."""
    base = os.path.join(OUT, str(model), str(analysis))
    if not os.path.isdir(base):
        return None
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subs:
        return None
    return max(subs, key=lambda d: os.path.getmtime(os.path.join(base, d)))


def load_shards(model, analysis, tag=None, prefix='result'):
    """Every shard npz in a run dir, as a list of dicts (loaded eagerly). Used by --merge."""
    import numpy as np
    if tag is None:
        tag = latest_run(model, analysis)
    if tag is None:
        return []
    d = out_dir(model, analysis, tag, create=False)
    files = sorted(glob.glob(os.path.join(d, f'{prefix}*.npz')))
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def _git(*args, default=None):
    try:
        return subprocess.check_output(['git', *args], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def git_commit():
    return _git('rev-parse', '--short', 'HEAD', default='nogit')


# --------------------------------------------------------------------------- #
#  Provenance guard
#  ----------------
#  The failure this exists to prevent (it happened in input_screen, Aug 2026): five driver
#  scripts lived only in a scratch directory, wrote results into out/, were cited in the
#  project summary -- and were then lost when the scratch dir was cleaned. meta.json had
#  dutifully recorded a git SHA for every run, but that commit never contained the code, so
#  the provenance silently pointed at nothing. An UNTRACKED entry script means the recorded
#  commit cannot reproduce the output.
#
#  Set CLOCKZOO_STRICT_PROVENANCE=1 to turn the warning into a hard error (do this on the
#  cluster, where nobody is watching stderr).
# --------------------------------------------------------------------------- #
STRICT_PROVENANCE = os.environ.get('CLOCKZOO_STRICT_PROVENANCE', '') not in ('', '0', 'false')


def provenance():
    """Describe the CODE producing this output: {git, script, script_tracked, dirty, warn}."""
    script = os.path.abspath(sys.argv[0]) if (sys.argv and sys.argv[0]) else ''
    rel = os.path.relpath(script, HERE).replace('\\', '/') if script else '<interactive>'
    tracked = None
    if script and os.path.exists(script):
        tracked = _git('ls-files', '--error-unmatch', '--', script) is not None
    dirty = bool(_git('status', '--porcelain', '--untracked-files=no', default=''))
    warn = []
    if tracked is False:
        warn.append(f"entry script {rel!r} is NOT tracked by git -- this result is NOT reproducible")
    if dirty:
        warn.append("working tree has uncommitted changes to tracked files")
    return dict(git=git_commit(), script=rel, script_tracked=tracked, dirty=dirty, warn=warn)


def check_provenance(prov=None):
    """Warn (or raise, under CLOCKZOO_STRICT_PROVENANCE) about unreproducible output.
    Returns the provenance dict, minus the transient 'warn' list."""
    prov = prov or provenance()
    for w in prov['warn']:
        msg = f"[paths] PROVENANCE: {w}"
        if STRICT_PROVENANCE:
            raise RuntimeError(msg + "  (CLOCKZOO_STRICT_PROVENANCE=1)")
        print(msg, file=sys.stderr, flush=True)
    return {k: v for k, v in prov.items() if k != 'warn'}


def write_meta(model, analysis, tag=None, **cfg):
    meta = dict(model=model, analysis=analysis, tag=(str(tag) if tag else None),
                timestamp=datetime.now().isoformat(timespec='seconds'),
                host=socket.gethostname(), **check_provenance(), **cfg)
    with open(out_path(model, analysis, 'meta.json', tag), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def save_pickle(obj, model, analysis, name, tag=None, meta=None):
    check_provenance()
    with open(out_path(model, analysis, name, tag), 'wb') as f:
        pickle.dump(obj, f)
    if meta is not None:
        write_meta(model, analysis, tag, **meta)


def savez(path, **arrays):
    """np.savez_compressed + a sidecar '<path>.meta.json' recording provenance.

    Use this instead of a bare np.savez_compressed for anything whose result gets cited.
    `path` may be a plain path or a (model, analysis, name[, tag]) tuple."""
    import numpy as np
    if isinstance(path, (tuple, list)):
        path = out_path(*path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    prov = check_provenance()
    np.savez_compressed(path, **arrays)
    with open(path + '.meta.json', 'w') as f:
        json.dump(dict(timestamp=datetime.now().isoformat(timespec='seconds'),
                       host=socket.gethostname(), arrays=sorted(arrays), **prov),
                  f, indent=2, default=str)
    return path
