"""
fit/config.py
=============
One declaration of every knob a radialization run has, usable from the command line, from a
JSON file, or from a campaign matrix.

WHY THIS EXISTS
    Settings had accumulated in three places that disagreed. `--max-factor` defaulted to 6.0 in
    `fit/radial.py` while `fit_dose_grid` defaulted to 8.0, so the CLI silently overrode the
    considered choice documented in `fit/doses.py`. `pulse` was an 8 h `make_ptc` default that
    no caller could reach, so a pulse experiment could not vary its own pulse. `popsize`,
    `w_osc`, `w_amp` and `bound` existed as `run()` arguments with no way to set them. And a
    run recorded none of this, so "what produced this .npz" was answerable only by reading the
    source at the commit it was made from.

    A RunConfig is therefore the SINGLE source of truth: every default lives here once, every
    field is settable, and the whole object is written into the run's .npz so a result carries
    its own definition.

THE `sweep` FIELD IS WHAT MAKES CAMPAIGNS POSSIBLE
    Any field may be given a LIST instead of a scalar. `expand()` then returns the cartesian
    product as concrete configs -- so "three genes x four seeds x two population sizes" is one
    file and 24 runs, and `--index` picks one out for a SLURM array task. See fit/campaign.py.
"""
import copy
import dataclasses
import itertools
import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

import numpy as np

#: fields that are structural rather than scientific -- excluded from the auto-generated run
#: label, because putting the output tag inside the label would be circular
_LABEL_SKIP = {'tag', 'out_root', 'workers', 'verbose', 'log_every', 'notes'}


@dataclass
class RunConfig:
    """Everything a radialization run needs. Every field is settable; nothing is hard-wired."""

    # ---- what is being fitted -------------------------------------------------------- #
    model: str = 'almeida'
    #: the gene the DOSE is applied to. Not the phase observable -- see `section`/`readout`.
    target: str = 'BMAL1'
    mode: str = 'instant'                       # 'instant' | 'pulse'
    #: pulse duration in HOURS; ignored when mode == 'instant'. Was an unreachable make_ptc
    #: default of 8.0, which made a pulse-duration experiment impossible to express.
    pulse: float = 8.0

    #: Poincare section species (orbit phase condition). None -> the model's own default.
    section: Optional[str] = None
    #: phase readout species for the PTC. None -> model default. These are separate because
    #: they have different requirements -- REPO_MAP hazard 14.
    readout: Optional[str] = None

    # ---- PTC grid -------------------------------------------------------------------- #
    n_phase: int = 20
    n_dose: int = 14
    #: dose window as multiples of the measured S_crit. The defaults put S_crit at the 25th
    #: log-percentile so ~75% of the range is type-0, where the twist lives (fit/doses.py).
    lo_factor: float = 0.5
    max_factor: float = 8.0

    # ---- integration ----------------------------------------------------------------- #
    backend: str = 'diffrax'
    dt: float = 0.02
    #: periods of transient to discard before reading asymptotic phase. None -> derived from
    #: the model's leading Floquet multiplier.
    skip_p: Optional[int] = None

    # ---- target surface -------------------------------------------------------------- #
    #: 'pinned'   target fixed at the seed's own measured singularity (default)
    #: 'profiled' (k, psi) re-fitted at every evaluation -- the target moves with the model
    #: 'explicit' target fixed at `target_scrit` / `target_phi` given below
    target_mode: str = 'pinned'
    target_scrit: Optional[float] = None
    target_phi: Optional[float] = None

    # ---- cost ------------------------------------------------------------------------ #
    w_osc: float = 0.2
    w_amp: float = 1.0
    bound: float = 3.0

    # ---- optimizer ------------------------------------------------------------------- #
    optimizer: str = 'cma'                      # 'cma' | 'bobyqa' | 'lbfgs' | 'lm'
    #: TOTAL cost evaluations. One currency for every optimizer, so budgets are comparable.
    maxfev: int = 4000
    #: CMA population. None -> `recommend_popsize`, which oversubscribes to `workers`.
    popsize: Optional[int] = None
    sigma0: float = 0.5
    cma_mode: str = 'anneal'                    # 'anneal' | 'ipop'
    restarts: int = 2
    #: WHERE EACH SEED STARTS. This is what decides whether a seed sweep measures anything.
    #:
    #: 'base'      every seed starts from the base parameters (CMA's default behaviour).
    #:             MEASURED: the seeds' initial population CENTROIDS then sit 0.55 apart while
    #:             their own members are 2.9 apart -- the clouds overlap almost completely, so
    #:             the seeds differ by sampling noise and all explore one neighbourhood. Fine
    #:             for "how good an optimum is there near base", useless for multimodality.
    #: 'dispersed' seeds are placed on a Latin hypercube of radius `start_radius`, so N seeds
    #:             COVER the region instead of clustering in it. This is the setting the
    #:             multimodality question needs.
    #: 'random'    independent uniform draws of radius `start_radius`; simpler than a Latin
    #:             hypercube and clumpier for small N.
    start: str = 'base'
    #: dispersion radius in gauge-quotient log-parameter units.
    #:
    #: MEASURED (almeida/BMAL1, 6 Latin-hypercube starts per radius, fraction of the PTC grid
    #: alive at the start):
    #:
    #:     radius  0.25   0.50   1.00   1.50   3.00
    #:     alive   0.67   0.50   0.17   0.00   0.00
    #:     dead    2/6    1/6    5/6    6/6    4/6
    #:
    #: Viability collapses almost at once, so 0.25 is the largest defensible default and even
    #: it starts a third of the seeds on a dead clock. Note what this implies: at radius 0.25
    #: the per-coordinate spread is ~0.14 while CMA's own sigma0=0.5 population spans ~0.5, so
    #: ISOTROPIC dispersion is NARROWER than the sampling cloud it is meant to escape. It
    #: cannot separate seeds without killing the oscillator, and 'dispersed' is therefore a
    #: weak instrument for multimodality. Dispersing along the model's FLAT directions instead
    #: -- which move far in parameter space while preserving the dynamics -- is the right tool
    #: and is not built yet.
    start_radius: float = 0.25

    #: seeds to run. A LIST runs several independent searches in one job, which is what the
    #: multimodality question needs; results are saved per seed and compared.
    seeds: Any = field(default_factory=lambda: [0])

    # ---- execution ------------------------------------------------------------------- #
    workers: int = 1
    tag: Optional[str] = None
    verbose: bool = True
    log_every: int = 10
    notes: str = ''

    # ------------------------------------------------------------------ validation ---- #
    def validate(self):
        """Fail LOUDLY and EARLY on an impossible configuration.

        A campaign submits many jobs at once; a typo that only surfaces three hours into a
        queued run costs far more than one that surfaces at submission.
        """
        bad = []
        if self.mode not in ('instant', 'pulse'):
            bad.append(f"mode {self.mode!r} must be 'instant' or 'pulse'")
        if self.mode == 'pulse' and not (self.pulse > 0):
            bad.append(f"pulse must be > 0 for mode='pulse', got {self.pulse}")
        if self.target_mode not in ('pinned', 'profiled', 'explicit'):
            bad.append(f"target_mode {self.target_mode!r} must be pinned/profiled/explicit")
        if self.target_mode == 'explicit' and (self.target_scrit is None
                                               or self.target_phi is None):
            bad.append("target_mode='explicit' needs both target_scrit and target_phi")
        if self.optimizer not in ('cma', 'bobyqa', 'lbfgs', 'lm'):
            bad.append(f"optimizer {self.optimizer!r} is not one of cma/bobyqa/lbfgs/lm")
        if self.n_phase < 4 or self.n_dose < 3:
            bad.append(f"grid {self.n_phase}x{self.n_dose} is too coarse to detect a winding")
        if not (0 < self.lo_factor < self.max_factor):
            bad.append(f"need 0 < lo_factor < max_factor, got {self.lo_factor}/"
                       f"{self.max_factor}")
        if self.workers < 1:
            bad.append(f"workers must be >= 1, got {self.workers}")
        if self.optimizer != 'cma' and self.workers > 1:
            bad.append(f"workers>1 parallelises a CMA POPULATION; optimizer "
                       f"{self.optimizer!r} is sequential, so it would gain nothing")
        if not self.seed_list:
            bad.append("seeds is empty")
        # SEEDS ONLY MEAN SOMETHING FOR A STOCHASTIC SEARCH. BOBYQA, L-BFGS and LM all start
        # from the same point and take deterministic steps, so N seeds give N identical runs --
        # measured: a 2-seed BOBYQA smoke campaign returned cost spread 0.000000 and pairwise
        # distance 0.000. That is not multimodality evidence, it is the same run twice, and on
        # a cluster it is N-1 wasted array tasks.
        if self.start not in ('base', 'dispersed', 'random'):
            bad.append(f"start {self.start!r} must be base/dispersed/random")
        if self.start != 'base' and not (0 < self.start_radius <= self.bound):
            bad.append(f"start_radius must be in (0, bound={self.bound}], got "
                       f"{self.start_radius}")
        if (len(self.seed_list) > 1 and self.start == 'base'
                and self.optimizer == 'cma'):
            print(f"  [config] NOTE: {len(self.seed_list)} seeds with start='base' all begin "
                  f"from the same point and differ only by sampling noise. For a MULTIMODALITY "
                  f"sweep use start='dispersed'.")
        if len(self.seed_list) > 1 and self.optimizer in ('bobyqa', 'lbfgs', 'lm'):
            bad.append(f"optimizer {self.optimizer!r} is DETERMINISTIC from a fixed start, so "
                       f"{len(self.seed_list)} seeds would produce {len(self.seed_list)} "
                       f"identical runs. Use optimizer='cma' for a seed sweep, or seeds=[0].")
        if bad:
            raise SystemExit("invalid RunConfig:\n  " + "\n  ".join(bad))
        return self

    # ------------------------------------------------------------------ helpers ------- #
    @property
    def seed_list(self):
        s = self.seeds
        if isinstance(s, (int, np.integer)):
            return [int(s)]
        return [int(x) for x in s]

    def label(self):
        """A short, stable, human-readable name derived from what differs from the defaults.

        Used for run tags in a campaign so a directory listing says what each run WAS, instead
        of `job_0` ... `job_23`.
        """
        d = RunConfig()
        parts = []
        for f in fields(self):
            if f.name in _LABEL_SKIP:
                continue
            v, dv = getattr(self, f.name), getattr(d, f.name)
            if v == dv:
                continue
            if f.name == 'seeds':
                v = 's' + '-'.join(str(x) for x in self.seed_list)
            parts.append(f"{f.name}-{v}")
        safe = [str(p).replace(' ', '').replace('/', '-').replace('=', '-') for p in parts]
        return '_'.join(safe) or 'defaults'

    def label_over(self, axes):
        """Label using only `axes` -- what VARIES within a campaign.

        `label()` compares against the global defaults, which inside a campaign is noise: if
        every entry sets section=PER and readout=REV, repeating that in all four directory
        names hides the one field that actually differs.
        """
        parts = []
        for name in sorted(axes):
            v = self.seed_list if name == 'seeds' else getattr(self, name)
            parts.append(f"{name}-{v}")
        safe = [str(p).replace(' ', '').replace('/', '-').replace('=', '-') for p in parts]
        return '_'.join(safe) or 'run'

    def identity(self):
        """A canonical key in which settings that CANNOT MATTER are erased.

        `mode=instant` ignores `pulse`, so sweeping mode x pulse otherwise yields two identical
        instant runs -- a wasted array task each, and two directories that look like different
        experiments. Likewise the CMA knobs mean nothing to a sequential optimizer.
        """
        d = self.to_dict()
        d.pop('tag', None)
        d.pop('notes', None)
        d.pop('workers', None)
        if d.get('mode') != 'pulse':
            d['pulse'] = None
        if d.get('optimizer') != 'cma':
            for k in ('popsize', 'sigma0', 'cma_mode', 'restarts'):
                d[k] = None
        if d.get('target_mode') != 'explicit':
            d['target_scrit'] = d['target_phi'] = None
        return json.dumps(d, sort_keys=True, default=str)

    def to_dict(self):
        d = dataclasses.asdict(self)
        d['seeds'] = self.seed_list
        return d

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return path

    @staticmethod
    def from_dict(d):
        known = {f.name for f in fields(RunConfig)}
        unknown = set(d) - known
        if unknown:
            # A silently ignored key is a setting the user THINKS they changed. Refuse it.
            raise SystemExit(f"unknown config key(s): {sorted(unknown)}\n"
                             f"known keys: {sorted(known)}")
        return RunConfig(**d)

    @staticmethod
    def load(path):
        with open(path, encoding='utf-8') as fh:
            return RunConfig.from_dict(json.load(fh))

    # ------------------------------------------------------------------ sweeps -------- #
    def expand(self):
        """Cartesian product over any field given as a list -> concrete RunConfigs.

        `seeds` is NOT swept: a list of seeds means "run these seeds in this job", which is the
        unit the multimodality question is asked in. Everything else expands, so

            target=['BMAL1','PER','CRY'], popsize=[12,33], seeds=[0,1,2,3]

        is six runs of four seeds each, not 24 single-seed runs.
        """
        axes = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name != 'seeds' and isinstance(v, (list, tuple)):
                axes[f.name] = list(v)
        if not axes:
            return [copy.deepcopy(self)]
        names = sorted(axes)
        out = []
        for combo in itertools.product(*(axes[n] for n in names)):
            c = copy.deepcopy(self)
            for n, v in zip(names, combo):
                setattr(c, n, v)
            out.append(c)
        return out

    # ------------------------------------------------------------------ CLI ----------- #
    @staticmethod
    def add_arguments(ap):
        """Every field becomes a flag, derived from the dataclass so the two cannot drift."""
        d = RunConfig()
        for f in fields(RunConfig):
            if f.name in ('seeds', 'notes'):
                continue
            flag = '--' + f.name.replace('_', '-')
            cur = getattr(d, f.name)
            if f.type in ('bool', bool):
                ap.add_argument(flag, dest=f.name, action='store_true', default=None)
                ap.add_argument('--no-' + f.name.replace('_', '-'), dest=f.name,
                                action='store_false')
            else:
                typ = (int if f.name in ('n_phase', 'n_dose', 'maxfev', 'popsize', 'workers',
                                         'restarts', 'skip_p', 'log_every')
                       else float if f.name in ('pulse', 'lo_factor', 'max_factor', 'dt',
                                                'w_osc', 'w_amp', 'bound', 'sigma0',
                                                'target_scrit', 'target_phi')
                       else str)
                ap.add_argument(flag, dest=f.name, type=typ, default=None,
                                help=f"(default {cur!r})")
        ap.add_argument('--seeds', type=str, default=None,
                        help="comma-separated, e.g. 0,1,2,3 (default 0). All seeds run in "
                             "THIS job; they are the unit of the multimodality question.")
        ap.add_argument('--config', type=str, default=None,
                        help='JSON file of settings; explicit flags override it')
        ap.add_argument('--print-config', action='store_true',
                        help='resolve settings, print them, and exit without running')
        return ap

    @staticmethod
    def from_args(args):
        """File first, then explicit flags on top, then environment defaults. A flag left
        unset is None, so it cannot silently clobber a value from --config."""
        cfg = RunConfig.load(args.config) if getattr(args, 'config', None) else RunConfig()
        for f in fields(RunConfig):
            v = getattr(args, f.name, None)
            if v is not None:
                setattr(cfg, f.name, v)
        if getattr(args, 'seeds', None):
            cfg.seeds = [int(x) for x in str(args.seeds).replace(' ', '').split(',') if x]
        if cfg.model == 'almeida' and os.environ.get('MODEL'):
            cfg.model = os.environ['MODEL']
        return cfg


def start_points(cfg, n_free):
    """Starting point for each seed, in gauge-quotient coordinates.

    Returned for ALL seeds at once and indexed by position, because a Latin hypercube is a
    JOINT construction: its whole point is that the N points cover the region between them,
    which cannot be reproduced by generating one point at a time. The design is a deterministic
    function of (seeds, radius, n_free), so a run is reproducible and array task 3 gets the
    same start whether or not tasks 0-2 ever ran.
    """
    seeds = cfg.seed_list
    n = int(n_free)
    if cfg.start == 'base':
        return {sd: np.zeros(n) for sd in seeds}
    if cfg.start == 'random':
        return {sd: np.random.default_rng(1000 + sd).uniform(
            -cfg.start_radius, cfg.start_radius, n) for sd in seeds}
    from scipy.stats import qmc
    # seeded by the SEED LIST, so the design depends on which seeds were asked for and not on
    # the order they happen to run in
    eng = qmc.LatinHypercube(d=n, seed=abs(hash(tuple(seeds))) % (2 ** 31))
    pts = qmc.scale(eng.random(len(seeds)), -cfg.start_radius, cfg.start_radius)
    return {sd: pts[i] for i, sd in enumerate(seeds)}


def describe(cfg):
    """A one-screen dump of the resolved settings, with non-defaults marked.

    Printed at the top of every run. When a campaign result looks strange the first question is
    always "what was this actually run with", and the answer should be in the log rather than
    reconstructed from a filename.
    """
    d = RunConfig()
    lines = [f"{'setting':<16s} {'value':<28s} {'default'}"]
    lines.append('-' * 66)
    for f in fields(cfg):
        v, dv = getattr(cfg, f.name), getattr(d, f.name)
        if f.name == 'seeds':
            v, dv = cfg.seed_list, d.seed_list
        mark = '' if v == dv else '  <-'
        lines.append(f"{f.name:<16s} {str(v):<28s} {str(dv)}{mark}")
    return '\n'.join(lines)
