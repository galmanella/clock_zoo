"""
analysis/genemap.py
===================
WHICH SPECIES OF WHICH MODEL IS "THE SAME GENE".

The three models describe the same clock at three different levels of description, so a
cross-model comparison has to say -- explicitly, in one place, reviewable as a diff -- what it
is calling equivalent. This module is that declaration. It computes nothing.

    $PY -m analysis.genemap            # print the table, and audit it against the models

THE TWO AXES OF THE MAPPING
    gene   the clock component: Bmal1, Per, Cry, RevErb, Dbp, PerCry, Ror, E4bp4.
    level  WHERE in that gene's expression the perturbation lands:

        'mrna'      the transcript                Korencic `<gene>x`, Goldbeter `M*`
        'protein'   the cytoplasmic/whole-cell protein or its model's stand-in for it
        'nuclear'   the nuclear form, where the model resolves transport
        'complex'   the PER-CRY heterodimer, where the model has one as a species

    Almeida is a TRANSCRIPTION-FACTOR-level model: it has no mRNA species at all, and its
    eight states are the active regulators. They are mapped to 'protein'. Korencic's chain is
    x -> y -> z with no protein/mRNA distinction inside it; the model's own docstring calls x
    the transcript and z the regulator, and y "a modelling device with no clean experimental
    counterpart" (it is not perturbable). So x -> 'mrna', z -> 'protein'.

WHY LEVEL MATTERS MORE THAN IT LOOKS
    Pushing an mRNA and pushing its protein are not the same experiment: the mRNA route is
    filtered through translation and so is both delayed and low-pass. A figure that puts
    Goldbeter's MP beside Almeida's PER without saying which is which invites reading a
    LEVEL difference as a MODEL difference. Every comparison figure therefore labels the
    level, and `pairs(gene, level=...)` exists so a like-for-like row can be requested.

WHAT IS DELIBERATELY ABSENT
    Goldbeter has no REV-ERB and no DBP species -- its negative loop is PER/CRY and its
    positive arm is BMAL1 with a lumped activation, so those two genes are 2-model
    comparisons, not 3. Korencic has no complex. Almeida's ROR and E4BP4 have no counterpart
    anywhere else and are single-model panels. None of that is an error to be worked around;
    it is the structural difference the zoo exists to expose.
"""
import argparse

#: Levels, ordered from transcript outwards. Figures use this order for rows/columns.
LEVELS = ('mrna', 'protein', 'nuclear', 'complex')

#: gene -> model -> [(state, level), ...].  THE declaration. Edit here, nowhere else.
GENE_MAP = {
    'Bmal1': {
        'almeida':   [('BMAL1', 'protein')],
        'korencic':  [('Bmalx', 'mrna'), ('Bmalz', 'protein')],
        'goldbeter': [('MB', 'mrna'), ('BC', 'protein'), ('BN', 'nuclear')],
    },
    'Per': {
        'almeida':   [('PER', 'protein')],
        'korencic':  [('Perx', 'mrna'), ('Perz', 'protein')],
        'goldbeter': [('MP', 'mrna'), ('PC', 'protein')],
    },
    'Cry': {
        'almeida':   [('CRY', 'protein')],
        'korencic':  [('Cryx', 'mrna'), ('Cryz', 'protein')],
        'goldbeter': [('MC', 'mrna'), ('CC', 'protein')],
    },
    'RevErb': {
        'almeida':   [('REV', 'protein')],
        'korencic':  [('Reverbx', 'mrna'), ('Reverbz', 'protein')],
        # Goldbeter 2003 has no REV-ERB species.
    },
    'Dbp': {
        'almeida':   [('DBP', 'protein')],
        'korencic':  [('Dbpx', 'mrna'), ('Dbpz', 'protein')],
        # Goldbeter 2003 has no DBP species.
    },
    'PerCry': {
        'almeida':   [('PER_CRY', 'complex')],
        'goldbeter': [('PCC', 'complex'), ('PCN', 'nuclear')],
        # Korencic has no complexes -- its genes never bind each other explicitly.
    },
    'Ror': {
        'almeida':   [('ROR', 'protein')],
    },
    'E4bp4': {
        'almeida':   [('E4BP4', 'protein')],
    },
}

#: Order the models are drawn in, everywhere.
MODELS = ('almeida', 'korencic', 'goldbeter')

#: Genes, in the order figures should present them: the shared ones first, by how many models
#: carry them, then the model-specific tails.
GENES = ('Bmal1', 'Per', 'Cry', 'RevErb', 'Dbp', 'PerCry', 'Ror', 'E4bp4')


def genes(min_models=1):
    """Gene names carried by at least `min_models` models, in figure order."""
    return [g for g in GENES if len(GENE_MAP[g]) >= min_models]


def pairs(gene, level=None, models=MODELS):
    """[(model, state, level)] for one gene, in model order. `level` filters (str or tuple)."""
    want = None if level is None else ({level} if isinstance(level, str) else set(level))
    out = []
    for m in models:
        for st, lv in GENE_MAP.get(gene, {}).get(m, []):
            if want is None or lv in want:
                out.append((m, st, lv))
    return out


def gene_of(model, state):
    """(gene, level) for a model's state, or (None, None) if it is unmapped."""
    for g, bym in GENE_MAP.items():
        for st, lv in bym.get(model, []):
            if st == state:
                return g, lv
    return None, None


def targets_of(model):
    """{state: (gene, level)} for every mapped state of a model."""
    return {st: (g, lv) for g, bym in GENE_MAP.items()
            for st, lv in bym.get(model, [])}


def label(model, state, level=None, gene=None):
    """The label a comparison panel carries: 'goldbeter MP (Per, mrna)'."""
    if gene is None or level is None:
        gene, level = gene_of(model, state)
    return f"{model} {state}" + (f" ({level})" if level else "")


def audit(verbose=True):
    """Every mapped state must be a PERTURBABLE target of its model, and every perturbable
    target should be mapped -- an unmapped one silently vanishes from every comparison.

    This is the check that the declaration above has not drifted from the models. It imports
    them, so it is the one function here that is not free.
    """
    from models import get_model
    problems = []
    for m in MODELS:
        model = get_model(m)
        allowed = set(model.perturbable_targets())
        mapped = targets_of(m)
        for st in mapped:
            if st not in allowed:
                problems.append(f"{m}: {st!r} is mapped but is not a perturbable target")
        unmapped = sorted(allowed - set(mapped))
        if unmapped:
            problems.append(f"{m}: perturbable targets NOT in the gene map: {unmapped}")
        if verbose:
            print(f"  {m:10s} {len(mapped)}/{len(allowed)} perturbable targets mapped")
    if verbose:
        for p in problems:
            print(f"  [BAD] {p}")
        print(f"  [{'PASS' if not problems else 'FAIL'}] gene map vs models")
    return problems


def print_table():
    print(f"\n{'=' * 78}\nGENE MAP -- which species is 'the same gene'\n{'=' * 78}")
    print(f"  {'gene':8s} {'level':9s} " + ' '.join(f'{m:<14s}' for m in MODELS))
    for g in GENES:
        rows = sorted({lv for _m, _s, lv in pairs(g)}, key=LEVELS.index)
        for k, lv in enumerate(rows):
            cells = []
            for m in MODELS:
                st = [s for (mm, s, l) in pairs(g, lv) if mm == m]
                cells.append(f"{(st[0] if st else '--'):<14s}")
            print(f"  {(g if k == 0 else ''):8s} {lv:9s} " + ' '.join(cells))
    print(f"\n  {len(genes(2))} genes exist in >=2 models: {', '.join(genes(2))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description='the cross-model gene mapping, and its audit')
    ap.add_argument('--no-audit', action='store_true')
    a = ap.parse_args(argv)
    print_table()
    if a.no_audit:
        return 0
    print()
    return 0 if not audit() else 1


if __name__ == '__main__':
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
#  SCOPE
# --------------------------------------------------------------------------- #
#: WHICH LEVEL each model contributes to the comparison.
#:
#: The experiment perturbs a GENE, and the honest cross-model row is the one closest to
#: "more transcript of gene X". So where a model resolves transcription, only its mRNA
#: species are in scope; the protein forms, the phospho-forms and the complexes are internal
#: machinery, not things an experiment expresses. Almeida is the exception BY CONSTRUCTION:
#: it is a reduced, transcription-factor-level model with no mRNA species at all, so its
#: regulator states ARE its gene products and they stand in for the mRNA row.
#:
#: Complexes are out of scope everywhere -- Almeida's PER_CRY as much as Goldbeter's
#: PCC/PCN -- because "overexpress the heterodimer" is not an experiment, and keeping one
#: model's complex while dropping another's would put a structural difference into the
#: comparison as if it were a biological one.
SCOPE_LEVELS = {
    'almeida':   ('protein',),      # no mRNA exists; the TFs are the gene products
    'korencic':  ('mrna',),         # the x stage; y and z are the delay chain
    'goldbeter': ('mrna',),         # MP / MC / MB (and MR in the Rev-erb variant)
}


def scope_targets(model, levels=None):
    """The in-scope states of a model, in gene order."""
    want = set(levels or SCOPE_LEVELS.get(model, LEVELS))
    return [st for g in GENES for st, lv in GENE_MAP[g].get(model, []) if lv in want]


def in_scope(model, state, levels=None):
    _g, lv = gene_of(model, state)
    return lv in set(levels or SCOPE_LEVELS.get(model, LEVELS))


def scope_report():
    print(f"\n{'=' * 78}\nCOMPARISON SCOPE\n{'=' * 78}")
    tot = 0
    for m in MODELS:
        ts = scope_targets(m)
        tot += len(ts)
        allt = [s for g in GENES for s, _l in GENE_MAP[g].get(m, [])]
        drop = [s for s in allt if s not in ts]
        print(f"  {m:10s} level={'/'.join(SCOPE_LEVELS.get(m, ())):8s} "
              f"{len(ts)}/{len(allt)} in scope: {', '.join(ts)}")
        if drop:
            print(f"             dropped: {', '.join(drop)}")
    print(f"\n  {tot} targets in the comparison.")
    for g in GENES:
        ms = sorted({m for m in MODELS if any(s in scope_targets(m)
                                              for s, _l in GENE_MAP[g].get(m, []))},
                    key=MODELS.index)
        print(f"    {g:8s} {len(ms)} model(s): {', '.join(ms) if ms else '--'}")
