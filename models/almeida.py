"""
models/almeida.py
=================
Almeida, Rocha, Boada, Garcia-Ojalvo (2020) -- a transcription-factor-level mammalian clock.
8 ODEs, 18 parameters, period ~24.83 h. Ported from `circadian_modeling.ipynb` cell 8; the
equations and the nominal parameter set are unchanged.

STRUCTURE
    Three promoter activities drive everything:

        EBOX = ve * BMAL1 / (BMAL1 + ke + ker*BMAL1*CRY)     CLOCK:BMAL1, blocked by CRY
        RRE  = vr * ROR/(ROR + kr) * krr^2/(krr^2 + REV^2)   ROR activates, REV-ERB represses
        DBOX = vd * DBP/(DBP + kd) * kdr/(kdr + E4BP4)       DBP activates, E4BP4 represses

    and each gene product is a weighted sum of them minus its own turnover. PER and CRY
    additionally form the reversible complex PER:CRY, which titrates BMAL1 away (the
    `gamma_BP * BMAL1 * PER_CRY` term -- an irreversible co-degradation, not a tracked
    complex).

    Note the model is PROTEIN-LEVEL ONLY: there is no separate mRNA species, so
    `observable_pairs()` is empty and every state is directly perturbable.

WHY THIS MODEL FIRST
    It is the smallest realistic mammalian clock in the notebook, and it is exceptionally
    well conditioned: all 8 states are strictly unimodal over the cycle, all stay strictly
    positive (minimum 0.092, in BMAL1), and the local Jacobian's stiffness ratio is only
    ~2.5e3. So it exercises the whole stack without the numerical hazards that would confound
    a first validation.

NON-NEGATIVITY
    Both `derivatives` and `jax_rhs` clamp the state at 0 before evaluating. The notebook
    version does not. The two therefore agree exactly on the non-negative orthant (which is
    all that is physical, and where the validation gate compares them) and differ only where
    a concentration has gone negative -- which a perturbation can transiently cause and where
    the unclamped field is meaningless anyway.
"""
import numpy as np

from models.base import ClockModel, JaxDictParams, register_model

#: Denominator floor. Every denominator below is a sum of non-negative quantities and is
#: strictly positive at any sane parameter set; this only stops a 0/0 NaN from propagating if
#: a sweep drives a threshold parameter to ~0 while the corresponding state is also ~0. At
#: nominal scales (ke = 214, krr = 80) it is a ~1e-14 relative perturbation.
_EPS = 1e-12

_STATES = ['BMAL1', 'ROR', 'REV', 'DBP', 'E4BP4', 'CRY', 'PER', 'PER_CRY']

#: Nominal parameters (Almeida 2020 supplementary; notebook cell 8, verbatim).
_PARAMS = {
    'vr': 44.4,          # RRE   max activation rate
    'kr': 3.54,          # RRE   ROR half-activation
    'krr': 80.1,         # RRE   REV half-repression
    've': 30.3,          # EBOX  max activation rate
    'ke': 214.0,         # EBOX  BMAL1 half-activation
    'ker': 1.24,         # EBOX  CRY inhibition strength (1/[CRY])
    'vd': 202.0,         # DBOX  max activation rate
    'kd': 5.32,          # DBOX  DBP half-activation
    'kdr': 94.7,         # DBOX  E4BP4 half-repression
    'gamma_ROR': 2.55,   # ROR     turnover
    'gamma_REV': 0.4,    # REV     turnover
    'gamma_P': 0.844,    # PER     turnover
    'gamma_c': 2.34,     # CRY     turnover
    'gamma_DB': 0.156,   # DBP     turnover
    'gamma_E4': 0.295,   # E4BP4   turnover
    'gamma_PC': 0.191,   # PER + CRY -> PER:CRY   association (1/([PER] h))
    'gamma_CP': 0.141,   # PER:CRY -> PER + CRY   dissociation
    'gamma_BP': 2.58,    # BMAL1 + PER:CRY -> 0   co-degradation (1/([PER_CRY] h))
}

#: A point ON the limit cycle: the BMAL1 maximum, from a 3000 h LSODA relaxation at rtol
#: 1e-12 (|dBMAL1/dt| = 1.5e-4 there). Only ever used as the orbit solver's initial guess,
#: which never enters a derivative -- but starting on the cycle makes Newton converge in one
#: or two steps instead of needing a long transient.
_IC = {
    'BMAL1': 27.878922, 'ROR': 8.873550, 'REV': 60.295569, 'DBP': 3.203113,
    'E4BP4': 137.610948, 'CRY': 10.525914, 'PER': 10.105601, 'PER_CRY': 0.281659,
}


class AlmeidaModel(JaxDictParams, ClockModel):
    """Almeida et al. 2020 transcription-factor clock (8 ODEs, 18 parameters)."""

    #: BMAL1: the largest relative amplitude (4.34) and cleanly unimodal, which is what the
    #: orbit solver's phase condition f(y0)[ref] = 0 needs.
    reference_variable = 'BMAL1'
    approx_period = 24.83

    def __init__(self):
        self._state_names = list(_STATES)
        self._idx = {s: i for i, s in enumerate(_STATES)}
        self.params = dict(_PARAMS)

    # --- CORE -------------------------------------------------------------- #
    def get_initial_state(self):
        return np.array([_IC[s] for s in _STATES])

    def var_index(self, name):
        try:
            return self._idx[name]
        except KeyError:
            raise KeyError(f"AlmeidaModel has no state {name!r}; states are {_STATES}") from None

    def derivatives(self, t, state, params=None):
        p = self.params if params is None else params
        y = np.maximum(np.asarray(state, float), 0.0)
        BMAL1, ROR, REV, DBP, E4BP4, CRY, PER, PER_CRY = y

        EBOX = p['ve'] * BMAL1 / (BMAL1 + p['ke'] + p['ker'] * BMAL1 * CRY + _EPS)
        RRE = (p['vr'] * ROR / (ROR + p['kr'] + _EPS)
               * p['krr'] ** 2 / (p['krr'] ** 2 + REV ** 2 + _EPS))
        DBOX = (p['vd'] * DBP / (DBP + p['kd'] + _EPS)
                * p['kdr'] / (p['kdr'] + E4BP4 + _EPS))

        bind = p['gamma_PC'] * PER * CRY - p['gamma_CP'] * PER_CRY      # PER + CRY -> PER:CRY
        titr = p['gamma_BP'] * BMAL1 * PER_CRY                          # BMAL1 + PER:CRY -> 0

        return np.array([
            RRE - titr,                                                 # BMAL1
            EBOX + RRE - p['gamma_ROR'] * ROR,                          # ROR
            2 * EBOX + DBOX - p['gamma_REV'] * REV,                     # REV
            EBOX - p['gamma_DB'] * DBP,                                 # DBP
            2 * RRE - p['gamma_E4'] * E4BP4,                            # E4BP4
            EBOX + 2 * RRE - bind - p['gamma_c'] * CRY,                 # CRY
            EBOX + DBOX - bind - p['gamma_P'] * PER,                    # PER
            bind - titr,                                                # PER:CRY
        ])

    # --- JAX --------------------------------------------------------------- #
    @staticmethod
    def jax_rhs(y, P):
        """Pure, jittable dy/dt. Term-for-term identical to `derivatives`."""
        import jax.numpy as jnp
        yy = jnp.maximum(y, 0.0)
        BMAL1, ROR, REV, DBP, E4BP4, CRY, PER, PER_CRY = (yy[i] for i in range(8))

        EBOX = P['ve'] * BMAL1 / (BMAL1 + P['ke'] + P['ker'] * BMAL1 * CRY + _EPS)
        RRE = (P['vr'] * ROR / (ROR + P['kr'] + _EPS)
               * P['krr'] ** 2 / (P['krr'] ** 2 + REV ** 2 + _EPS))
        DBOX = (P['vd'] * DBP / (DBP + P['kd'] + _EPS)
                * P['kdr'] / (P['kdr'] + E4BP4 + _EPS))

        bind = P['gamma_PC'] * PER * CRY - P['gamma_CP'] * PER_CRY
        titr = P['gamma_BP'] * BMAL1 * PER_CRY

        return jnp.stack([
            RRE - titr,
            EBOX + RRE - P['gamma_ROR'] * ROR,
            2 * EBOX + DBOX - P['gamma_REV'] * REV,
            EBOX - P['gamma_DB'] * DBP,
            2 * RRE - P['gamma_E4'] * E4BP4,
            EBOX + 2 * RRE - bind - P['gamma_c'] * CRY,
            EBOX + DBOX - bind - P['gamma_P'] * PER,
            bind - titr,
        ])

    # --- DIMENSIONS -------------------------------------------------------- #
    def scale_coupled_species(self):
        """Species forced to share one concentration unit.

        The rule (models/api.py): any two species whose rate equations share an ADDITIVE TERM
        must be counted in the same units. Almeida has no dimer stoichiometry to appeal to --
        the coupling comes from the shared promoter activities, which is exactly why the
        general form of the rule matters here.

        Each group below is one shared term. They overlap, and union-find merges them into a
        single group of all 8 species -- so the model is predicted to have just TWO gauge
        generators (one concentration + one time) out of 18 parameters. `gauge/` derives that
        count rather than trusting this comment.
        """
        return [
            ['ROR', 'REV', 'DBP', 'CRY', 'PER'],   # all receive EBOX
            ['BMAL1', 'ROR', 'E4BP4', 'CRY'],      # all receive RRE
            ['REV', 'PER'],                        # all receive DBOX
            ['CRY', 'PER', 'PER_CRY'],             # share gamma_PC*PER*CRY and gamma_CP*PER_CRY
            ['BMAL1', 'PER_CRY'],                  # share gamma_BP*BMAL1*PER_CRY
        ]

    def parameter_dimensions(self):
        """param -> {species: exponent, ..., TIME: exponent}. See models/api.py.

        Thresholds carry the units of the species they SENSE. Maximal rates carry the units of
        what they produce -- and because a single activity feeds several species, that choice
        of representative is arbitrary but harmless: all 8 sit in one scale group, so any of
        them selects the same gauge column.
        """
        from models.api import TIME
        return {
            # EBOX = ve * BMAL1 / (BMAL1 + ke + ker*BMAL1*CRY)
            've': {'ROR': 1, TIME: 1},        # a production rate: [product]/time
            'ke': {'BMAL1': 1},               # added to BMAL1 in the denominator
            'ker': {'CRY': -1},               # ker*BMAL1*CRY must match BMAL1 => 1/[CRY]
            # RRE = vr * ROR/(ROR+kr) * krr^2/(krr^2+REV^2)
            'vr': {'BMAL1': 1, TIME: 1},
            'kr': {'ROR': 1},
            'krr': {'REV': 1},
            # DBOX = vd * DBP/(DBP+kd) * kdr/(kdr+E4BP4)
            'vd': {'REV': 1, TIME: 1},
            'kd': {'DBP': 1},
            'kdr': {'E4BP4': 1},
            # first-order turnover: 1/time, unit-blind
            'gamma_ROR': {TIME: 1}, 'gamma_REV': {TIME: 1}, 'gamma_P': {TIME: 1},
            'gamma_c': {TIME: 1}, 'gamma_DB': {TIME: 1}, 'gamma_E4': {TIME: 1},
            'gamma_CP': {TIME: 1},                       # PER:CRY -> PER + CRY, first order
            # second order: 1/([partner] * time)
            'gamma_PC': {'PER': -1, TIME: 1},            # gamma_PC*PER*CRY ~ [CRY]/time
            'gamma_BP': {'PER_CRY': -1, TIME: 1},        # gamma_BP*BMAL1*PER_CRY ~ [BMAL1]/time
        }

    # --- PERTURBATION / OBSERVABLES ---------------------------------------- #
    def perturbable_targets(self):
        """Every state is a transcription factor an experiment could overexpress. PER_CRY is
        included because the complex is an explicit species here, and pushing it is a
        genuinely different perturbation from pushing PER or CRY separately."""
        return list(_STATES)

    def observable_states(self):
        return list(_STATES)

    def observable_pairs(self):
        return []                    # protein-level model: no mRNA/protein pairs


register_model('almeida', AlmeidaModel)
