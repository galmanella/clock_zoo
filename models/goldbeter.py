"""
models/goldbeter.py
===================
Leloup & Goldbeter (2003) mammalian circadian model. 16 ODEs, 52 parameters, period ~23.85 h.
Ported from `circadian_modeling.ipynb` cell 156; the equations and nominal values are
unchanged (see the note on four dropped entries below).

STRUCTURE -- the most mechanistically detailed of the three models here
    3 mRNAs (MP, MC, MB) transcribed under nuclear BMAL1: BN activates Per and Cry (Hill n),
    and represses Bmal1 (Hill m). Each mRNA is removed by a Michaelian term plus a
    first-order one.

    PER and CRY proteins then run a full post-translational cascade:
        PC, CC          cytosolic, unphosphorylated
        PCP, CCP        phosphorylated (V1/V2 pairs), then Michaelis-degraded
        PC + CC <-> PCC reversible complexation (k3 / k4)
        PCC <-> PCN     nuclear transport (k1 / k2), each with its own phospho-form
        BC <-> BN       BMAL1 transport (k5 / k6), each with its own phospho-form
        BN + PCN <-> IN the inactive nuclear complex (k7 / k8) that closes the loop

    So unlike Almeida (transcription-factor level) and Korencic (delay chains), this model
    resolves phosphorylation, complexation and compartments explicitly -- and it is the
    closest of the three in size and spirit to Mirsky.

FOUR NOTEBOOK PARAMETERS ARE DROPPED, DELIBERATELY
    `V1PCP`, `V1BP`, `V2PCP`, `V2BP` appear in the notebook's parameter dict but in NONE of
    its 16 equations -- verified numerically: perturbing each leaves the RHS bit-identical.
    (The paper's phospho-form transitions are driven by V1PC/V2PC and V1B/V2B, which the
    equations do use.) They are excluded here rather than carried along, because a parameter
    that provably cannot affect the model would contribute four guaranteed-flat directions to
    every identifiability count -- silently inflating exactly the quantity this project is
    trying to measure. 52 parameters, all of which do something.

CONDITIONING
    All 16 states are unimodal over the cycle and strictly positive, but the phospho-forms run
    small (BCP amplitude 0.031, PCP 0.067) against MB at 7.6. Expect the perturbation targets
    to matter more here than in the other two models.
"""
import numpy as np

from models.base import ClockModel, JaxDictParams, register_model

_STATES = ['MP', 'MC', 'MB', 'PC', 'CC', 'PCP', 'CCP', 'PCC', 'PCN', 'PCCP', 'PCNP',
           'BC', 'BCP', 'BN', 'BNP', 'IN']

#: The 13 protein states, all forced into one concentration unit (see scale_coupled_species).
_PROTEINS = ['PC', 'CC', 'PCP', 'CCP', 'PCC', 'PCN', 'PCCP', 'PCNP',
             'BC', 'BCP', 'BN', 'BNP', 'IN']

_PARAMS = {
    # transcription
    'vsP': 1.5, 'vsC': 1.1, 'vsB': 1.0, 'KAP': 0.7, 'KAC': 0.6, 'KIB': 2.2,
    'n': 4.0, 'm': 2.0,
    # mRNA degradation (Michaelian + first order)
    'vmP': 1.1, 'vmC': 1.0, 'vmB': 0.8, 'KmP': 0.31, 'KmC': 0.4, 'KmB': 0.4,
    'kdmp': 0.01, 'kdmc': 0.01, 'kdmb': 0.01,
    # translation
    'ksP': 0.6, 'ksC': 1.6, 'ksB': 0.12,
    # (de)phosphorylation maxima and shared thresholds
    'V1P': 0.4, 'V1C': 0.6, 'V1PC': 0.4, 'V1B': 0.5,
    'V2P': 0.3, 'V2C': 0.1, 'V2PC': 0.1, 'V2B': 0.1,
    'V3PC': 0.4, 'V4PC': 0.1, 'V3B': 0.5, 'V4B': 0.2,
    'Kp': 0.1, 'Kdp': 0.1, 'Kd': 0.3,
    # degradation of the phosphorylated forms
    'vdPC': 0.7, 'vdCC': 0.7, 'vdPCC': 0.7, 'vdPCN': 0.7,
    'vdBC': 0.5, 'vdBN': 0.6, 'vdIN': 0.8,
    # transport / complexation
    'k1': 0.4, 'k2': 0.2, 'k3': 0.4, 'k4': 0.2,
    'k5': 0.4, 'k6': 0.2, 'k7': 0.5, 'k8': 0.1,
    # non-specific protein degradation
    'kdn': 0.01, 'kdnc': 0.12,
}

#: A point ON the limit cycle: the MP maximum, from a 3000 h LSODA relaxation at rtol 1e-12.
_IC = {
    'MP': 4.558084, 'MC': 2.847143, 'MB': 7.572134, 'PC': 1.134696, 'CC': 7.658449,
    'PCP': 0.121579, 'CCP': 0.697409, 'PCC': 4.645191, 'PCN': 1.611693, 'PCCP': 0.245872,
    'PCNP': 0.216974, 'BC': 2.037729, 'BCP': 0.925405, 'BN': 0.883198, 'BNP': 0.316488,
    'IN': 0.602624,
}


class GoldbeterModel(JaxDictParams, ClockModel):
    """Leloup & Goldbeter 2003 mammalian clock (16 ODEs, 52 parameters)."""

    #: MP: largest relative amplitude (2.49), unimodal.
    reference_variable = 'MP'
    approx_period = 23.85

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
            raise KeyError(f"GoldbeterModel has no state {name!r}") from None

    @staticmethod
    def _rhs_terms(p, y, pw):
        """The 16 derivatives, written once and shared by numpy and JAX so they cannot drift.
        `pw(x, k)` is x**k for the backend in use; `y` is the (clamped) state, indexable."""
        MP, MC, MB, PC, CC, PCP, CCP, PCC, PCN, PCCP, PCNP, BC, BCP, BN, BNP, IN = \
            (y[i] for i in range(16))
        Kp, Kdp, Kd = p['Kp'], p['Kdp'], p['Kd']
        mm = lambda V, x, K: V * x / (K + x)             # Michaelis-Menten

        BNn, BNm = pw(BN, p['n']), pw(BN, p['m'])
        dMP = p['vsP'] * BNn / (pw(p['KAP'], p['n']) + BNn) - mm(p['vmP'], MP, p['KmP']) \
            - p['kdmp'] * MP
        dMC = p['vsC'] * BNn / (pw(p['KAC'], p['n']) + BNn) - mm(p['vmC'], MC, p['KmC']) \
            - p['kdmc'] * MC
        KIBm = pw(p['KIB'], p['m'])
        dMB = p['vsB'] * KIBm / (KIBm + BNm) - mm(p['vmB'], MB, p['KmB']) - p['kdmb'] * MB

        complexation = p['k3'] * PC * CC - p['k4'] * PCC      # PC + CC <-> PCC
        dPC = (p['ksP'] * MP - mm(p['V1P'], PC, Kp) + mm(p['V2P'], PCP, Kdp)
               - complexation - p['kdn'] * PC)
        dCC = (p['ksC'] * MC - mm(p['V1C'], CC, Kp) + mm(p['V2C'], CCP, Kdp)
               - complexation - p['kdnc'] * CC)
        dPCP = (mm(p['V1P'], PC, Kp) - mm(p['V2P'], PCP, Kdp)
                - mm(p['vdPC'], PCP, Kd) - p['kdn'] * PCP)
        dCCP = (mm(p['V1C'], CC, Kp) - mm(p['V2C'], CCP, Kdp)
                - mm(p['vdCC'], CCP, Kd) - p['kdn'] * CCP)

        transport = p['k1'] * PCC - p['k2'] * PCN             # PCC <-> PCN
        sequester = p['k7'] * BN * PCN - p['k8'] * IN         # BN + PCN <-> IN
        dPCC = (-mm(p['V1PC'], PCC, Kp) + mm(p['V2PC'], PCCP, Kdp)
                + complexation - transport - p['kdn'] * PCC)
        dPCN = (-mm(p['V3PC'], PCN, Kp) + mm(p['V4PC'], PCNP, Kdp)
                + transport - sequester - p['kdn'] * PCN)
        dPCCP = (mm(p['V1PC'], PCC, Kp) - mm(p['V2PC'], PCCP, Kdp)
                 - mm(p['vdPCC'], PCCP, Kd) - p['kdn'] * PCCP)
        dPCNP = (mm(p['V3PC'], PCN, Kp) - mm(p['V4PC'], PCNP, Kdp)
                 - mm(p['vdPCN'], PCNP, Kd) - p['kdn'] * PCNP)

        b_transport = p['k5'] * BC - p['k6'] * BN             # BC <-> BN
        dBC = (p['ksB'] * MB - mm(p['V1B'], BC, Kp) + mm(p['V2B'], BCP, Kdp)
               - b_transport - p['kdn'] * BC)
        dBCP = (mm(p['V1B'], BC, Kp) - mm(p['V2B'], BCP, Kdp)
                - mm(p['vdBC'], BCP, Kd) - p['kdn'] * BCP)
        dBN = (-mm(p['V3B'], BN, Kp) + mm(p['V4B'], BNP, Kdp)
               + b_transport - sequester - p['kdn'] * BN)
        dBNP = (mm(p['V3B'], BN, Kp) - mm(p['V4B'], BNP, Kdp)
                - mm(p['vdBN'], BNP, Kd) - p['kdn'] * BNP)

        dIN = sequester - mm(p['vdIN'], IN, Kd) - p['kdn'] * IN

        return [dMP, dMC, dMB, dPC, dCC, dPCP, dCCP, dPCC, dPCN, dPCCP, dPCNP,
                dBC, dBCP, dBN, dBNP, dIN]

    def derivatives(self, t, state, params=None):
        p = self.params if params is None else params
        y = np.maximum(np.asarray(state, float), 0.0)
        pw = lambda x, k: np.where(x > 0.0, np.where(x > 0.0, x, 1.0) ** k, 0.0)
        return np.array(self._rhs_terms(p, y, pw))

    # --- JAX --------------------------------------------------------------- #
    @staticmethod
    def jax_rhs(y, P):
        import jax.numpy as jnp

        def pw(x, k):
            # x**k with a finite gradient at x = 0 (JAX's pow rule contains x**k*log(x),
            # which is 0 * -inf = NaN there, and the clamp below puts states exactly at 0).
            xs = jnp.where(x > 0.0, x, 1.0)
            return jnp.where(x > 0.0, xs ** k, 0.0)

        yy = jnp.maximum(y, 0.0)
        return jnp.stack(GoldbeterModel._rhs_terms(P, yy, pw))

    # --- DIMENSIONS -------------------------------------------------------- #
    def scale_coupled_species(self):
        """All 13 protein states share one unit; the 3 mRNAs scale freely.

        The chain of shared additive terms: `k3*PC*CC` ties PC, CC and PCC; the V1/V2
        Michaelian pairs tie each species to its phospho-form; `k1*PCC`/`k2*PCN` tie PCC to
        PCN; `k5*BC`/`k6*BN` tie BC to BN; and `k7*BN*PCN`/`k8*IN` tie BN, PCN and IN. Union-
        find closes that into a single group. The mRNAs share no term with anything -- `ksP*MP`
        appears only in dPC -- so each is its own group.

        Predicted gauge: 3 mRNA scales + 1 protein scale + time = 5 generators of 52.
        """
        return [_PROTEINS]

    def parameter_dimensions(self):
        """Thresholds carry the units of what they SENSE; maximal rates the units of what they
        act on; first-order rates are unit-blind; second-order rates carry 1/[partner]."""
        from models.api import TIME
        PROT = 'PC'          # any protein-group representative: they share one gauge column
        d = {
            # transcription: production carries the mRNA's units; thresholds sense BN
            'vsP': {'MP': 1, TIME: 1}, 'vsC': {'MC': 1, TIME: 1}, 'vsB': {'MB': 1, TIME: 1},
            'KAP': {'BN': 1}, 'KAC': {'BN': 1}, 'KIB': {'BN': 1}, 'n': {}, 'm': {},
            # mRNA removal: Michaelian max-rate + its own threshold, plus a first-order term
            'vmP': {'MP': 1, TIME: 1}, 'vmC': {'MC': 1, TIME: 1}, 'vmB': {'MB': 1, TIME: 1},
            'KmP': {'MP': 1}, 'KmC': {'MC': 1}, 'KmB': {'MB': 1},
            'kdmp': {TIME: 1}, 'kdmc': {TIME: 1}, 'kdmb': {TIME: 1},
            # translation converts mRNA units into protein units
            'ksP': {PROT: 1, 'MP': -1, TIME: 1},
            'ksC': {PROT: 1, 'MC': -1, TIME: 1},
            'ksB': {PROT: 1, 'MB': -1, TIME: 1},
            # shared protein thresholds
            'Kp': {PROT: 1}, 'Kdp': {PROT: 1}, 'Kd': {PROT: 1},
            # first-order transport / complex dissociation / non-specific degradation
            'k1': {TIME: 1}, 'k2': {TIME: 1}, 'k4': {TIME: 1},
            'k5': {TIME: 1}, 'k6': {TIME: 1}, 'k8': {TIME: 1},
            'kdn': {TIME: 1}, 'kdnc': {TIME: 1},
            # second order: 1 / ([partner] * time)
            'k3': {PROT: -1, TIME: 1},           # k3 * PC * CC  ->  [PC]/time
            'k7': {PROT: -1, TIME: 1},           # k7 * BN * PCN ->  [BN]/time
        }
        # every remaining V*/vd* is a Michaelian maximal rate acting on a protein species
        for name in _PARAMS:
            if name not in d:
                d[name] = {PROT: 1, TIME: 1}
        return d

    # --- PERTURBATION / OBSERVABLES ---------------------------------------- #
    def perturbable_targets(self):
        """The species an experiment could plausibly drive: the three transcripts and the
        unphosphorylated, non-sequestered protein forms. The phospho-forms and the inactive
        complex IN are products of internal reactions, not things you express."""
        return ['MP', 'MC', 'MB', 'PC', 'CC', 'BC', 'BN', 'PCC', 'PCN']

    def observable_states(self):
        """The three transcripts -- what a luciferase reporter or RNA time course measures."""
        return ['MP', 'MC', 'MB']

    def observable_pairs(self):
        """(mRNA, its primary protein product), for protein-to-mRNA lag observables."""
        return [('MP', 'PC'), ('MC', 'CC'), ('MB', 'BC')]


register_model('goldbeter', GoldbeterModel)
