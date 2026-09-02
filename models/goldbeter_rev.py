"""
models/goldbeter_rev.py
=======================
Leloup & Goldbeter (2003) mammalian circadian model, EXTENDED WITH REV-ERBalpha.
19 ODEs, 63 parameters. Ported from `circadian_modeling.ipynb` cell 156
(`LeloupGoldbeterModelRevErb`); equations and nominal values unchanged.

WHAT IS DIFFERENT FROM `models/goldbeter.py`, AND WHY IT MATTERS
    The 2003 base model closes its positive loop by having nuclear BMAL1 repress its OWN
    transcript directly:

        dMB/dt = vsB * KIB^m / (KIB^m + BN^m) - ...            (equation 3)

    That is a lumped stand-in for a repression that biology routes through REV-ERBalpha. This
    variant spells the route out -- BMAL1 activates Rev-erb, and NUCLEAR REV-ERB represses
    Bmal1:

        dMR/dt = vsR * BN^h / (KAR^h + BN^h) - vmR*MR/(KmR+MR) - kdmr*MR   (17)
        dRC/dt = ksR*MR - k9*RC + k10*RN - vdRC*RC/(Kd+RC) - kdn*RC        (18)
        dRN/dt = k9*RC - k10*RN - vdRN*RN/(Kd+RN) - kdn*RN                 (19)
        dMB/dt = vsB * KIB^m / (KIB^m + RN^m) - ...                        (3', NOT 3)

    So the positive arm gains two extra delays (transcription + transport) between BMAL1 and
    its own repression. That is exactly the kind of change that should lengthen the nonlinear
    transient the base model struggles with -- Goldbeter/MP needed ~40 periods to relax after
    a large kick where its Floquet estimate said 5 (analysis/relax.py) -- so whether this
    variant is better behaved is a question worth asking of the same measurements.

    It also gives Goldbeter a REV-ERB species for the first time, making RevErb a three-model
    gene in the cross-model comparison instead of two.

NOMINAL PARAMETERS ARE NOT THE BASE MODEL'S
    They come from the paper's Fig. S8 legend and differ substantially -- vsP 2.4 vs 1.5, V1P
    9.6 vs 0.4, Kp 1.006 vs 0.1, n = 2 vs 4. This is a different parameterisation of a
    different model, NOT the base model plus three equations, and its features must not be
    read as "what adding REV-ERB does" to the 2003 numbers.

THE SAME FOUR DEAD PARAMETERS, DROPPED THE SAME WAY
    `V1PCP`, `V1BP`, `V2PCP`, `V2BP` are in the notebook's dict and in none of the 19
    equations -- the phospho-form transitions use V1PC/V2PC/V3PC/V4PC and V1B/V2B/V3B/V4B.
    Verified numerically here as it was for the base model (`--selftest` perturbs each and
    checks the RHS is bit-identical). Carrying them would add four guaranteed-flat directions
    to every identifiability count, inflating the quantity this project measures.
"""
import numpy as np

from models.base import ClockModel, JaxDictParams, register_model

_STATES = ['MP', 'MC', 'MB', 'PC', 'CC', 'PCP', 'CCP', 'PCC', 'PCN', 'PCCP', 'PCNP',
           'BC', 'BCP', 'BN', 'BNP', 'IN', 'MR', 'RC', 'RN']

#: The 15 protein states, all forced into one concentration unit. RC and RN join the group
#: because they are degraded through the SHARED threshold `Kd`, exactly as the phospho-forms
#: and IN are -- one parameter used by several species forces those species into one unit.
_PROTEINS = ['PC', 'CC', 'PCP', 'CCP', 'PCC', 'PCN', 'PCCP', 'PCNP',
             'BC', 'BCP', 'BN', 'BNP', 'IN', 'RC', 'RN']

#: The four notebook entries that appear in no equation. Named so the self-test can prove it.
_DEAD_IN_NOTEBOOK = ('V1PCP', 'V1BP', 'V2PCP', 'V2BP')

_PARAMS = {
    # transcription (vsR/KAR/h are new: BMAL1 -> Rev-erb)
    'vsP': 2.4, 'vsC': 2.2, 'vsB': 1.8, 'vsR': 1.6,
    'KAP': 0.6, 'KAC': 0.6, 'KAR': 0.6, 'KIB': 2.2,
    'n': 2.0, 'm': 2.0, 'h': 2.0,
    # mRNA degradation (Michaelian + first order)
    'vmP': 2.2, 'vmC': 2.0, 'vmB': 1.3, 'vmR': 1.6,
    'KmP': 0.3, 'KmC': 0.4, 'KmB': 0.4, 'KmR': 0.4,
    'kdmp': 0.02, 'kdmc': 0.02, 'kdmb': 0.02, 'kdmr': 0.02,
    # translation
    'ksP': 1.2, 'ksC': 3.2, 'ksB': 0.32, 'ksR': 1.7,
    # (de)phosphorylation maxima and shared thresholds
    'V1P': 9.6, 'V1C': 1.2, 'V1PC': 2.4, 'V1B': 1.4,
    'V2P': 0.6, 'V2C': 0.2, 'V2PC': 0.2, 'V2B': 0.2,
    'V3PC': 2.4, 'V4PC': 0.2, 'V3B': 1.4, 'V4B': 0.4,
    'Kp': 1.006, 'Kdp': 0.1, 'Kd': 0.3,
    # Michaelian degradation (vdRC/vdRN are new, and act on the UNphosphorylated REV forms)
    'vdPC': 3.4, 'vdCC': 1.4, 'vdPCC': 1.4, 'vdPCN': 1.4,
    'vdBC': 3.0, 'vdBN': 3.0, 'vdIN': 1.6, 'vdRC': 4.4, 'vdRN': 0.8,
    # transport / complexation (k9/k10 are new: REV-ERB nuclear transport)
    'k1': 0.8, 'k2': 0.4, 'k3': 0.8, 'k4': 0.4, 'k5': 0.8, 'k6': 0.4,
    'k7': 1.0, 'k8': 0.2, 'k9': 0.8, 'k10': 0.4,
    # non-specific protein degradation
    'kdn': 0.02, 'kdnc': 0.02,
}

#: A point ON the limit cycle: the MP maximum, from a 5000 h LSODA relaxation at rtol
#: 1e-12. The notebook's own initial state is a rough guess, not on the cycle.
_IC = {
    'MP': 4.645269,
    'MC': 5.366686,
    'MB': 10.219064,
    'PC': 0.022140,
    'CC': 420.190706,
    'PCP': 0.012749,
    'CCP': 0.763296,
    'PCC': 5.018497,
    'PCN': 1.531897,
    'PCCP': 11.816179,
    'PCNP': 1.025619,
    'BC': 3.948415,
    'BCP': 0.149061,
    'BN': 1.797764,
    'BNP': 0.092807,
    'IN': 4.185963,
    'MR': 2.915905,
    'RC': 2.244957,
    'RN': 2.118227,
}


class GoldbeterRevModel(JaxDictParams, ClockModel):
    """Leloup & Goldbeter 2003 with explicit REV-ERBalpha (19 ODEs, 63 parameters)."""

    #: MP, as in the base model: largest relative amplitude and unimodal. Checked, not assumed
    #: -- see `--selftest`, which reports the amplitude and modality of every state.
    reference_variable = 'MP'
    approx_period = 28.275

    #: MEASURED, not guessed. At the solver's default 1024 steps this model's flow closes
    #: the loop only to 2.4e-04 and Newton diverges to NaN; at 4096 it closes to 3.2e-07
    #: and converges to |F| = 1.1e-13 with T = 28.2749, matching an LSODA relaxation at
    #: rtol 1e-12. The cause is the 4.5-decade spread inside the state vector (CC ~ 4.2e2
    #: against PCP ~ 1.3e-2), which is what a fixed-step method is worst at.
    orbit_steps = 4096

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
            raise KeyError(f"GoldbeterRevModel has no state {name!r}") from None

    @staticmethod
    def _rhs_terms(p, y, pw):
        """The 19 derivatives, written once and shared by numpy and JAX so they cannot drift.
        `pw(x, k)` is x**k for the backend in use; `y` is the (clamped) state, indexable."""
        (MP, MC, MB, PC, CC, PCP, CCP, PCC, PCN, PCCP, PCNP,
         BC, BCP, BN, BNP, IN, MR, RC, RN) = (y[i] for i in range(19))
        Kp, Kdp, Kd = p['Kp'], p['Kdp'], p['Kd']
        mm = lambda V, x, K: V * x / (K + x)             # Michaelis-Menten

        # --- transcription ------------------------------------------------- #
        BNn, BNh = pw(BN, p['n']), pw(BN, p['h'])
        dMP = p['vsP'] * BNn / (pw(p['KAP'], p['n']) + BNn) - mm(p['vmP'], MP, p['KmP']) \
            - p['kdmp'] * MP
        dMC = p['vsC'] * BNn / (pw(p['KAC'], p['n']) + BNn) - mm(p['vmC'], MC, p['KmC']) \
            - p['kdmc'] * MC
        # EQUATION 3', THE ONE STRUCTURAL CHANGE: Bmal1 is repressed by nuclear REV-ERB (RN),
        # not by nuclear BMAL1 itself as in the base model's equation 3.
        KIBm = pw(p['KIB'], p['m'])
        dMB = p['vsB'] * KIBm / (KIBm + pw(RN, p['m'])) - mm(p['vmB'], MB, p['KmB']) \
            - p['kdmb'] * MB
        dMR = p['vsR'] * BNh / (pw(p['KAR'], p['h']) + BNh) - mm(p['vmR'], MR, p['KmR']) \
            - p['kdmr'] * MR

        # --- PER / CRY in the cytosol --------------------------------------- #
        complexation = p['k3'] * PC * CC - p['k4'] * PCC      # PC + CC <-> PCC
        dPC = (p['ksP'] * MP - mm(p['V1P'], PC, Kp) + mm(p['V2P'], PCP, Kdp)
               - complexation - p['kdn'] * PC)
        dCC = (p['ksC'] * MC - mm(p['V1C'], CC, Kp) + mm(p['V2C'], CCP, Kdp)
               - complexation - p['kdnc'] * CC)
        dPCP = (mm(p['V1P'], PC, Kp) - mm(p['V2P'], PCP, Kdp)
                - mm(p['vdPC'], PCP, Kd) - p['kdn'] * PCP)
        dCCP = (mm(p['V1C'], CC, Kp) - mm(p['V2C'], CCP, Kdp)
                - mm(p['vdCC'], CCP, Kd) - p['kdn'] * CCP)

        # --- PER-CRY complex, cytosol and nucleus ---------------------------- #
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

        # --- BMAL1, cytosol and nucleus -------------------------------------- #
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

        # --- REV-ERB, cytosol and nucleus (new) ------------------------------ #
        # NB the Michaelian degradation acts on the UNPHOSPHORYLATED forms here -- this model
        # has no phospho-REV -- so vdRC/vdRN sit where vdPC/vdBC sit for the other arms.
        r_transport = p['k9'] * RC - p['k10'] * RN            # RC <-> RN
        dRC = (p['ksR'] * MR - r_transport - mm(p['vdRC'], RC, Kd) - p['kdn'] * RC)
        dRN = (r_transport - mm(p['vdRN'], RN, Kd) - p['kdn'] * RN)

        return [dMP, dMC, dMB, dPC, dCC, dPCP, dCCP, dPCC, dPCN, dPCCP, dPCNP,
                dBC, dBCP, dBN, dBNP, dIN, dMR, dRC, dRN]

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
        return jnp.stack(GoldbeterRevModel._rhs_terms(P, yy, pw))

    # --- DIMENSIONS -------------------------------------------------------- #
    def scale_coupled_species(self):
        """All 15 protein states share one unit; the 4 mRNAs scale freely.

        Same chain as the base model, plus RC/RN: they are tied to each other by k9/k10 and to
        the rest of the protein group by the SHARED Michaelian threshold `Kd`, which also
        appears in the degradation of PCP, CCP, PCCP, PCNP, BCP, BNP and IN. A threshold used
        by several species forces those species into one unit.

        Predicted gauge: 4 mRNA scales + 1 protein scale + time = 6 generators of 63.
        """
        return [_PROTEINS]

    def parameter_dimensions(self):
        """Thresholds carry the units of what they SENSE; maximal rates the units of what they
        act on; first-order rates are unit-blind; second-order rates carry 1/[partner]."""
        from models.api import TIME
        PROT = 'PC'          # any protein-group representative: they share one gauge column
        d = {
            'vsP': {'MP': 1, TIME: 1}, 'vsC': {'MC': 1, TIME: 1},
            'vsB': {'MB': 1, TIME: 1}, 'vsR': {'MR': 1, TIME: 1},
            # activation thresholds sense BN; the Bmal1 repression threshold now senses RN
            'KAP': {'BN': 1}, 'KAC': {'BN': 1}, 'KAR': {'BN': 1}, 'KIB': {'RN': 1},
            'n': {}, 'm': {}, 'h': {},
            'vmP': {'MP': 1, TIME: 1}, 'vmC': {'MC': 1, TIME: 1},
            'vmB': {'MB': 1, TIME: 1}, 'vmR': {'MR': 1, TIME: 1},
            'KmP': {'MP': 1}, 'KmC': {'MC': 1}, 'KmB': {'MB': 1}, 'KmR': {'MR': 1},
            'kdmp': {TIME: 1}, 'kdmc': {TIME: 1}, 'kdmb': {TIME: 1}, 'kdmr': {TIME: 1},
            # translation converts mRNA units into protein units
            'ksP': {PROT: 1, 'MP': -1, TIME: 1},
            'ksC': {PROT: 1, 'MC': -1, TIME: 1},
            'ksB': {PROT: 1, 'MB': -1, TIME: 1},
            'ksR': {PROT: 1, 'MR': -1, TIME: 1},
            'Kp': {PROT: 1}, 'Kdp': {PROT: 1}, 'Kd': {PROT: 1},
            'k1': {TIME: 1}, 'k2': {TIME: 1}, 'k4': {TIME: 1}, 'k5': {TIME: 1},
            'k6': {TIME: 1}, 'k8': {TIME: 1}, 'k9': {TIME: 1}, 'k10': {TIME: 1},
            'kdn': {TIME: 1}, 'kdnc': {TIME: 1},
            'k3': {PROT: -1, TIME: 1},           # k3 * PC * CC  ->  [PC]/time
            'k7': {PROT: -1, TIME: 1},           # k7 * BN * PCN ->  [BN]/time
        }
        for name in _PARAMS:                     # every remaining V*/vd* is a Michaelian max
            if name not in d:
                d[name] = {PROT: 1, TIME: 1}
        return d

    # --- PERTURBATION / OBSERVABLES ---------------------------------------- #
    def perturbable_targets(self):
        """The four transcripts and the unphosphorylated, non-sequestered protein forms. The
        phospho-forms and the inactive complex IN are products of internal reactions, not
        things an experiment expresses."""
        return ['MP', 'MC', 'MB', 'MR', 'PC', 'CC', 'BC', 'BN', 'PCC', 'PCN', 'RC', 'RN']

    def observable_states(self):
        """The four transcripts -- what a luciferase reporter or RNA time course measures."""
        return ['MP', 'MC', 'MB', 'MR']

    def observable_pairs(self):
        """(mRNA, its primary protein product), for protein-to-mRNA lag observables."""
        return [('MP', 'PC'), ('MC', 'CC'), ('MB', 'BC'), ('MR', 'RC')]


register_model('goldbeter_rev', GoldbeterRevModel)
