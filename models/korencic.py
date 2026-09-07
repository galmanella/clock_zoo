"""
models/korencic.py
==================
Korencic et al. liver-clock model, in the parameterisation of Grabe et al. 15 ODEs, 34
parameters, period ~25.81 h. Ported from `circadian_modeling.ipynb` cell 9; equations and the
nominal parameter set are unchanged, INCLUDING the notebook's note that the published Grabe
values gave damped oscillations and these were adjusted to sustain them.

STRUCTURE
    Five genes -- Bmal, Reverb, Per, Cry, Dbp -- each represented by a three-stage linear
    chain x -> y -> z:

        d(gene_x)/dt = TRANSCRIPTION(z-variables) - deg_g * gene_x
        d(gene_y)/dt = gene_x - d_g * gene_y
        d(gene_z)/dt = gene_y - d_g * gene_z

    so x is the transcript and the y -> z chain is a distributed delay standing in for
    processing, translation and nuclear entry; the z variable is what regulates. Transcription
    is a product of activation and repression factors in the Grabe form

        activation by A:   ((actn * A/act + 1) / (A/act + 1))^h     -> 1 .. actn
        repression by R:   (1 / (R/inh + 1))^h                      -> 1 .. 0

    Regulator index: 1 = Bmalz (activator), 2 = Reverbz (repressor), 3 = Perz (repressor),
    4 = Cryz (repressor), 5 = Dbpz (activator). So `act13` is Bmal's activation threshold on
    gene 3 (Per), `inh42` is Cry's repression threshold on gene 2 (Reverb), and so on.

WHY IT IS INTERESTING HERE
    No complexes, no mass action, and no protein species at all -- structurally about as far
    from Almeida as a mammalian clock model gets, which is exactly what makes it a useful
    second data point for "does PTC data add identifiability".

    And it has ESSENTIALLY NO GAUGE FREEDOM: transcription is a bare product of Hill factors
    with no maximal-rate parameter in front, so there is nothing to absorb a rescaling of the
    transcript. See `scale_constraints` -- all 15 species scales are pinned to the single time
    rescale, leaving 1 generator for 34 parameters (against Almeida's 2 of 18 and Mirsky's 13
    of 132). Nearly every parameter is therefore physically meaningful, and the identifiability
    question here is uncontaminated by units.
"""
import numpy as np

from models.base import ClockModel, JaxDictParams, register_model

_GENES = ['Bmal', 'Reverb', 'Per', 'Cry', 'Dbp']
_STATES = [f'{g}{s}' for g in _GENES for s in 'xyz']

_PARAMS = {
    # gene 1: Bmal   -- repressed by Reverbz only
    'inh21': 4.953866920890614, 'deg1': 0.35141429548963354, 'dBmal': 0.7105134024245657,
    # gene 2: Reverb -- activated by Bmalz and Dbpz, repressed by Perz and Cryz
    'actn12': 4.34, 'act12': 1.3, 'inh32': 1.6873888615143173,
    'actn52': 2.55, 'act52': 0.15, 'inh42': 97.47629562264864,
    'deg2': 0.6714864481495963, 'dReverb': 0.5406033626744334,
    # gene 3: Per
    'actn13': 3.58, 'act13': 5.2, 'inh33': 64.66,
    'actn53': 12.74, 'act53': 0.09, 'inh43': 0.32,
    'deg3': 0.5158681205917364, 'dPer': 0.7559991611804646,
    # gene 4: Cry    -- also repressed by Reverbz
    'inh24': 1.115416629241297, 'actn14': 1.44, 'act14': 0.05,
    'inh34': 10.204093173809058, 'actn54': 28.71, 'act54': 0.72,
    'inh44': 0.8139563979637088, 'deg4': 0.2496053693529315, 'dCry': 0.3920159126239804,
    # gene 5: Dbp    -- activated by Bmalz, repressed by Perz and Cryz (no Dbp autoregulation)
    'actn15': 13.65, 'act15': 0.01, 'inh35': 0.9847679588575912,
    'inh45': 1.4748602547120395, 'deg5': 0.6878005635277191, 'dDbp': 4.579060824806708,
}

#: A point ON the limit cycle: the Reverbx maximum, from a 4000 h LSODA relaxation at rtol
#: 1e-12. Used only as the orbit solver's initial guess.
_IC = {
    'Bmalx': 0.472465, 'Bmaly': 0.922946, 'Bmalz': 1.642013,
    'Reverbx': 7.635765, 'Reverby': 11.842099, 'Reverbz': 16.348459,
    'Perx': 1.152033, 'Pery': 1.444430, 'Perz': 1.894837,
    'Cryx': 0.128898, 'Cryy': 0.523038, 'Cryz': 1.719827,
    'Dbpx': 11.609095, 'Dbpy': 2.446749, 'Dbpz': 0.515076,
}


class KorencicModel(JaxDictParams, ClockModel):
    """Korencic/Grabe 5-gene delay-chain clock (15 ODEs, 34 parameters)."""

    #: SECTION. Reverbx: the largest relative amplitude (2.44 at base, 2.35 median over the
    #: search box) and cleanly unimodal -- and, unlike Almeida's BMAL1, it stays that way under
    #: displacement. `analysis.observable` (49 draws at |v| = 0.25..3.0, 25 with a usable orbit)
    #: finds ALL FIFTEEN species unimodal at 100% of live points, so the section is essentially
    #: unconstrained here and amplitude is the only tie-break. Korencic's whole state vector
    #: spans 2.7 decades (median; 5.1 worst) against the THIRTY that made Almeida's RAD01
    #: optimum stiff enough to stall Newton -- this model is far better conditioned.
    reference_variable = 'Reverbx'

    #: READOUT, AND IT IS DELIBERATELY NOT THE SECTION -- the one place Korencic does need the
    #: split (REPO_MAP hazard 14). Two quantities decide it and they pull in opposite
    #: directions, so both were measured rather than argued:
    #:
    #:   base_ratio = min/|mean| over the cycle is the COLLAPSE signature -- what reached
    #:       1e-27 on Almeida and what `fit/viability` rejects an entire parameter set for
    #:       below 1e-3. It is measured at DISPLACED parameter sets, because that is where a
    #:       fit goes. Reverbx is the WORST of all fifteen species here.
    #:   the two-engine cross-check (`engine.reference.cross_check`) is measured at BASE and
    #:       rewards a LARGE relative amplitude, because the reference engine matches peaks
    #:       and a shallow oscillation gives it a noisier peak. Reverbx is the best here.
    #:
    #: Picking on either alone gives a different answer. Measured, korencic/Bmalx/pulse:
    #:
    #:     readout    cross-check (cyc)   base_ratio    median rel_amp
    #:     Reverbx           6.56e-04      1.92e-03           2.350   <- section, worst baseline
    #:     Dbpx              5.35e-04      8.02e-02           1.410   <- best on BOTH counts
    #:     Perx              9.88e-04      2.14e-01           1.081
    #:     Bmalx             1.11e-03      3.35e-02           1.771
    #:     Cryx              1.26e-03      2.69e-02           1.661
    #:
    #: Dbpx wins outright: the lowest cross-check error of any species -- better than Reverbx's
    #: -- with a baseline 42x safer. There is no trade-off left to make.
    #:
    #: Three things make it the natural choice beyond the numbers. It is a TRANSCRIPT, so it is
    #: in `observable_states()` and at the mRNA level `analysis/genemap` scopes Korencic to.
    #: Dbp-luciferase is the standard circadian reporter, so this is what the experiment
    #: actually watches. And Dbpx is Korencic's sole NON-RESETTER (no S_crit out to dose 1e4),
    #: so it is excluded from perturbation work anyway -- the readout therefore shares no role
    #: with any target, which is the overlap that started hazard 14 in the first place.
    #:
    #: THE SWITCH IS FREE, AND THAT IS CHECKED RATHER THAN ASSERTED: asymptotic phase is a
    #: property of the state, so candidates must agree up to the calibrated origin. Measured on
    #: Bmalx over 12 phases x 4 doses, worst |d new_phase| after removing the constant offset is
    #: 1.0e-03 cyc (instant) and 1.4e-03 (pulse) across the candidates -- ~1.5 s of a 25.81 h
    #: period.
    readout_variable = 'Dbpx'

    approx_period = 25.81

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
            raise KeyError(f"KorencicModel has no state {name!r}") from None

    @staticmethod
    def _terms(p, Bz, Rz, Pz, Cz, Dz, act, inh):
        """The five transcription factors, shared by numpy and JAX so the two cannot drift.
        `act(n, k, x, h)` and `inh(k, x, h)` are supplied by the caller's backend."""
        return dict(
            Bmal=inh(p['inh21'], Rz, 2),
            Reverb=(act(p['actn12'], p['act12'], Bz, 3) * inh(p['inh32'], Pz, 3)
                    * act(p['actn52'], p['act52'], Dz, 1) * inh(p['inh42'], Cz, 3)),
            Per=(act(p['actn13'], p['act13'], Bz, 2) * inh(p['inh33'], Pz, 2)
                 * act(p['actn53'], p['act53'], Dz, 1) * inh(p['inh43'], Cz, 2)),
            Cry=(inh(p['inh24'], Rz, 2) * act(p['actn14'], p['act14'], Bz, 2)
                 * inh(p['inh34'], Pz, 2) * act(p['actn54'], p['act54'], Dz, 1)
                 * inh(p['inh44'], Cz, 2)),
            Dbp=(act(p['actn15'], p['act15'], Bz, 3) * inh(p['inh35'], Pz, 3)
                 * inh(p['inh45'], Cz, 3)),
        )

    def derivatives(self, t, state, params=None):
        p = self.params if params is None else params
        y = np.maximum(np.asarray(state, float), 0.0)
        Bz, Rz, Pz, Cz, Dz = (y[self._idx[f'{g}z']] for g in _GENES)
        act = lambda n, k, x, h: ((n * x / k + 1.0) / (x / k + 1.0)) ** h
        inh = lambda k, x, h: (1.0 / (x / k + 1.0)) ** h
        prod = self._terms(p, Bz, Rz, Pz, Cz, Dz, act, inh)
        d = np.zeros(len(_STATES))
        for gi, g in enumerate(_GENES):
            xi = 3 * gi
            dg = p[f'd{g}']
            d[xi] = prod[g] - p[f'deg{gi + 1}'] * y[xi]
            d[xi + 1] = y[xi] - dg * y[xi + 1]
            d[xi + 2] = y[xi + 1] - dg * y[xi + 2]
        return d

    # --- JAX --------------------------------------------------------------- #
    @staticmethod
    def jax_rhs(y, P):
        import jax.numpy as jnp
        yy = jnp.maximum(y, 0.0)
        Bz, Rz, Pz, Cz, Dz = yy[2], yy[5], yy[8], yy[11], yy[14]
        act = lambda n, k, x, h: ((n * x / k + 1.0) / (x / k + 1.0)) ** h
        inh = lambda k, x, h: (1.0 / (x / k + 1.0)) ** h
        prod = KorencicModel._terms(P, Bz, Rz, Pz, Cz, Dz, act, inh)
        out = []
        for gi, g in enumerate(_GENES):
            xi = 3 * gi
            dg = P[f'd{g}']
            out += [prod[g] - P[f'deg{gi + 1}'] * yy[xi],
                    yy[xi] - dg * yy[xi + 1],
                    yy[xi + 1] - dg * yy[xi + 2]]
        return jnp.stack(out)

    # --- DIMENSIONS -------------------------------------------------------- #
    def scale_coupled_species(self):
        """No two species share a unit here. The chain terms relate the scales without
        equating them ([gene_y] = [gene_x] * time), which is what `scale_constraints`
        expresses."""
        return []

    def scale_constraints(self):
        """Korencic's scales are PINNED, not free -- three constraints per gene.

        Writing the defining identity f(S y; theta_gauged) = rho * S * f(y; theta) term by
        term for gene g with chain (gx, gy, gz):

          d(gx)/dt = HILL(...) - deg*gx    The Hill product is dimensionless and invariant
                                           once thresholds are scaled, so it contributes a
                                           bare 1. Matching it against rho*S_gx*1 forces
                                                S_gx * rho = 1.
          d(gy)/dt = gx - d*gy             Matching the gx term: S_gx = rho * S_gy.
          d(gz)/dt = gy - d*gz             Matching the gy term: S_gy = rho * S_gz.

        Every species scale is therefore determined by rho (S_gx = 1/rho, S_gy = 1/rho^2,
        S_gz = 1/rho^3), leaving exactly ONE generator: the global time rescale. `gauge/` takes
        the null space of these and `gauge/validate.py identity` confirms it numerically -- the
        declaration is checked, not trusted.
        """
        from models.api import TIME
        cons = []
        for g in _GENES:
            cons.append({f'{g}x': 1, TIME: 1})                   # S_gx * rho = 1
            cons.append({f'{g}y': 1, f'{g}x': -1, TIME: 1})      # S_gy * rho = S_gx
            cons.append({f'{g}z': 1, f'{g}y': -1, TIME: 1})      # S_gz * rho = S_gy
        return cons

    def parameter_dimensions(self):
        """Thresholds carry the units of the REGULATOR they sense; `actn` multiplies an
        already-dimensionless ratio and so is dimensionless; degradation and chain-transfer
        rates are first order."""
        from models.api import TIME
        d = {}
        # regulator of each threshold family: 1 = Bmalz, 2 = Reverbz, 3 = Perz, 4 = Cryz,
        # 5 = Dbpz. Key form is <act|actn|inh><regulator><gene>.
        reg = {'1': 'Bmalz', '2': 'Reverbz', '3': 'Perz', '4': 'Cryz', '5': 'Dbpz'}
        for name in _PARAMS:
            if name.startswith('actn'):
                d[name] = {}                                     # dimensionless gain
            elif name.startswith(('act', 'inh')):
                d[name] = {reg[name[3]]: 1}                      # threshold ~ regulator units
            else:                                                # deg1..deg5, dBmal..dDbp
                d[name] = {TIME: 1}
        return d

    # --- PERTURBATION / OBSERVABLES ---------------------------------------- #
    def perturbable_targets(self):
        """The transcripts (x) and the regulator stages (z). An experiment overexpresses a
        gene product, which enters as transcript; pushing z directly is the closer analogue of
        an ectopic protein, since z is what actually regulates. The intermediate y is a
        modelling device with no clean experimental counterpart."""
        return [f'{g}{s}' for g in _GENES for s in ('x', 'z')]

    def observable_states(self):
        """Transcript levels -- what a reporter or RNA-seq time course measures."""
        return [f'{g}x' for g in _GENES]

    def observable_pairs(self):
        return []                    # x/y/z are stages of one gene, not an mRNA/protein pair


register_model('korencic', KorencicModel)
