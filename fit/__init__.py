"""
fit/
====
Fitting a clock model to a PTC target.

    fit/target.py   the analytic radial-isochron (Poincare) PTC, with its two nuisance
                    parameters profiled out at zero model cost
    fit/cost.py     the objective, built so a dead oscillator is the WORST point in the space
    fit/search.py   L-BFGS (primary) and CMA-ES (cross-check) over the gauge quotient
    fit/recover.py  the self-recovery control: refit a KNOWN displaced parameter set

THE ONE THING TO KNOW BEFORE READING ANY OF IT
    input_screen ran this experiment on Mirsky (radialize.py) and it CONVERGED TO A DEGENERATE
    OPTIMUM: cost 0.89 -> 0.159 in 11 L-BFGS iterations, achieved by annihilating the phase
    singularity and collapsing the limit-cycle amplitude -- the optimizer walked the clock to a
    Hopf bifurcation, and the twist "improved" because the oscillation was dying. Parameters
    moved less than 4%. Both of that cost's terms (twist span, singularity position) are
    MINIMIZED by a dying oscillator and nothing forbade it.

    Every design choice in fit/cost.py is aimed at that failure. The load-bearing one is that
    an unusable cell scores the MAXIMUM, not zero and not "excluded" -- see cost.py.
"""
JAX_X64 = True
import os
os.environ.setdefault('JAX_ENABLE_X64', '1')
