"""Render the two candidate BMAL1 radializations on FIGURE 1's grid (linear 0-100, 97 rows)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, paths
from fit.rescan import rescan_run
DOSES = np.linspace(0.0, 100.0, 97)          # identical to R01 figure 1
for tag in ('arm0_ctl__seeds-1_start-base', 'arm4_floor__seeds-2_start-base'):
    f = f'out/almeida/fit_radial/{tag}/radial_BMAL1_instant_cma.npz'
    out = rescan_run(f, DOSES, n_phase=32)
    S = out['target_S']
    d, dl = out['doses'], np.abs(out['delta'])
    lo, hi = (d > 0) & (d < S), d >= S
    print(f"    rms|delta| below S* {np.sqrt(np.nanmean(dl[:, lo]**2)):.4f} "
          f"({lo.sum()} rows) | above {np.sqrt(np.nanmean(dl[:, hi]**2)):.4f} ({hi.sum()} rows)",
          flush=True)
    p = paths.out_path('almeida', 'fit_rescan', f'lin_{tag}.npz', 'r01')
    paths.savez(p, **{k: v for k, v in out.items()})
print("RENDERDONE", flush=True)
