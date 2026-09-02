"""Goldbeter cross-check, each backend forced. Is 2.29e-3 from the Tsit5 switch or older?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine.ptc as PTC, engine.orbit as ORB
from engine.reference import cross_check
orig = PTC.make_ptc
for bk in ('rk4', 'diffrax'):
    ORB.ORBIT_BACKEND = bk
    PTC.make_ptc = (lambda *a, _b=bk, **k: orig(*a, **{**k, 'backend': _b}))
    print(f"########## backend = {bk} ##########", flush=True)
    try:
        cross_check('goldbeter', target='MP', mode='pulse')
    except Exception as e:
        print('  raised', type(e).__name__, e, flush=True)
PTC.make_ptc = orig
print("ABDONE", flush=True)
