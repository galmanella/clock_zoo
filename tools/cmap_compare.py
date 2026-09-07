"""
tools/cmap_compare.py
=====================
Render figure 1 (the Bmal1 cross-model row) under every available CYCLIC colour map, so the
current choice can be compared against alternatives on real data rather than on a swatch.

    $PY -m tools.cmap_compare

A phase map MUST be cyclic -- phase 0 and phase 1 are the same state, and any map whose ends
differ puts a false discontinuity across the wrap. That rules out viridis and friends and
leaves a short list: cmocean's `phase` (what the figures use now) and the five Scientific
colour maps from cmcrameri whose names end in O (bamO, brocO, corkO, romaO, vikO).

Writes to out/zoo/figures/cmaps/, NOT to R01_sup_page -- these are candidates, not panels.

The swap works by rebinding `plotting.PHASE_CMAP`: `plotting.phase_cmap()` reads that global
at call time and returns a fresh copy with the bad-value colour set, so every panel picks the
replacement up without touching panels.py.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plotting                                                       # noqa: E402
from R01_sup_page.panels import BMAL, fig_bmal, CM                    # noqa: E402
from analysis.features import load as load_features                   # noqa: E402

#: The provenance of the published fig1, from its .source.txt -- same data, only the map moves.
TAGS = {'*': 'zoo04lin', 'korencic': 'korph256'}
MODE = 'instant'
OUT = os.path.join('out', 'zoo', 'figures', 'cmaps')


def candidates():
    """[(label, cmap)] -- the current choice first, then every cmcrameri cyclic map."""
    out = [('current_cmocean_phase', plotting.PHASE_CMAP)]
    try:
        from cmcrameri import cm as ccm
    except ImportError:
        print('[cmap] cmcrameri not installed; nothing to compare against', file=sys.stderr)
        return out
    for n in sorted(x for x in dir(ccm)
                    if x.endswith('O') and not x.startswith('_') and not x.endswith('_rO')):
        out.append((f'cmcrameri_{n}', getattr(ccm, n)))
    return out


def sources(mode=MODE, tags=TAGS):
    cache, srcs, note = {}, [], []
    for model, target in BMAL:
        t = tags.get(model, tags.get('*'))
        if t not in cache:
            cache[t] = load_features(mode, t, source='surface')
        rows, surfaces, _z = cache[t]
        if (model, target) not in surfaces:
            print(f'[cmap] {model}/{target} missing from tag {t!r}', file=sys.stderr)
            continue
        r = next((x for x in rows if x['model'] == model and x['target'] == target), None)
        srcs.append((model, surfaces[(model, target)], r['S_surf'] if r else np.nan))
        note.append(f'  {model:10s} {target:7s} tag={t}')
    return srcs, note


def strip(cands, path, width_cm=10.0):
    """One reference strip: each candidate map swept over a full cycle."""
    fig, axes = plt.subplots(len(cands), 1,
                             figsize=(width_cm * CM, 0.42 * len(cands) * CM * 2.2))
    grad = np.linspace(0, 1, 512)[None, :]
    for ax, (lab, cmap) in zip(np.atleast_1d(axes), cands):
        ax.imshow(grad, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(lab.replace('cmcrameri_', ''), fontsize=6, rotation=0,
                      ha='right', va='center', labelpad=4)
    fig.suptitle('cyclic colour maps, 0 -> 1 cycle', fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'[cmap] -> {path}', flush=True)



# --------------------------------------------------------------------------- #
#  Lightening a cyclic map without breaking what makes it cyclic
# --------------------------------------------------------------------------- #
def _srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _lstar(rgb):
    """CIE L* from sRGB. Used only to CHECK that lightening stayed even."""
    lin = _srgb_to_lin(np.asarray(rgb, float))
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(y > 0.008856, np.cbrt(y), (903.3 * y + 16) / 116)
    return 116 * f - 16


def lighten(cmap, amount, n=256):
    """A copy of `cmap` blended toward white by `amount`, in LINEAR light.

    Blending in linear light rather than in sRGB matters: sRGB is gamma-encoded, so mixing
    there lightens the dark part of the cycle more than the light part and tilts a map that was
    built to be iso-luminant. cmocean `phase` carries phase in HUE at nearly constant
    lightness, and a lightening that is uneven across the cycle would add a brightness ramp
    that reads as structure in the data. Blending toward white is also gamut-safe by
    construction, unlike raising L* in Lab at fixed chroma, which clips on the saturated hues
    and clips them unevenly."""
    from matplotlib.colors import ListedColormap
    rgba = cmap(np.linspace(0, 1, n))
    lin = _srgb_to_lin(rgba[:, :3])
    out = rgba.copy()
    out[:, :3] = np.clip(_lin_to_srgb(lin + amount * (1.0 - lin)), 0, 1)
    return ListedColormap(out, name=f'{cmap.name}_light{amount:g}')



_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_MI = np.linalg.inv(_M)
_WP = np.array([0.95047, 1.0, 1.08883])


def _f(t):
    return np.where(t > 0.008856, np.cbrt(np.maximum(t, 0)), (903.3 * t + 16) / 116)


def _finv(t):
    t3 = t ** 3
    return np.where(t3 > 0.008856, t3, (116 * t - 16) / 903.3)


def rgb_to_lab(rgb):
    xyz = _srgb_to_lin(np.asarray(rgb, float)) @ _M.T
    fx, fy, fz = (_f(xyz[:, i] / _WP[i]) for i in range(3))
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=1)


def lab_to_rgb(lab):
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16) / 116
    xyz = np.stack([_WP[0] * _finv(fy + a / 500), _WP[1] * _finv(fy),
                    _WP[2] * _finv(fy - b / 200)], axis=1)
    return _lin_to_srgb(np.clip(xyz @ _MI.T, 0, 1))


def lighten_lab(cmap, dL, n=256):
    """Raise L* by a constant, KEEPING a* and b* -- so hue and chroma are preserved.

    The white-blend lightener desaturates as it lightens, which is what makes it read as
    washed out. Shifting L* at fixed (a*, b*) keeps the colour and only raises the lightness.
    The cost is gamut: saturated hues can leave sRGB and get clipped, and clipping is not
    uniform across the cycle -- so `lighten_report` prints how much was clipped and the caller
    should keep dL small enough that it stays negligible."""
    from matplotlib.colors import ListedColormap
    rgba = cmap(np.linspace(0, 1, n))
    lab = rgb_to_lab(rgba[:, :3])
    lab[:, 0] = np.clip(lab[:, 0] + dL, 0, 100)
    out = rgba.copy()
    out[:, :3] = np.clip(lab_to_rgb(lab), 0, 1)
    return ListedColormap(out, name=f'{cmap.name}_L+{dL:g}')



def recolor(cmap, chroma=1.0, dL=0.0, n=256):
    """Scale chroma by `chroma` and shift L* by `dL`, in Lab. Hue is untouched.

    THE VIVIDNESS DIAL IS CHROMA, NOT LIGHTNESS. cmocean `phase` reads as "psychedelic"
    because it sweeps the full hue circle at high chroma (mean C* ~ 62). Lightening it at
    fixed chroma -- `lighten_lab` -- makes that WORSE, because the same chroma at higher
    lightness reads as more saturated, not less. Scaling a* and b* toward the neutral axis at
    constant L* is the direct fix: same hues, same brightness, less shout.

    chroma=1, dL=0 is the identity."""
    from matplotlib.colors import ListedColormap
    rgba = cmap(np.linspace(0, 1, n))
    lab = rgb_to_lab(rgba[:, :3])
    lab[:, 1:] *= chroma
    lab[:, 0] = np.clip(lab[:, 0] + dL, 0, 100)
    out = rgba.copy()
    out[:, :3] = np.clip(lab_to_rgb(lab), 0, 1)
    tag = f'C{chroma:g}' + (f'_L{dL:+g}' if dL else '')
    return ListedColormap(out, name=f'{cmap.name}_{tag}')


def _chroma(rgb):
    lab = rgb_to_lab(rgb)
    return np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)


def lighten_report(cmap, amounts):
    """Print the L* profile before and after, so 'evenly' is checked, not assumed."""
    base = cmap(np.linspace(0, 1, 256))[:, :3]
    l0 = _lstar(base)
    c0 = _chroma(base)
    print(f"  {'map':22}{'L* mean':>9}{'spread':>8}{'chroma':>9}")
    print(f"  {'phase (original)':22}{l0.mean():9.1f}{l0.max() - l0.min():8.1f}"
          f"{c0.mean():9.1f}")
    for a in amounts:
        rgb = lighten(cmap, a)(np.linspace(0, 1, 256))[:, :3]
        l, c = _lstar(rgb), _chroma(rgb)
        print(f"  {'white-blend ' + format(a, '.2f'):22}{l.mean():9.1f}"
              f"{l.max() - l.min():8.1f}{c.mean():9.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description='render fig1 under every cyclic colour map')
    ap.add_argument('--mode', default=MODE)
    ap.add_argument('--width-cm', type=float, default=10.0)
    ap.add_argument('--height-cm', type=float, default=4.0)
    ap.add_argument('--dpi', type=int, default=600)
    ap.add_argument('--chroma', default=None,
                    help="comma list of chroma scale factors, e.g. '0.75,0.55,0.4'")
    ap.add_argument('--chroma-dl', type=float, default=0.0,
                    help='L* shift applied alongside --chroma')
    ap.add_argument('--lighten-lab', default=None,
                    help="comma list of L* shifts, e.g. '4,7,10' "
                         '(keeps hue and chroma; gentler than a white blend)')
    ap.add_argument('--lighten', default=None,
                    help="comma list of lightening amounts, e.g. '0.25,0.4,0.55'")
    a = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    srcs, note = sources(a.mode)
    if not srcs:
        raise SystemExit('no surfaces loaded')
    if a.chroma:
        ks = [float(x) for x in a.chroma.split(',')]
        base = plotting.PHASE_CMAP
        cands = [('current_cmocean_phase', base)] +                 [(f'phase_C{k:g}'.replace('.', 'p')
                  + (f'_L{a.chroma_dl:+g}'.replace('.', 'p') if a.chroma_dl else ''),
                  recolor(base, k, a.chroma_dl)) for k in ks]
    elif a.lighten_lab:
        ds = [float(x) for x in a.lighten_lab.split(',')]
        base = plotting.PHASE_CMAP
        cands = [('current_cmocean_phase', base)] +                 [(f'phase_Lplus{d:g}'.replace('.', 'p'), lighten_lab(base, d)) for d in ds]
    elif a.lighten:
        amounts = [float(x) for x in a.lighten.split(',')]
        base = plotting.PHASE_CMAP
        print('[cmap] lightness check (phase is near iso-luminant; keep the spread small):')
        lighten_report(base, amounts)
        cands = [('current_cmocean_phase', base)] +                 [(f'phase_lighter_{x:g}'.replace('.', 'p'), lighten(base, x)) for x in amounts]
    else:
        cands = candidates()
    sname = ('chroma_strip.png' if a.chroma else
             'lightenlab_strip.png' if a.lighten_lab else
             'lighten_strip.png' if a.lighten else 'cyclic_maps_strip.png')
    strip(cands, os.path.join(OUT, sname), a.width_cm)

    original = plotting.PHASE_CMAP
    try:
        for lab, cmap in cands:
            plotting.PHASE_CMAP = cmap            # phase_cmap() reads this at call time
            fig = fig_bmal(srcs, a.width_cm, a.height_cm, cbar=True)
            p = os.path.join(OUT, f'fig1_{lab}.png')
            fig.savefig(p, dpi=a.dpi)
            plt.close(fig)
            print(f'[cmap] -> {p}', flush=True)
    finally:
        plotting.PHASE_CMAP = original            # never leave the global swapped

    with open(os.path.join(OUT, 'README.txt'), 'w') as f:
        f.write('fig1 rendered under each cyclic colour map.\n'
                'script  tools/cmap_compare.py\n'
                f'mode    {a.mode}\n'
                'data:\n' + '\n'.join(note) + '\n'
                'These are CANDIDATES for comparison, not published panels;\n'
                'the panel in R01_sup_page/ still uses cmocean phase.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
