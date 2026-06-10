"""
Scheme comparison at the competition bit budget (40,960 bits total per image).

Three schemes:
  1. LBT  N=4, s=sqrt(2), M=16
  2. DCT  N=8, uniform step
  3. DWT  4-level, uniform step

Custom Huffman table header cost: (16 + 162) * 8 = 1,424 bits.
VLC target (with 200-bit safety margin): 39,336 bits.

Run:
    cd /Users/tomjackson/PycharmProjects/cued_sf2_lab
    .venv/bin/python scheme-comparison.py

Output:
    comparison_all.png  — full grid: every image × every scheme
    comparison_summary_table.txt — numeric results
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import (
    jpeg_encode_lbt, jpeg_decode_lbt,
    jpeg_encode_dct, jpeg_decode_dct,
    jpeg_encode_dwt,
    rms_error, compress_to_bits, load_image,
)
import scheme_4

# ---------------------------------------------------------------------------
# Competition constants
# ---------------------------------------------------------------------------
COMPETITION_BITS = 40_960
HUFF_HEADER_BITS = (16 + 162) * 8          # 1,424 bits
VLC_BUDGET       = COMPETITION_BITS - HUFF_HEADER_BITS - 200  # 39,336 bits

# dcbits=11: zero-centred images have DC coefficients up to ±1024,
# which fits in dcbits=11 (range ±1024) but not dcbits=10 (range ±512).
_DCBITS = 11

# ---------------------------------------------------------------------------
# Scheme encode wrappers — each returns {'bits', 'Z', 'rms'}
# 'bits' = VLC bits only; total = bits + HUFF_HEADER_BITS
# ---------------------------------------------------------------------------

def scheme1_encode(X: np.ndarray, step: float) -> dict:
    """LBT N=4, s=√2, M=16."""
    N, s, M = 4, np.sqrt(2), 16
    enc = jpeg_encode_lbt(X, step, N=N, s=s, M=M, opthuff=True, dcbits=_DCBITS)
    Z   = jpeg_decode_lbt(enc['vlc'], step, N=N, s=s, M=M,
                          hufftab=enc['hufftab'],
                          W=X.shape[1], H=X.shape[0], dcbits=_DCBITS)
    return {'bits': enc['bits'], 'Z': Z, 'rms': rms_error(X, Z)}


def scheme2_encode(X: np.ndarray, step: float) -> dict:
    """DCT N=8, uniform quantisation."""
    N = 8
    enc = jpeg_encode_dct(X, step, N=N, M=N, opthuff=True, dcbits=_DCBITS)
    Z   = jpeg_decode_dct(enc['vlc'], step, N=N, M=N,
                          hufftab=enc['hufftab'],
                          W=X.shape[1], H=X.shape[0], dcbits=_DCBITS)
    return {'bits': enc['bits'], 'Z': Z, 'rms': rms_error(X, Z)}


def scheme3_encode(X: np.ndarray, step: float) -> dict:
    """DWT 4-level, uniform per-band step."""
    enc = jpeg_encode_dwt(X, n=4, qstep_base=step, opthuff=True, dcbits=_DCBITS)
    return {'bits': enc['bits'], 'Z': enc['Z'], 'rms': enc['rms']}


def scheme4_encode(X: np.ndarray, q: float) -> dict:
    """DCT N=8 with JPEG standard luma quantisation table scaled by q."""
    return scheme_4.encode(X, q)

# Tighter budget for scheme4: tol_frac=0.02 allows ≈800-bit overshoot
_VLC_BUDGET_S4 = VLC_BUDGET - 800


SCHEMES = {
    'LBT N=4 s=√2':    scheme1_encode,
    'DCT N=8 uniform':  scheme2_encode,
    'DWT 4-level':      scheme3_encode,
    'DCT JPEG-luma':    scheme4_encode,
}

# ---------------------------------------------------------------------------
# Load all images
# ---------------------------------------------------------------------------

def try_load(path: str):
    try:
        return load_image(path)
    except Exception:
        return None


IMAGES = {}  # ordered: standard images first, then past competition images

for name in ('lighthouse', 'bridge', 'flamingo'):
    img = try_load(f'{name}.mat')
    if img is not None:
        IMAGES[name] = img

for year in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
    for fname in (
        f'SF2_competition_image_{year}.mat',
        f'SF2_Competition_Image{year}.mat',
    ):
        img = try_load(fname)
        if img is not None:
            IMAGES[f'{year}'] = img
            break

print(f'Loaded {len(IMAGES)} images: {list(IMAGES.keys())}')

# ---------------------------------------------------------------------------
# Compress every image with every scheme
# ---------------------------------------------------------------------------

print('\nCompressing...')
results = {}   # results[img_name][scheme_name] = {'bits', 'total', 'Z', 'rms', 'ok'}

for img_name, X in IMAGES.items():
    results[img_name] = {}
    for scheme_name, enc_fn in SCHEMES.items():
        sys.stdout.write(f'  {img_name:<12} {scheme_name:<18} ')
        sys.stdout.flush()
        try:
            budget = _VLC_BUDGET_S4 if scheme_name == 'DCT JPEG-luma' else VLC_BUDGET
            enc   = compress_to_bits(X, enc_fn, budget, step_bounds=(1.0, 500.0))
            total = enc['bits'] + HUFF_HEADER_BITS
            ok    = total <= COMPETITION_BITS
            results[img_name][scheme_name] = {
                'bits':  enc['bits'],
                'total': total,
                'Z':     enc['Z'],
                'rms':   enc['rms'],
                'ok':    ok,
            }
            print(f"{total:>6} bits  RMS={enc['rms']:.3f}  {'✓' if ok else '✗ OVER'}")
        except Exception as e:
            results[img_name][scheme_name] = None
            print(f'ERROR: {e}')

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

SCHEME_NAMES = list(SCHEMES.keys())

def print_table():
    col = 22
    header = f"{'Image':<12}" + ''.join(f"  {s:>{col}}" for s in SCHEME_NAMES)
    print('\n' + '=' * len(header))
    print('  RESULTS  (total bits / RMS / budget)')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for img_name, img_results in results.items():
        row = f'{img_name:<12}'
        for s in SCHEME_NAMES:
            r = img_results.get(s)
            if r is None:
                row += f"  {'ERROR':>{col}}"
            else:
                flag = '✓' if r['ok'] else '✗'
                cell = f"{r['total']}b / {r['rms']:.2f} / {flag}"
                row += f"  {cell:>{col}}"
        print(row)
    print('=' * len(header))

print_table()

# Save table to file too
with open('comparison_summary_table.txt', 'w') as f:
    orig_stdout = sys.stdout
    sys.stdout = f
    print_table()
    sys.stdout = orig_stdout
print('\nSaved comparison_summary_table.txt')

# ---------------------------------------------------------------------------
# Big grid figure: rows = images, cols = Original + each scheme
# ---------------------------------------------------------------------------

N_IMAGES  = len(IMAGES)
N_SCHEMES = len(SCHEMES)
N_COLS    = N_SCHEMES + 1   # original + schemes

CELL_W = 2.4   # inches per cell
CELL_H = 3.0   # inches per cell (image + label room)

fig = plt.figure(figsize=(CELL_W * N_COLS, CELL_H * N_IMAGES), dpi=100)

# Column headers (top row annotation only — drawn as text above first image row)
col_titles = ['Original'] + list(SCHEMES.keys())

image_items = list(IMAGES.items())

for row_idx, (img_name, X) in enumerate(image_items):
    vmin, vmax = X.min(), X.max()

    for col_idx in range(N_COLS):
        ax = fig.add_subplot(N_IMAGES, N_COLS, row_idx * N_COLS + col_idx + 1)

        if col_idx == 0:
            # Original image
            ax.imshow(X, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
            if row_idx == 0:
                ax.set_title('Original', fontsize=9, fontweight='bold', pad=4)
            ax.set_ylabel(img_name, fontsize=8, rotation=0, ha='right',
                          va='center', labelpad=4)
        else:
            scheme_name = SCHEME_NAMES[col_idx - 1]
            r = results[img_name].get(scheme_name)

            if r is None or 'Z' not in r:
                ax.text(0.5, 0.5, 'ERROR', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='red')
                ax.set_facecolor('#222')
            else:
                ax.imshow(r['Z'], cmap='gray', vmin=vmin, vmax=vmax,
                          interpolation='nearest')
                flag   = '✓' if r['ok'] else '✗'
                colour = 'green' if r['ok'] else 'red'
                label  = f"{r['total']:,} bits {flag}\nRMS={r['rms']:.2f}"
                ax.set_title(label, fontsize=7.5, color=colour, pad=3)

            if row_idx == 0:
                # Add scheme name above the title on the first row
                ax.text(0.5, 1.18, scheme_name, ha='center', va='bottom',
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        color='black')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

fig.suptitle(
    f'Compression scheme comparison — budget {COMPETITION_BITS:,} bits/image  '
    f'(VLC + {HUFF_HEADER_BITS}-bit Huffman header)',
    fontsize=11, fontweight='bold', y=1.002,
)

plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.3)

out = 'comparison_all.png'
fig.savefig(out, dpi=100, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}  ({os.path.getsize(out) // 1024} KB)')

# Open automatically on macOS
subprocess.run(['open', out], check=False)
print('Done.')
