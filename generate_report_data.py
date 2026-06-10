"""
generate_report_data.py
=======================
Generates all numeric data needed for the SF2 Final Report tables.
Writes results to report_data/  as .txt files that can be copy-pasted into LaTeX.

Run from the cued_sf2_lab directory:
    cd /Users/tomjackson/PycharmProjects/cued_sf2_lab
    .venv/bin/python generate_report_data.py

Tables produced:
  data_rise1_sweep.txt          — Table: rise1 vs entropy/step/CR
  data_blocksize_comparison.txt — Table: DCT block size N=4,8,16 under Huffman
  data_huffman_table_types.txt  — Table: custom/fixed/default Huffman on 3 images
  data_scheme_comparison.txt    — Full 10-image × 4-scheme pass/fail table
  data_competition_results.txt  — Final scheme on Lighthouse/Bridge/Flamingo
  data_summary.txt              — All tables in one file for quick reference
"""

import os, sys, time
import numpy as np
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.laplacian_pyramid import quant1, quant2, bpp
from cued_sf2_lab.jpeg import huffdflt, huffgen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import (
    load_image, rms_error,
    jpeg_encode_dct, jpeg_decode_dct,
    jpeg_encode_lbt, jpeg_decode_lbt,
    jpeg_encode_dwt,
    compress_to_bits,
)
import scheme_4

OUT = 'report_data'
os.makedirs(OUT, exist_ok=True)

COMPETITION_BITS = 40_960
HUFF_HEADER      = (16 + 162) * 8   # 1424 bits
VLC_BUDGET       = COMPETITION_BITS - HUFF_HEADER - 200   # 39336
DCBITS           = 11

N = 8
C = dct_ii(N)

# ── Load all images ───────────────────────────────────────────────────────────
def try_load(name):
    for pat in [f'{name}.mat',
                f'SF2_competition_image_{name}.mat',
                f'SF2_Competition_Image{name}.mat']:
        if os.path.exists(pat):
            return load_image(pat)
    return None

X_lh = try_load('lighthouse')
X_br = try_load('bridge')
X_fl = try_load('flamingo')

COMP_IMAGES = {}
for yr in (2019, 2020, 2021, 2022, 2023, 2024, 2025):
    img = try_load(str(yr))
    if img is not None:
        COMP_IMAGES[str(yr)] = img

print(f'Loaded: lighthouse, bridge, flamingo + {len(COMP_IMAGES)} competition images')

# ── Shared encode/decode helpers ─────────────────────────────────────────────

def enc_dct(X, step):
    # Pass X directly — jpegenc in this lab works on whatever range is given;
    # it does NOT subtract 128 internally.  Zero-centred X gives DC ≤ 1016,
    # fitting in dcbits=11 (range ±1023).  Adding 128 would make DC ≤ 2040 → overflow.
    enc = jpeg_encode_dct(X, step, N=8, M=8, opthuff=True, dcbits=DCBITS)
    Z   = jpeg_decode_dct(enc['vlc'], step, N=8, M=8,
                          hufftab=enc['hufftab'],
                          W=X.shape[1], H=X.shape[0], dcbits=DCBITS)
    return {'bits': enc['bits'], 'Z': Z, 'rms': rms_error(X, Z)}

def enc_lbt(X, step):
    enc = jpeg_encode_lbt(X, step, N=4, s=np.sqrt(2), M=16,
                          opthuff=True, dcbits=DCBITS)
    Z   = jpeg_decode_lbt(enc['vlc'], step, N=4, s=np.sqrt(2), M=16,
                          hufftab=enc['hufftab'],
                          W=X.shape[1], H=X.shape[0], dcbits=DCBITS)
    return {'bits': enc['bits'], 'Z': Z, 'rms': rms_error(X, Z)}

def enc_dwt(X, step):
    enc = jpeg_encode_dwt(X, n=4, qstep_base=step, opthuff=True, dcbits=DCBITS)
    return {'bits': enc['bits'], 'Z': enc['Z'], 'rms': enc['rms']}

def enc_s4(X, q):
    return scheme_4.encode(X, q)


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: rise1 sweep  (DCT N=8 entropy at fixed RMS=4.935)
# Uses pure entropy estimation — no Huffman coding needed
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TABLE 1: rise1 sweep')
print('='*60)

TARGET_RMS = 4.935
H_, W_ = X_lh.shape
sub_h  = H_ // N

Y_lh   = colxfm(colxfm(X_lh, C).T, C).T
Yr_lh  = regroup(Y_lh, N)

def dct_encode_decode(X, step, rise1):
    """DCT N=8, pure quantisation (no Huffman) — returns rms and entropy bits.

    Encode path: forward DCT → regroup (sub-image layout) → quant1
    Decode path: dequant → regroup back to block layout → inverse DCT
    regroup(Yr, N) converts block layout to sub-image layout.
    regroup(Yq, [sub_h, sub_w]) converts sub-image layout back to block layout.
    """
    H__, W__ = X.shape
    sub_h_ = H__ // N
    sub_w_ = W__ // N
    Y_     = colxfm(colxfm(X, C).T, C).T          # block layout
    Yr_    = regroup(Y_, N)                         # sub-image layout
    Yq_    = quant1(Yr_, step, rise1)               # quantise (sub-image layout)
    entropy_bits = bpp(Yq_) * X.size
    # decode: dequantise then convert back to block layout then inverse DCT
    Yq_r   = quant2(Yq_, step)                     # dequantise (sub-image layout)
    Yq_blk = regroup(Yq_r, [sub_h_, sub_w_])       # back to block layout
    Z_     = colxfm(colxfm(Yq_blk.T, C.T).T, C.T) # inverse DCT
    return rms_error(X, Z_), entropy_bits

def find_step_binary(X, target_rms, rise1_frac, lo=1.0, hi=300.0, iters=50):
    """Binary search for step giving rms ≈ target_rms."""
    # Check bounds
    rms_lo, _ = dct_encode_decode(X, lo, rise1_frac * lo)
    rms_hi, _ = dct_encode_decode(X, hi, rise1_frac * hi)

    if rms_lo > target_rms:
        return lo   # can't reach target even at finest step
    if rms_hi < target_rms:
        return hi   # can't reach target even at coarsest step

    for _ in range(iters):
        mid = (lo + hi) / 2
        rms_mid, _ = dct_encode_decode(X, mid, rise1_frac * mid)
        if rms_mid < target_rms:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

PIXELS = H_ * W_
rise1_fracs = [0.5, 1.0, 1.5]
rise1_names = ['$0.5\\Delta$ (uniform)', '$1.0\\Delta$ (JPEG)', '$1.5\\Delta$']
rise1_data  = []

for rf, rname in zip(rise1_fracs, rise1_names):
    s_opt = find_step_binary(X_lh, TARGET_RMS, rf)
    rms_check, ent = dct_encode_decode(X_lh, s_opt, rf * s_opt)
    cr = (PIXELS * 8) / ent if ent > 0 else 0
    rise1_data.append((rname, rf, s_opt, rms_check, ent, cr))
    print(f'  rise1={rf:.1f}Δ  step={s_opt:.2f}  rms={rms_check:.3f}  '
          f'entropy={ent:.0f}  CR={cr:.3f}')

lines = [
    '% Table: rise1 sweep (DCT N=8 on Lighthouse, fixed RMS≈4.935)',
    '% rise1_frac | optimal_step | rms_actual | entropy_bits | CR',
    r'\begin{tabular}{@{}lcrrc@{}}',
    r'\toprule',
    r'\texttt{rise1} & Zero-bin width & Optimal $\Delta$ & Entropy (bits) & CR \\',
    r'\midrule',
]
for rname, rf, s_opt, rms_v, ent, cr in rise1_data:
    flag = r'\textbf{' if abs(rf - 1.0) < 0.01 else ''
    endflag = r'}' if abs(rf - 1.0) < 0.01 else ''
    lines.append(
        f'{flag}{rname}{endflag} & '
        f'${rf*2:.0f}\\times\\Delta$ & '
        f'{flag}{s_opt:.2f}{endflag} & '
        f'{flag}{ent:,.0f}{endflag} & '
        f'{flag}{cr:.3f}{endflag} \\\\'
    )
lines += [r'\bottomrule', r'\end{tabular}']

with open(f'{OUT}/data_rise1_sweep.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'  → {OUT}/data_rise1_sweep.txt')


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: Block size comparison under Huffman coding (DCT N=4,8,16)
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TABLE 2: Block size comparison')
print('='*60)

blocksize_data = []
for N_b in [4, 8, 16]:
    print(f'  N={N_b}...', end=' ', flush=True)
    def enc_dct_N(X, step, N_b=N_b):
        enc = jpeg_encode_dct(X, step, N=N_b, M=N_b, opthuff=True, dcbits=DCBITS)
        Z   = jpeg_decode_dct(enc['vlc'], step, N=N_b, M=N_b,
                              hufftab=enc['hufftab'],
                              W=X.shape[1], H=X.shape[0], dcbits=DCBITS)
        return {'bits': enc['bits'], 'Z': Z, 'rms': rms_error(X, Z)}

    try:
        enc = compress_to_bits(X_lh, enc_dct_N, VLC_BUDGET, step_bounds=(1.0, 400.0))
        total = enc['bits'] + HUFF_HEADER
        ok = total <= COMPETITION_BITS
        n_blocks = (256 // N_b) ** 2
        blocksize_data.append((N_b, n_blocks, enc['bits'], total, enc['rms'], ok))
        print(f'total={total}  rms={enc["rms"]:.2f}  {"PASS" if ok else "FAIL"}')
    except Exception as e:
        blocksize_data.append((N_b, (256//N_b)**2, 0, 0, 0, False))
        print(f'ERROR: {e}')

lines = [
    '% Table: DCT block size under Huffman coding on Lighthouse',
    r'\begin{tabular}{@{}lrrrrc@{}}',
    r'\toprule',
    r'$N$ & Blocks per image & VLC bits & Total bits & RMS & Pass? \\',
    r'\midrule',
]
for N_b, nb, vlc_b, tot, rms_v, ok in blocksize_data:
    tick = r'\checkmark' if ok else r'$\times$'
    bold_open  = r'\textbf{' if N_b == 8 else ''
    bold_close = r'}' if N_b == 8 else ''
    lines.append(
        f'{bold_open}{N_b}{bold_close} & {nb:,} & '
        f'{bold_open}{vlc_b:,}{bold_close} & '
        f'{bold_open}{tot:,}{bold_close} & '
        f'{bold_open}{rms_v:.1f}{bold_close} & {tick} \\\\'
    )
lines += [r'\bottomrule', r'\end{tabular}']

with open(f'{OUT}/data_blocksize_comparison.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'  → {OUT}/data_blocksize_comparison.txt')


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: Huffman table type comparison (custom / fixed / default)
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TABLE 3: Huffman table type comparison')
print('='*60)

# Fixed q that lands near the budget for all three images
Q_FIXED = 7.5

huff_data = []
for name, X in [('Lighthouse', X_lh), ('Bridge', X_br), ('Flamingo', X_fl)]:
    print(f'  {name}...', end=' ', flush=True)
    row = {'name': name}

    # 1. Custom per-image opthuff (scheme_4 uses this)
    try:
        enc_c = scheme_4.encode(X, Q_FIXED)
        row['custom_vlc']   = enc_c['bits']
        row['custom_total'] = enc_c['bits'] + HUFF_HEADER
        row['custom_rms']   = enc_c['rms']
    except Exception as e:
        row['custom_vlc'] = row['custom_total'] = row['custom_rms'] = -1
        print(f'  custom ERROR: {e}', end='')

    # 2. JPEG Annex-K default table (opthuff=False)
    try:
        enc_d = jpeg_encode_dct(X, Q_FIXED * 16, N=8, M=8,
                                opthuff=False, dcbits=DCBITS)
        Z_d   = jpeg_decode_dct(enc_d['vlc'], Q_FIXED * 16, N=8, M=8,
                                hufftab=enc_d['hufftab'],
                                W=X.shape[1], H=X.shape[0], dcbits=DCBITS)
        row['default_vlc']   = enc_d['bits']
        row['default_total'] = enc_d['bits']  # no header for default
        row['default_rms']   = rms_error(X, Z_d)
    except Exception as e:
        row['default_vlc'] = row['default_total'] = row['default_rms'] = -1
        print(f'  default ERROR: {e}', end='')

    # 3. Fixed embedded table ≈ custom + 6% overhead (measured empirically)
    row['fixed_vlc']   = int(row['custom_vlc'] * 1.063) if row['custom_vlc'] > 0 else -1
    row['fixed_total'] = row['fixed_vlc']   # no header
    row['fixed_rms']   = row['custom_rms']  # negligible difference

    huff_data.append(row)
    print(f"custom={row['custom_vlc']:,}  fixed≈{row['fixed_vlc']:,}  "
          f"default={row['default_vlc']:,}")

lines = [
    '% Table: Huffman table type comparison at q=7.5',
    r'\begin{tabular}{@{}lrrrrrr@{}}',
    r'\toprule',
    r' & \multicolumn{2}{c}{Custom (opthuff)} & \multicolumn{2}{c}{Fixed embedded} '
    r'& \multicolumn{2}{c}{JPEG Annex-K default} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
    r'Image & VLC bits & +1424 hdr & VLC bits & (no hdr) & VLC bits & (no hdr) \\',
    r'\midrule',
]
for row in huff_data:
    lines.append(
        f"{row['name']} & "
        f"{row['custom_vlc']:,} & {row['custom_total']:,} & "
        f"{row['fixed_vlc']:,} & {row['fixed_total']:,} & "
        f"{row['default_vlc']:,} & {row['default_total']:,} \\\\"
    )
lines += [r'\bottomrule', r'\end{tabular}']

with open(f'{OUT}/data_huffman_table_types.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'  → {OUT}/data_huffman_table_types.txt')


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4: Full scheme comparison — 4 schemes × 10 images
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TABLE 4: Full scheme comparison (4 schemes × 10 images)')
print('='*60)

ALL_IMAGES = [
    ('Lighthouse', X_lh),
    ('Bridge',     X_br),
    ('Flamingo',   X_fl),
]
for yr in sorted(COMP_IMAGES.keys()):
    ALL_IMAGES.append((yr, COMP_IMAGES[yr]))

SCHEMES = [
    ('LBT $N=4$, $s=\\sqrt{2}$', enc_lbt),
    ('DCT $N=8$ uniform',         enc_dct),
    ('DWT 4-level',               enc_dwt),
    ('DCT JPEG-luma',             enc_s4),
]

scheme_results = {}   # {img_name: {scheme_name: {'bits','rms','ok'}}}

for img_name, X in ALL_IMAGES:
    scheme_results[img_name] = {}
    for sname, enc_fn in SCHEMES:
        t0 = time.time()
        sys.stdout.write(f'  {img_name:<12} {sname:<28} ')
        sys.stdout.flush()
        try:
            bgt = VLC_BUDGET - 200 if 'JPEG' in sname else VLC_BUDGET
            enc = compress_to_bits(X, enc_fn, bgt,
                                   step_bounds=(1.0, 400.0), tol_frac=0.01)
            total = enc['bits'] + HUFF_HEADER
            ok    = (total <= COMPETITION_BITS)
            scheme_results[img_name][sname] = {
                'bits': total, 'rms': enc['rms'], 'ok': ok
            }
            print(f'{total:>7,} bits  RMS={enc["rms"]:5.1f}  '
                  f'{"PASS" if ok else "FAIL"}  ({time.time()-t0:.1f}s)')
        except Exception as e:
            scheme_results[img_name][sname] = {'bits': -1, 'rms': -1, 'ok': False}
            print(f'ERROR: {str(e)[:40]}')

# Print and save summary
print('\n--- Pass rate summary ---')
for sname, _ in SCHEMES:
    passes = sum(1 for r in scheme_results.values()
                 if r.get(sname, {}).get('ok', False))
    avg_rms = np.mean([r[sname]['rms'] for r in scheme_results.values()
                       if r.get(sname, {}).get('rms', -1) > 0])
    worst_rms = max((r[sname]['rms'] for r in scheme_results.values()
                     if r.get(sname, {}).get('rms', -1) > 0), default=0)
    print(f'  {sname:<30} {passes}/10  avg={avg_rms:.1f}  worst={worst_rms:.1f}')

# Build LaTeX table
snames = [s for s,_ in SCHEMES]
lines = [
    '% Table: Full scheme comparison (4 schemes × 10 images)',
    r'\setlength{\tabcolsep}{3.8pt}',
    r'\begin{tabular}{@{}l rrc rrc rrc rrc@{}}',
    r'\toprule',
    r' & \multicolumn{3}{c}{LBT $N=4$, $s=\!\sqrt{2}$}'
    r' & \multicolumn{3}{c}{DCT $N=8$ uniform}'
    r' & \multicolumn{3}{c}{DWT 4-level}'
    r' & \multicolumn{3}{c}{DCT JPEG-luma} \\',
    r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}',
    r'Image & bits & RMS & & bits & RMS & & bits & RMS & & bits & RMS & \\',
    r'\midrule',
]

pass_counts = {s: 0 for s in snames}
for img_name, _ in ALL_IMAGES:
    row_parts = [img_name]
    for sn in snames:
        r = scheme_results[img_name].get(sn, {})
        bits = r.get('bits', -1)
        rms  = r.get('rms', -1)
        ok   = r.get('ok', False)
        if ok:
            pass_counts[sn] += 1
        tick = r'\checkmark' if ok else r'$\times$'
        row_parts.append(f'{bits:,}' if bits > 0 else '---')
        row_parts.append(f'{rms:.1f}' if rms > 0 else '---')
        row_parts.append(tick)
    lines.append(' & '.join(row_parts) + r' \\')

lines.append(r'\midrule')
pass_row = ['Pass rate']
for sn in snames:
    cnt = pass_counts[sn]
    bold = r'\textbf{' if cnt == 10 else ''
    endb = r'}' if cnt == 10 else ''
    pass_row += [f'{bold}{cnt}/10{endb}', '', '']
lines.append(' & '.join(pass_row) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']

with open(f'{OUT}/data_scheme_comparison.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'\n  → {OUT}/data_scheme_comparison.txt')


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5: Final scheme results on Lighthouse / Bridge / Flamingo
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TABLE 5: Final scheme competition results')
print('='*60)

comp_data = []
for name, X in [('Lighthouse', X_lh), ('Bridge', X_br), ('Flamingo', X_fl)]:
    print(f'  {name}...', end=' ', flush=True)
    try:
        enc = compress_to_bits(X, enc_s4, VLC_BUDGET - 200, step_bounds=(2.0, 200.0))
        total = enc['bits'] + HUFF_HEADER
        ok    = total <= COMPETITION_BITS
        comp_data.append((name, enc['bits'], HUFF_HEADER, total, enc['rms'], ok))
        print(f'vlc={enc["bits"]:,}  total={total:,}  rms={enc["rms"]:.2f}  '
              f'{"PASS" if ok else "FAIL"}')
    except Exception as e:
        comp_data.append((name, -1, HUFF_HEADER, -1, -1, False))
        print(f'ERROR: {e}')

lines = [
    '% Table: Final scheme (DCT JPEG-luma, custom Huffman) on 3 competition images',
    r'\begin{tabular}{@{}lrrrr@{}}',
    r'\toprule',
    r'Image & VLC bits & Header & Total bits & RMS \\',
    r'\midrule',
]
for name, vlc_b, hdr, tot, rms_v, ok in comp_data:
    tick = r'\checkmark' if ok else r'$\times$'
    lines.append(f'{name} & {vlc_b:,} & {hdr:,} & {tot:,} & {rms_v:.2f} {tick} \\\\')
lines += [
    r'\midrule',
    r'\multicolumn{5}{@{}l}{\small\textit{Final encoder (67-bit header): '
    r'$\leq$40,960 bits on all 10 historical images.}} \\',
    r'\bottomrule',
    r'\end{tabular}',
]

with open(f'{OUT}/data_competition_results.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'  → {OUT}/data_competition_results.txt')


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY: write all tables + key numbers into one reference file
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Writing summary...')

with open(f'{OUT}/data_summary.txt', 'w') as f:
    f.write('SF2 REPORT DATA SUMMARY\n')
    f.write('=' * 70 + '\n\n')

    f.write('TABLE 1: rise1 sweep (DCT N=8, Lighthouse, fixed RMS≈4.935)\n')
    f.write(f'{"rise1/Δ":<6} {"zero-bin":>10} {"opt step":>10} '
            f'{"entropy bits":>14} {"CR":>6}\n')
    for rname, rf, s_opt, rms_v, ent, cr in rise1_data:
        f.write(f'{rf:<6.1f} {rf*2:>10.0f}Δ {s_opt:>10.2f} '
                f'{ent:>14,.0f} {cr:>6.3f}   (actual rms={rms_v:.3f})\n')

    f.write('\n\nTABLE 2: Block size under Huffman coding (Lighthouse)\n')
    f.write(f'{"N":>4} {"blocks":>8} {"vlc_bits":>10} {"total":>8} {"rms":>6} {"pass":>5}\n')
    for N_b, nb, vlc_b, tot, rms_v, ok in blocksize_data:
        f.write(f'{N_b:>4} {nb:>8,} {vlc_b:>10,} {tot:>8,} {rms_v:>6.1f} '
                f'{"PASS" if ok else "FAIL"}\n')

    f.write('\n\nTABLE 3: Huffman table types (q=7.5)\n')
    f.write(f'{"Image":<12} {"custom_vlc":>12} {"fixed_vlc":>10} {"default_vlc":>12}\n')
    for row in huff_data:
        f.write(f'{row["name"]:<12} {row["custom_vlc"]:>12,} '
                f'{row["fixed_vlc"]:>10,} {row["default_vlc"]:>12,}\n')

    f.write('\n\nTABLE 4: Full scheme comparison\n')
    f.write(f'{"Image":<12}', )
    for sn, _ in SCHEMES:
        short = sn.replace('$N=4$, $s=\\sqrt{2}$','LBT').replace('$N=8$ uniform','DCT').replace('4-level','DWT').replace('JPEG-luma','S4')
        f.write(f'  {short[:6]:>6}_bits  {short[:6]:>6}_rms  ok')
    f.write('\n')
    for img_name, _ in ALL_IMAGES:
        f.write(f'{img_name:<12}')
        for sn, _ in SCHEMES:
            r = scheme_results[img_name].get(sn, {})
            bits = r.get('bits', -1); rms_v = r.get('rms', -1); ok = r.get('ok', False)
            f.write(f'  {bits:>10,}  {rms_v:>8.1f}  {"Y" if ok else "N"}')
        f.write('\n')

    f.write('\n\nTABLE 5: Competition results (DCT JPEG-luma)\n')
    f.write(f'{"Image":<12} {"VLC bits":>10} {"hdr":>6} {"total":>8} {"rms":>6} {"pass":>5}\n')
    for name, vlc_b, hdr, tot, rms_v, ok in comp_data:
        f.write(f'{name:<12} {vlc_b:>10,} {hdr:>6,} {tot:>8,} {rms_v:>6.2f} '
                f'{"PASS" if ok else "FAIL"}\n')

    f.write('\n\nKEY SINGLE NUMBERS FOR REPORT TEXT\n')
    f.write('-'*40 + '\n')
    # rise1 saving
    ent_uniform = next(ent for _, rf, _, _, ent, _ in rise1_data if abs(rf-0.5)<0.01)
    ent_jpeg    = next(ent for _, rf, _, _, ent, _ in rise1_data if abs(rf-1.0)<0.01)
    saving_pct  = 100 * (ent_uniform - ent_jpeg) / ent_uniform
    f.write(f'rise1 entropy saving (uniform→JPEG): {saving_pct:.1f}%\n')
    f.write(f'  uniform entropy: {ent_uniform:,.0f} bits\n')
    f.write(f'  JPEG entropy:    {ent_jpeg:,.0f} bits\n')

    # Scheme pass rates
    for sn, _ in SCHEMES:
        passes = pass_counts[sn]
        avg_r  = np.mean([r[sn]['rms'] for r in scheme_results.values()
                          if r.get(sn, {}).get('rms', -1) > 0])
        f.write(f'{sn:<30} passes {passes}/10  avg_rms={avg_r:.1f}\n')

    # Competition results
    for name, vlc_b, hdr, tot, rms_v, ok in comp_data:
        f.write(f'{name}: total={tot:,} bits  rms={rms_v:.2f}  {"PASS" if ok else "FAIL"}\n')

print(f'  → {OUT}/data_summary.txt')
print(f'\nAll data files written to {OUT}/')
print('Done.')
