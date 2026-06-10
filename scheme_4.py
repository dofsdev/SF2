"""
Scheme 4: DCT N=8 with JPEG standard luminance quantisation table.

Steps scaled by factor q (binary-searched to hit bit budget).
Uses double-width zero bin (rise1=step) and custom Huffman table.

Better perceptual quality than uniform DCT: reduces step sizes for
low-frequency subbands (perceptually important) relative to high-freq.
"""

import numpy as np

from helpers import (
    dct_ii, colxfm, regroup,
    quant1, quant2,
    huffman_encode_blocks, huffman_decode_blocks,
    rms_error, load_image,
)

_N = 8
_DCBITS = 11

# JPEG standard luminance quantisation table (ISO/IEC 10918-1 Annex K)
JPEG_LUMA_QUANT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=float)


def encode(X: np.ndarray, q: float) -> dict:
    """
    Encode X with DCT N=8 + JPEG luma quant table scaled by q.

    q acts like a quality factor: smaller q → fewer bits, higher RMS.
    rise1=step for each subband (double-width zero bin, same as jpegenc).
    Returns dict with keys: vlc, hufftab, bits, Z, rms, q, step_matrix.
    """
    N = _N
    C = dct_ii(N)
    H, W = X.shape
    sub_h, sub_w = H // N, W // N

    # 2-D DCT then subband layout
    Y = colxfm(colxfm(X, C).T, C).T
    Yr = regroup(Y, N)

    step_matrix = q * JPEG_LUMA_QUANT

    Yq_int_r = np.zeros_like(Yr)
    for u in range(N):
        for v in range(N):
            s = step_matrix[u, v]
            sub = Yr[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w]
            # rise1=s → double-width zero bin (JPEG standard centre-clip)
            Yq_int_r[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w] = quant1(sub, s, s)

    # Convert to block layout for Huffman coding
    Yq_int = regroup(Yq_int_r, [sub_h, sub_w])

    vlc, hufftab = huffman_encode_blocks(Yq_int, N, opthuff=True, dcbits=_DCBITS)
    bits = int(vlc[:, 1].sum())

    # Decode
    Zq_int = huffman_decode_blocks(vlc, N, hufftab=hufftab, H=H, W=W, dcbits=_DCBITS)
    Zq_int_r = regroup(Zq_int, N)
    Zr = np.zeros_like(Yr)
    for u in range(N):
        for v in range(N):
            s = step_matrix[u, v]
            sub = Zq_int_r[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w]
            Zr[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w] = quant2(sub, s)

    Zr_blocks = regroup(Zr, [sub_h, sub_w])
    Z = colxfm(colxfm(Zr_blocks.T, C.T).T, C.T)

    return {
        'vlc': vlc,
        'hufftab': hufftab,
        'bits': bits,
        'Z': Z,
        'rms': rms_error(X, Z),
        'q': q,
        'step_matrix': step_matrix,
    }


def decode(vlc: np.ndarray, q: float, hufftab, H: int, W: int) -> np.ndarray:
    N = _N
    C = dct_ii(N)
    sub_h, sub_w = H // N, W // N
    step_matrix = q * JPEG_LUMA_QUANT

    Zq_int = huffman_decode_blocks(vlc, N, hufftab=hufftab, H=H, W=W, dcbits=_DCBITS)
    Zq_int_r = regroup(Zq_int, N)
    Zr = np.zeros((H, W))
    for u in range(N):
        for v in range(N):
            s = step_matrix[u, v]
            sub = Zq_int_r[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w]
            Zr[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w] = quant2(sub, s)

    Zr_blocks = regroup(Zr, [sub_h, sub_w])
    return colxfm(colxfm(Zr_blocks.T, C.T).T, C.T)


if __name__ == '__main__':
    X = load_image('lighthouse.mat')
    r = encode(X, q=1.0)
    print(f'q=1.0  bits={r["bits"]}  RMS={r["rms"]:.3f}')
    r = encode(X, q=2.0)
    print(f'q=2.0  bits={r["bits"]}  RMS={r["rms"]:.3f}')
