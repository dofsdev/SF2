"""
helpers.py - Image compression helpers for SF2 lab.

Provides encode/decode functions for four schemes:
  - Laplacian Pyramid
  - DCT (Discrete Cosine Transform)
  - LBT (Lapped Bi-orthogonal Transform)
  - DWT (Discrete Wavelet Transform)

Each scheme exposes an encode() and decode() function with consistent
signatures, plus shared utilities for quantisation, entropy, and metrics.
"""

import numpy as np
import scipy.optimize
import scipy.io
from typing import Tuple, Optional

from cued_sf2_lab.laplacian_pyramid import (
    rowdec, rowdec2, rowint, rowint2,
    quantise, quant1, quant2, bpp,
)
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt, idwt, h1 as DWT_H1, h2 as DWT_H2, g1 as DWT_G1, g2 as DWT_G2


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(filename: str) -> np.ndarray:
    """Load a .mat image, return it zero-centred (subtract 128)."""
    data = scipy.io.loadmat(filename)
    X = data['X'].astype(float)
    return X - 128.0


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def rms_error(X: np.ndarray, Z: np.ndarray) -> float:
    """RMS (std-dev) error between reference X and reconstruction Z."""
    return float(np.std(X - Z))


def total_bits(imgs: list, steps) -> float:
    """Sum entropy-coded bits across a list of images with matching steps."""
    if np.isscalar(steps):
        steps = [steps] * len(imgs)
    return sum(bpp(quantise(img, s)) * img.size for img, s in zip(imgs, steps))


def dctbpp(Yr: np.ndarray, N: int = 16) -> float:
    """
    Estimate total bits for a regrouped DCT/LBT coefficient image.

    Yr  : output of regroup(Yq, transform_N)
    N   : coding block size (lab convention: always 16 for LBT, or the
          transform block size for plain DCT)
    """
    bits = 0.0
    for i in range(0, Yr.shape[0], N):
        for j in range(0, Yr.shape[1], N):
            block = Yr[i:i + N, j:j + N]
            bits += bpp(block) * block.size
    return bits


def find_step_for_rms(
    encode_fn,
    decode_fn,
    X: np.ndarray,
    target_rms: float,
    step_bounds: Tuple[float, float] = (0.01, 500.0),
    tol: float = 0.01,
) -> float:
    """
    Binary-search the scalar step size that makes rms_error(X, decode(encode(X, step))) == target_rms.

    encode_fn(step) -> encoded data
    decode_fn(encoded)  -> reconstructed image
    """
    def objective(step):
        enc = encode_fn(step)
        rec = decode_fn(enc)
        return abs(rms_error(X, rec) - target_rms)

    result = scipy.optimize.minimize_scalar(
        objective,
        bounds=step_bounds,
        method='bounded',
        options={'xatol': tol},
    )
    return float(result.x)


def direct_quantisation_reference(
    X: np.ndarray,
    step: float = 17.0,
    rise1: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Quantise X directly (no transform).

    Returns (Xq, rms, total_bits).
    """
    Xq = quantise(X, step, rise1)
    ref_rms = rms_error(X, Xq)
    ref_bits = bpp(Xq) * X.size
    return Xq, ref_rms, ref_bits


# ---------------------------------------------------------------------------
# 2-D separable filter helpers
# ---------------------------------------------------------------------------

def dec2(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Decimate rows then columns by 2 with filter h."""
    return rowdec(rowdec(X, h).T, h).T


def int2(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Interpolate rows then columns by 2 with filter h (DC gain = 2)."""
    return rowint(rowint(X, 2 * h).T, 2 * h).T


# ---------------------------------------------------------------------------
# Laplacian Pyramid
# ---------------------------------------------------------------------------

_DEFAULT_PYRAMID_H = np.array([1, 2, 1]) / 4.0


def pyramid_encode(
    X: np.ndarray,
    levels: int = 4,
    h: Optional[np.ndarray] = None,
    step: float = 17.0,
    steps: Optional[np.ndarray] = None,
    rise1: Optional[float] = None,
) -> dict:
    """
    Encode X with a Laplacian pyramid.

    Parameters
    ----------
    X      : zero-centred input image
    levels : number of pyramid levels (1-4)
    h      : decimation/interpolation filter (default [1,2,1]/4)
    step   : uniform step size (used if `steps` is None)
    steps  : per-band step sizes, length == levels+1; overrides `step`
    rise1  : quantiser first-rise (None => step/2; step => double-width zero bin)

    Returns dict with keys:
        'Yq'    : list of quantised high-pass images [Y0q, Y1q, ..., XNq]
        'steps' : array of step sizes used
        'bits'  : total estimated bits
        'rms'   : rms error between X and reconstruction
        'Z'     : reconstructed image
    """
    if h is None:
        h = _DEFAULT_PYRAMID_H

    # Build pyramid
    bands = []
    cur = X.copy()
    for _ in range(levels):
        low = dec2(cur, h)
        high = cur - int2(low, h)
        bands.append(high)
        cur = low
    bands.append(cur)  # final low-pass

    n_bands = len(bands)
    if steps is None:
        steps_arr = np.full(n_bands, step)
    else:
        steps_arr = np.asarray(steps, dtype=float)
        if len(steps_arr) != n_bands:
            raise ValueError(f'steps must have length {n_bands}, got {len(steps_arr)}')

    # Quantise
    Yq = [quantise(b, s, rise1) for b, s in zip(bands, steps_arr)]

    # Reconstruct
    Z = _pyramid_decode_bands(Yq, h)

    ref_bits = sum(bpp(q) * q.size for q in Yq)

    return {
        'Yq': Yq,
        'steps': steps_arr,
        'bits': ref_bits,
        'rms': rms_error(X, Z),
        'Z': Z,
    }


def _pyramid_decode_bands(Yq: list, h: np.ndarray) -> np.ndarray:
    """Reconstruct from a list [Y0q, Y1q, ..., XNq]."""
    Z = Yq[-1].copy()
    for band in reversed(Yq[:-1]):
        Z = int2(Z, h) + band
    return Z


def pyramid_decode(
    Yq: list,
    h: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Decode a quantised Laplacian pyramid.

    Yq : list [Y0q, Y1q, ..., XNq] as returned by pyramid_encode()['Yq']
    h  : filter used during encoding (default [1,2,1]/4)
    """
    if h is None:
        h = _DEFAULT_PYRAMID_H
    return _pyramid_decode_bands(Yq, h)


def pyramid_impulse_ratios(
    shape: Tuple[int, int],
    levels: int = 4,
    h: Optional[np.ndarray] = None,
    impulse_amp: float = 100.0,
) -> np.ndarray:
    """
    Compute equal-MSE step-size ratios for the Laplacian pyramid.

    Returns array of length (levels+1).  Ratios are relative to Y0 (=1).
    """
    if h is None:
        h = _DEFAULT_PYRAMID_H

    energies = np.zeros(levels + 1)
    for level in range(levels + 1):
        imp_bands = [np.zeros(shape) for _ in range(levels + 1)]
        imp = imp_bands[level]
        imp[imp.shape[0] // 2, imp.shape[1] // 2] = impulse_amp
        # adjust shape for deeper levels
        cur = np.zeros(shape)
        # rebuild pyramid shapes
        shapes = [shape]
        r, c = shape
        for _ in range(levels):
            r, c = (r + 1) // 2, (c + 1) // 2
            shapes.append((r, c))
        shapes.append(shapes[-1])  # final lowpass same size as level-4 dec

        imp_resized = [np.zeros(s) for s in shapes[:-1]] + [np.zeros(shapes[-2])]

        # Place impulse in the appropriate band
        target = imp_resized[level] if level < levels else imp_resized[-1]
        target[target.shape[0] // 2, target.shape[1] // 2] = impulse_amp

        Z = _pyramid_decode_bands(imp_resized, h)
        energies[level] = np.sum(Z ** 2)

    ratios = np.sqrt(energies[0] / np.maximum(energies, 1e-12))
    return ratios


def pyramid_equal_mse_steps(
    X: np.ndarray,
    levels: int = 4,
    h: Optional[np.ndarray] = None,
    target_rms: Optional[float] = None,
    base_step: float = 17.0,
    rise1: Optional[float] = None,
) -> np.ndarray:
    """
    Compute equal-MSE step sizes for the Laplacian pyramid.

    If target_rms is given, the base step is optimised to match it.
    Otherwise base_step scales the ratios directly.

    Returns array of step sizes, length == levels+1.
    """
    if h is None:
        h = _DEFAULT_PYRAMID_H

    # Build pyramid to get band shapes
    bands = []
    cur = X.copy()
    for _ in range(levels):
        low = dec2(cur, h)
        high = cur - int2(low, h)
        bands.append(high)
        cur = low
    bands.append(cur)

    # Measure impulse energies using actual band shapes
    energies = np.zeros(levels + 1)
    for idx, band in enumerate(bands):
        imp = np.zeros_like(band)
        imp[band.shape[0] // 2, band.shape[1] // 2] = 100.0
        imp_list = [np.zeros_like(b) for b in bands]
        imp_list[idx] = imp
        Z = _pyramid_decode_bands(imp_list, h)
        energies[idx] = np.sum(Z ** 2)

    ratios = np.sqrt(energies[0] / np.maximum(energies, 1e-12))

    if target_rms is not None:
        def obj(base):
            steps_try = base * ratios
            Yq = [quantise(b, s, rise1) for b, s in zip(bands, steps_try)]
            Z = _pyramid_decode_bands(Yq, h)
            return abs(rms_error(X, Z) - target_rms)

        res = scipy.optimize.minimize_scalar(obj, bounds=(0.01, 500.0), method='bounded')
        base_step = float(res.x)

    return base_step * ratios


# ---------------------------------------------------------------------------
# DCT
# ---------------------------------------------------------------------------

def dct_encode(
    X: np.ndarray,
    N: int = 8,
    step: float = 17.0,
    rise1: Optional[float] = None,
) -> dict:
    """
    Encode X with a block DCT.

    Parameters
    ----------
    X    : zero-centred input image (height must be multiple of N)
    N    : DCT block size
    step : uniform quantisation step
    rise1: quantiser first-rise (None => step/2)

    Returns dict with keys:
        'Yq'   : quantised DCT coefficients
        'N'    : block size used
        'step' : step size used
        'bits' : total estimated bits (using N-sized coding blocks)
        'rms'  : rms error
        'Z'    : reconstructed image
    """
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    Yq = quantise(Y, step, rise1)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)

    Yr = regroup(Yq, N)
    bits = dctbpp(Yr, N)

    return {
        'Yq': Yq,
        'N': N,
        'step': step,
        'bits': bits,
        'rms': rms_error(X, Z),
        'Z': Z,
    }


def dct_decode(
    Yq: np.ndarray,
    N: int = 8,
) -> np.ndarray:
    """Reconstruct from quantised DCT coefficients."""
    C = dct_ii(N)
    return colxfm(colxfm(Yq.T, C.T).T, C.T)


def dct_encode_perband(
    X: np.ndarray,
    N: int = 8,
    step_matrix: Optional[np.ndarray] = None,
    step: float = 17.0,
    rise1: Optional[float] = None,
    rise1_matrix: Optional[np.ndarray] = None,
) -> dict:
    """
    Encode X with a block DCT, applying per-subband step sizes.

    step_matrix : (N, N) array of step sizes, one per DCT frequency bin.
                  If None, uses uniform `step` for all bins.
    rise1_matrix: (N, N) array of rise1 values; use np.inf to suppress a bin.
    """
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    Yr = regroup(Y, N)

    if step_matrix is None:
        step_matrix = np.full((N, N), step)

    H, W = X.shape
    sub_h = H // N
    sub_w = W // N

    Yq_r = np.zeros_like(Yr)
    for u in range(N):
        for v in range(N):
            sub = Yr[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w]
            s = step_matrix[u, v]
            r1 = None if rise1_matrix is None else rise1_matrix[u, v]
            if r1 is None:
                r1 = rise1
            Yq_r[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w] = quantise(sub, s, r1)

    # Un-regroup to block layout
    Yq = regroup(Yq_r, [sub_h, sub_w])

    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    bits = dctbpp(Yq_r, N)

    return {
        'Yq': Yq,
        'Yq_r': Yq_r,
        'N': N,
        'step_matrix': step_matrix,
        'bits': bits,
        'rms': rms_error(X, Z),
        'Z': Z,
    }


# ---------------------------------------------------------------------------
# LBT (Lapped Bi-orthogonal Transform)
# ---------------------------------------------------------------------------

def lbt_encode(
    X: np.ndarray,
    N: int = 8,
    s: float = (1 + 5 ** 0.5) / 2,
    step: float = 17.0,
    rise1: Optional[float] = None,
    coding_N: int = 16,
) -> dict:
    """
    Encode X with the LBT (POT pre-filter + block DCT).

    Parameters
    ----------
    X        : zero-centred input image
    N        : DCT/POT block size (must be even)
    s        : POT scaling factor (1 <= s <= 2; golden ratio default)
               s=1  => LOT (Pf == Pr), s=sqrt(2) often optimal for compression
    step     : uniform quantisation step
    rise1    : quantiser first-rise (None => step/2)
    coding_N : block size for entropy estimate (lab convention: 16)

    Returns dict with keys:
        'Yq', 'N', 's', 'step', 'Pf', 'Pr', 'bits', 'rms', 'Z'
    """
    C = dct_ii(N)
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N // 2:-N // 2]

    # Forward POT pre-filter
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    # 2-D DCT
    Y = colxfm(colxfm(Xp, C).T, C).T
    Yq = quantise(Y, step, rise1)

    # Inverse 2-D DCT
    Zp = colxfm(colxfm(Yq.T, C.T).T, C.T)

    # Inverse POT post-filter
    Z = Zp.copy()
    Z[:, t] = colxfm(Z[:, t].T, Pr.T).T
    Z[t, :] = colxfm(Z[t, :], Pr.T)

    Yr = regroup(Yq, N)
    bits = dctbpp(Yr, coding_N)

    return {
        'Yq': Yq,
        'N': N,
        's': s,
        'step': step,
        'Pf': Pf,
        'Pr': Pr,
        'bits': bits,
        'rms': rms_error(X, Z),
        'Z': Z,
    }


def lbt_decode(
    Yq: np.ndarray,
    N: int = 8,
    s: float = (1 + 5 ** 0.5) / 2,
    Pr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct from quantised LBT coefficients.

    Pr may be supplied directly (e.g. from lbt_encode()['Pr']).
    If omitted it is recomputed from N and s.
    """
    C = dct_ii(N)
    if Pr is None:
        _, Pr = pot_ii(N, s)
    t = np.s_[N // 2:-N // 2]

    Zp = colxfm(colxfm(Yq.T, C.T).T, C.T)
    Z = Zp.copy()
    Z[:, t] = colxfm(Z[:, t].T, Pr.T).T
    Z[t, :] = colxfm(Z[t, :], Pr.T)
    return Z


def lbt_encode_perband(
    X: np.ndarray,
    N: int = 8,
    s: float = (1 + 5 ** 0.5) / 2,
    step_matrix: Optional[np.ndarray] = None,
    step: float = 17.0,
    rise1: Optional[float] = None,
    rise1_matrix: Optional[np.ndarray] = None,
    coding_N: int = 16,
) -> dict:
    """
    LBT encode with per-subband step sizes and optional suppression.

    step_matrix  : (N, N) array; use np.inf to suppress a frequency bin.
    rise1_matrix : (N, N) array of rise1 values per bin.
    """
    C = dct_ii(N)
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N // 2:-N // 2]

    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    Y = colxfm(colxfm(Xp, C).T, C).T
    Yr = regroup(Y, N)

    if step_matrix is None:
        step_matrix = np.full((N, N), step)

    H, W = X.shape
    sub_h = H // N
    sub_w = W // N

    Yq_r = np.zeros_like(Yr)
    for u in range(N):
        for v in range(N):
            sub = Yr[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w]
            sv = step_matrix[u, v]
            r1 = None if rise1_matrix is None else rise1_matrix[u, v]
            if r1 is None:
                r1 = rise1
            Yq_r[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w] = quantise(sub, sv, r1)

    Yq = regroup(Yq_r, [sub_h, sub_w])

    Zp = colxfm(colxfm(Yq.T, C.T).T, C.T)
    Z = Zp.copy()
    Z[:, t] = colxfm(Z[:, t].T, Pr.T).T
    Z[t, :] = colxfm(Z[t, :], Pr.T)

    bits = dctbpp(Yq_r, coding_N)

    return {
        'Yq': Yq,
        'Yq_r': Yq_r,
        'N': N,
        's': s,
        'step_matrix': step_matrix,
        'Pf': Pf,
        'Pr': Pr,
        'bits': bits,
        'rms': rms_error(X, Z),
        'Z': Z,
    }


# ---------------------------------------------------------------------------
# DWT (Discrete Wavelet Transform)
# ---------------------------------------------------------------------------

def nlevdwt(
    X: np.ndarray,
    n: int,
    h1: np.ndarray = DWT_H1,
    h2: np.ndarray = DWT_H2,
) -> np.ndarray:
    """n-level 2-D DWT using LeGall filters by default."""
    Y = X.copy()
    rows, cols = Y.shape
    for level in range(n):
        r = rows >> level
        c = cols >> level
        Y[:r, :c] = dwt(Y[:r, :c], h1, h2)
    return Y


def nlevidwt(
    Yq: np.ndarray,
    n: int,
    g1: np.ndarray = DWT_G1,
    g2: np.ndarray = DWT_G2,
) -> np.ndarray:
    """n-level 2-D inverse DWT using LeGall filters by default."""
    Z = Yq.copy()
    rows, cols = Z.shape
    for level in range(n - 1, -1, -1):
        r = rows >> level
        c = cols >> level
        Z[:r, :c] = idwt(Z[:r, :c], g1, g2)
    return Z


def quantdwt(
    Y: np.ndarray,
    dwtstep: np.ndarray,
    rise1: Optional[float] = None,
    rise1_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantise an n-level DWT image per subband.

    Parameters
    ----------
    Y        : n-level DWT image from nlevdwt()
    dwtstep  : (3, n+1) array.
               dwtstep[0,i] = step for top-right (HL) at level i
               dwtstep[1,i] = step for bottom-left (LH) at level i
               dwtstep[2,i] = step for bottom-right (HH) at level i
               dwtstep[0,n] = step for final low-pass (LL)
    rise1    : uniform first-rise for all bands (None => step/2)
    rise1_matrix : (3, n+1) per-band first-rise; overrides rise1

    Returns (Yq, dwtent) where dwtent has the same shape as dwtstep.
    """
    Yq = Y.copy()
    n = dwtstep.shape[1] - 1
    dwtent = np.zeros_like(dwtstep, dtype=float)
    rows, cols = Y.shape

    def _r1(k, i):
        if rise1_matrix is not None:
            return rise1_matrix[k, i]
        return rise1

    for level in range(n):
        r = rows >> level
        c = cols >> level
        r2 = r >> 1
        c2 = c >> 1

        # HL (top-right)
        sub = Y[:r2, c2:c]
        Yq[:r2, c2:c] = quantise(sub, dwtstep[0, level], _r1(0, level))
        dwtent[0, level] = bpp(Yq[:r2, c2:c])

        # LH (bottom-left)
        sub = Y[r2:r, :c2]
        Yq[r2:r, :c2] = quantise(sub, dwtstep[1, level], _r1(1, level))
        dwtent[1, level] = bpp(Yq[r2:r, :c2])

        # HH (bottom-right)
        sub = Y[r2:r, c2:c]
        Yq[r2:r, c2:c] = quantise(sub, dwtstep[2, level], _r1(2, level))
        dwtent[2, level] = bpp(Yq[r2:r, c2:c])

    # Final LL (low-pass)
    r_lp = rows >> n
    c_lp = cols >> n
    Yq[:r_lp, :c_lp] = quantise(Y[:r_lp, :c_lp], dwtstep[0, n], _r1(0, n))
    dwtent[0, n] = bpp(Yq[:r_lp, :c_lp])

    return Yq, dwtent


def dwt_total_bits(
    dwtent: np.ndarray,
    image_shape: Tuple[int, int],
    n: int,
) -> float:
    """Total entropy-coded bits from a dwtent matrix."""
    rows, cols = image_shape
    bits = 0.0
    for level in range(n):
        pixels = (rows >> (level + 1)) * (cols >> (level + 1))
        bits += (dwtent[0, level] + dwtent[1, level] + dwtent[2, level]) * pixels
    lp_pixels = (rows >> n) * (cols >> n)
    bits += dwtent[0, n] * lp_pixels
    return bits


def dwt_encode(
    X: np.ndarray,
    n: int = 4,
    step: float = 17.0,
    dwtstep: Optional[np.ndarray] = None,
    rise1: Optional[float] = None,
    rise1_matrix: Optional[np.ndarray] = None,
    h1: np.ndarray = DWT_H1,
    h2: np.ndarray = DWT_H2,
) -> dict:
    """
    Encode X with an n-level DWT.

    Parameters
    ----------
    X        : zero-centred input image
    n        : number of DWT levels
    step     : uniform step size for all subbands (used if dwtstep is None)
    dwtstep  : (3, n+1) per-subband step matrix; overrides step
    rise1    : uniform first-rise (None => step/2)
    rise1_matrix : (3, n+1) per-subband first-rise
    h1, h2   : analysis filters (LeGall by default)

    Returns dict with keys:
        'Yq', 'dwtent', 'dwtstep', 'n', 'bits', 'rms', 'Z'
    """
    Y = nlevdwt(X, n, h1, h2)

    if dwtstep is None:
        dwtstep = np.full((3, n + 1), step)

    Yq, dwtent = quantdwt(Y, dwtstep, rise1, rise1_matrix)
    Z = nlevidwt(Yq, n)

    bits = dwt_total_bits(dwtent, X.shape, n)

    return {
        'Yq': Yq,
        'dwtent': dwtent,
        'dwtstep': dwtstep,
        'n': n,
        'bits': bits,
        'rms': rms_error(X, Z),
        'Z': Z,
    }


def dwt_decode(
    Yq: np.ndarray,
    n: int = 4,
    g1: np.ndarray = DWT_G1,
    g2: np.ndarray = DWT_G2,
) -> np.ndarray:
    """Reconstruct from a quantised n-level DWT image."""
    return nlevidwt(Yq, n, g1, g2)


def dwt_equal_mse_steps(
    X: np.ndarray,
    n: int = 4,
    target_rms: Optional[float] = None,
    base_step: float = 17.0,
    rise1: Optional[float] = None,
    impulse_amp: float = 100.0,
    h1: np.ndarray = DWT_H1,
    h2: np.ndarray = DWT_H2,
) -> np.ndarray:
    """
    Compute equal-MSE step matrix for DWT subbands.

    Impulse response energies are measured per subband and used to
    scale step sizes so each band contributes equally to MSE.

    If target_rms is given, the base step is optimised to match it.

    Returns (3, n+1) step matrix.
    """
    rows, cols = X.shape
    energies = np.zeros((3, n + 1))

    def _subband_slices(level):
        r = rows >> level
        c = cols >> level
        r2 = r >> 1
        c2 = c >> 1
        return [
            (0, level, slice(0, r2), slice(c2, c)),
            (1, level, slice(r2, r), slice(0, c2)),
            (2, level, slice(r2, r), slice(c2, c)),
        ]

    for level in range(n):
        for k, lev, rs, cs in _subband_slices(level):
            Yimp = np.zeros((rows, cols))
            r_mid = (rs.start + rs.stop) // 2
            c_mid = (cs.start + cs.stop) // 2
            Yimp[r_mid, c_mid] = impulse_amp
            Zimp = nlevidwt(Yimp, n)
            energies[k, lev] = np.sum(Zimp ** 2)

    # Final low-pass
    r_lp = rows >> n
    c_lp = cols >> n
    Yimp = np.zeros((rows, cols))
    Yimp[r_lp // 2, c_lp // 2] = impulse_amp
    Zimp = nlevidwt(Yimp, n)
    energies[0, n] = np.sum(Zimp ** 2)

    # Ratios relative to LL band (smallest energy -> largest step)
    ref_energy = energies[0, n]
    ratios = np.sqrt(ref_energy / np.maximum(energies, 1e-12))

    if target_rms is not None:
        Y = nlevdwt(X, n, h1, h2)

        def obj(base):
            dstep = base * ratios
            Yq, dwtent = quantdwt(Y, dstep, rise1)
            Z = nlevidwt(Yq, n)
            return abs(rms_error(X, Z) - target_rms)

        res = scipy.optimize.minimize_scalar(obj, bounds=(0.01, 500.0), method='bounded')
        base_step = float(res.x)

    return base_step * ratios


# ---------------------------------------------------------------------------
# Convenience: encode with automatic step optimisation
# ---------------------------------------------------------------------------

def encode_matched_rms(
    X: np.ndarray,
    scheme: str,
    target_rms: float,
    scheme_kwargs: Optional[dict] = None,
    step_bounds: Tuple[float, float] = (0.01, 500.0),
) -> dict:
    """
    Encode X with any scheme, finding the step size that matches target_rms.

    scheme       : 'pyramid' | 'dct' | 'lbt' | 'dwt'
    scheme_kwargs: extra keyword args forwarded to the encode function
                   (excluding `step`)
    Returns the encode dict for the optimal step.
    """
    kwargs = dict(scheme_kwargs or {})

    fns = {
        'pyramid': pyramid_encode,
        'dct': dct_encode,
        'lbt': lbt_encode,
        'dwt': dwt_encode,
    }
    if scheme not in fns:
        raise ValueError(f'Unknown scheme "{scheme}". Choose from {list(fns)}.')

    fn = fns[scheme]

    def objective(step):
        result = fn(X, step=step, **kwargs)
        return abs(result['rms'] - target_rms)

    res = scipy.optimize.minimize_scalar(
        objective,
        bounds=step_bounds,
        method='bounded',
        options={'xatol': 1e-3},
    )
    return fn(X, step=float(res.x), **kwargs)


# ---------------------------------------------------------------------------
# Centre-clipped quantisation helpers
# ---------------------------------------------------------------------------

def optimal_rise1(
    X: np.ndarray,
    step: float,
    rise1_fractions: Tuple[float, ...] = (0.5, 1.0, 1.5),
) -> Tuple[float, float, float]:
    """
    Test several rise1/step ratios and return the one with the best (lowest) bpp.

    Returns (best_rise1, best_bpp, best_rms).
    """
    best = None
    for frac in rise1_fractions:
        r1 = frac * step
        Xq = quantise(X, step, r1)
        b = bpp(Xq)
        err = rms_error(X, Xq)
        if best is None or b < best[1]:
            best = (r1, b, err)
    return best


def suppress_highfreq_dct(
    N: int,
    suppress_last: int = 1,
    step: float = 17.0,
) -> np.ndarray:
    """
    Return a (N, N) step matrix that suppresses the highest-frequency corner.

    suppress_last: number of rows/cols from the bottom-right corner to suppress
                   (set their step to np.inf).
    """
    mat = np.full((N, N), step)
    for u in range(N - suppress_last, N):
        for v in range(N - suppress_last, N):
            mat[u, v] = np.inf
    return mat


def suppress_highfreq_dwt(
    n: int,
    step: float = 17.0,
    suppress_hh_level: int = 0,
) -> np.ndarray:
    """
    Return a (3, n+1) DWT step matrix that suppresses the HH band at the
    finest level (level 0) if suppress_hh_level == 0.

    suppress_hh_level: which level's HH band (index 2) to suppress (np.inf).
    """
    mat = np.full((3, n + 1), step)
    if 0 <= suppress_hh_level < n:
        mat[2, suppress_hh_level] = np.inf
    return mat
