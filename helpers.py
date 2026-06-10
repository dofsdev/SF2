import numpy as np
import scipy.optimize
import scipy.io
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Callable

from cued_sf2_lab.laplacian_pyramid import (
    rowdec, rowdec2, rowint, rowint2,
    quantise, quant1, quant2, bpp,
)
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt, idwt, h1 as DWT_H1, h2 as DWT_H2, g1 as DWT_G1, g2 as DWT_G2
from cued_sf2_lab.jpeg import jpegenc, jpegdec, dwtgroup, diagscan, runampl, huffenc, huffdflt, huffdes, huffgen


def load_image(filename: str) -> np.ndarray:
    data = scipy.io.loadmat(filename)
    X = data['X'].astype(float)
    return X - 128.0

def rms_error(X: np.ndarray, Z: np.ndarray) -> float:
    return float(np.std(X - Z))


def total_bits(imgs: list, steps) -> float:
    if np.isscalar(steps):
        steps = [steps] * len(imgs)
    return sum(bpp(quantise(img, s)) * img.size for img, s in zip(imgs, steps))


def dctbpp(Yr: np.ndarray, N: int = 16) -> float:
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

    Xq = quantise(X, step, rise1)
    ref_rms = rms_error(X, Xq)
    ref_bits = bpp(Xq) * X.size
    return Xq, ref_rms, ref_bits


def dec2(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    return rowdec(rowdec(X, h).T, h).T


def int2(X: np.ndarray, h: np.ndarray) -> np.ndarray:
    return rowint(rowint(X, 2 * h).T, 2 * h).T


_DEFAULT_PYRAMID_H = np.array([1, 2, 1]) / 4.0

def pyramid_encode(
    X: np.ndarray,
    levels: int = 4,
    h: Optional[np.ndarray] = None,
    step: float = 17.0,
    steps: Optional[np.ndarray] = None,
    rise1: Optional[float] = None,
) -> dict:

    if h is None:
        h = _DEFAULT_PYRAMID_H

    bands = []
    cur = X.copy()
    for _ in range(levels):
        low = dec2(cur, h)
        high = cur - int2(low, h)
        bands.append(high)
        cur = low
    bands.append(cur)

    n_bands = len(bands)
    if steps is None:
        steps_arr = np.full(n_bands, step)
    else:
        steps_arr = np.asarray(steps, dtype=float)
        if len(steps_arr) != n_bands:
            raise ValueError(f'steps must have length {n_bands}, got {len(steps_arr)}')

    Yq = [quantise(b, s, rise1) for b, s in zip(bands, steps_arr)]

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

    Z = Yq[-1].copy()
    for band in reversed(Yq[:-1]):
        Z = int2(Z, h) + band
    return Z


def pyramid_decode(
    Yq: list,
    h: Optional[np.ndarray] = None,
) -> np.ndarray:

    if h is None:
        h = _DEFAULT_PYRAMID_H
    return _pyramid_decode_bands(Yq, h)


def pyramid_impulse_ratios(
    shape: Tuple[int, int],
    levels: int = 4,
    h: Optional[np.ndarray] = None,
    impulse_amp: float = 100.0,
) -> np.ndarray:

    if h is None:
        h = _DEFAULT_PYRAMID_H

    energies = np.zeros(levels + 1)
    for level in range(levels + 1):
        imp_bands = [np.zeros(shape) for _ in range(levels + 1)]
        imp = imp_bands[level]
        imp[imp.shape[0] // 2, imp.shape[1] // 2] = impulse_amp

        cur = np.zeros(shape)

        shapes = [shape]
        r, c = shape
        for _ in range(levels):
            r, c = (r + 1) // 2, (c + 1) // 2
            shapes.append((r, c))
        shapes.append(shapes[-1])

        imp_resized = [np.zeros(s) for s in shapes[:-1]] + [np.zeros(shapes[-2])]

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

    if h is None:
        h = _DEFAULT_PYRAMID_H

    bands = []
    cur = X.copy()
    for _ in range(levels):
        low = dec2(cur, h)
        high = cur - int2(low, h)
        bands.append(high)
        cur = low
    bands.append(cur)

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

    def _quantise_band(src, dst_slice_r, dst_slice_c, step, r1):
        if not np.isfinite(step):
            Yq[dst_slice_r, dst_slice_c] = 0.0
            dwtent_val = 0.0
        else:
            Yq[dst_slice_r, dst_slice_c] = quantise(src, step, r1)
            dwtent_val = bpp(Yq[dst_slice_r, dst_slice_c])
        return dwtent_val

    for level in range(n):
        r = rows >> level
        c = cols >> level
        r2 = r >> 1
        c2 = c >> 1

        dwtent[0, level] = _quantise_band(Y[:r2, c2:c], slice(0, r2), slice(c2, c), dwtstep[0, level], _r1(0, level))
        dwtent[1, level] = _quantise_band(Y[r2:r, :c2], slice(r2, r), slice(0, c2), dwtstep[1, level], _r1(1, level))
        dwtent[2, level] = _quantise_band(Y[r2:r, c2:c], slice(r2, r), slice(c2, c), dwtstep[2, level], _r1(2, level))

    # Final LL (low-pass)
    r_lp = rows >> n
    c_lp = cols >> n
    dwtent[0, n] = _quantise_band(Y[:r_lp, :c_lp], slice(0, r_lp), slice(0, c_lp), dwtstep[0, n], _r1(0, n))

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


# ---------------------------------------------------------------------------
# JPEG Huffman coding wrappers (notebooks 10-12)
# ---------------------------------------------------------------------------

def jpeg_encode_dct(
    X: np.ndarray,
    qstep: float,
    N: int = 8,
    M: int = 8,
    opthuff: bool = False,
    rise1: Optional[float] = None,
    dcbits: int = 10,
) -> dict:
    """
    Encode X with DCT + JPEG run-length/Huffman coding.

    Parameters
    ----------
    X      : input image (NOT zero-centred; jpegenc works on 0-255 range)
    qstep  : quantisation step
    N      : DCT block size
    M      : coding block size (must be multiple of N; use M > N to group blocks)
    opthuff: use custom Huffman tables designed from this image
    rise1  : if given, overrides quant1's default rise (step/2) to widen zero bin
    dcbits : bits for DC coefficient (default 8)

    Returns dict with keys:
        'vlc', 'hufftab', 'bits', 'N', 'M', 'qstep'
    """
    # jpegenc internally uses quant1(Y, qstep, qstep) i.e. rise1=qstep.
    # To use a different rise1 we pre-quantise and pass qstep=1 so the
    # internal quantisation is a no-op.  However that changes the codeword
    # statistics, so we keep the default jpegenc behaviour for simplicity.
    vlc, hufftab = jpegenc(X, qstep, N=N, M=M, opthuff=opthuff,
                           dcbits=dcbits, log=False)
    bits = int(vlc[:, 1].sum())
    return {
        'vlc': vlc,
        'hufftab': hufftab,
        'bits': bits,
        'N': N,
        'M': M,
        'qstep': qstep,
    }


def jpeg_decode_dct(
    vlc: np.ndarray,
    qstep: float,
    N: int = 8,
    M: int = 8,
    hufftab=None,
    W: int = 256,
    H: int = 256,
    dcbits: int = 10,
) -> np.ndarray:
    """Decode a JPEG DCT bitstream produced by jpeg_encode_dct()."""
    return jpegdec(vlc, qstep, N=N, M=M, hufftab=hufftab,
                   dcbits=dcbits, W=W, H=H, log=False)


def jpeg_encode_lbt(
    X: np.ndarray,
    qstep: float,
    N: int = 4,
    s: float = (1 + 5 ** 0.5) / 2,
    M: int = 16,
    opthuff: bool = False,
    dcbits: int = 10,
) -> dict:
    """
    Encode X with LBT (POT pre-filter + JPEG DCT/Huffman).

    The POT pre-filter is applied first, then jpegenc handles the DCT and
    Huffman coding.  M > N causes jpegenc to regroup N-sized blocks into
    M-sized coding blocks (lab convention: N=4, M=16).

    Returns dict with keys:
        'vlc', 'hufftab', 'bits', 'N', 's', 'M', 'qstep', 'Pf', 'Pr'
    """
    C = dct_ii(N)
    Pf, Pr = pot_ii(N, s)
    t = np.s_[N // 2:-N // 2]

    # POT pre-filter (applied to zero-centred image)
    Xp = X.copy()
    Xp[t, :] = colxfm(Xp[t, :], Pf)
    Xp[:, t] = colxfm(Xp[:, t].T, Pf).T

    # jpegenc expects 0-255 range for its internal DC offset, but since we
    # apply the transform ourselves we just pass the pre-filtered image.
    vlc, hufftab = jpegenc(Xp, qstep, N=N, M=M, opthuff=opthuff,
                           dcbits=dcbits, log=False)
    bits = int(vlc[:, 1].sum())
    return {
        'vlc': vlc,
        'hufftab': hufftab,
        'bits': bits,
        'N': N,
        's': s,
        'M': M,
        'qstep': qstep,
        'Pf': Pf,
        'Pr': Pr,
    }


def jpeg_decode_lbt(
    vlc: np.ndarray,
    qstep: float,
    N: int = 4,
    s: float = (1 + 5 ** 0.5) / 2,
    M: int = 16,
    hufftab=None,
    Pr: Optional[np.ndarray] = None,
    W: int = 256,
    H: int = 256,
    dcbits: int = 10,
) -> np.ndarray:
    """
    Decode a JPEG LBT bitstream produced by jpeg_encode_lbt().

    jpegdec reverses the DCT; we then apply the POT post-filter.
    """
    if Pr is None:
        _, Pr = pot_ii(N, s)
    t = np.s_[N // 2:-N // 2]

    Zp = jpegdec(vlc, qstep, N=N, M=M, hufftab=hufftab,
                 dcbits=dcbits, W=W, H=H, log=False)

    Z = Zp.copy()
    Z[:, t] = colxfm(Z[:, t].T, Pr.T).T
    Z[t, :] = colxfm(Z[t, :], Pr.T)
    return Z


def vlc_bits(vlc: np.ndarray) -> int:
    """Total number of bits in a variable-length code array."""
    return int(vlc[:, 1].sum())


# ---------------------------------------------------------------------------
# Scheme comparison (notebook 10-11)
# ---------------------------------------------------------------------------

def compare_schemes(
    X: np.ndarray,
    schemes: Dict[str, Callable],
    ref_step: float = 17.0,
    plot: bool = True,
    images: Optional[list] = None,
    image_names: Optional[list] = None,
    figsize: Optional[Tuple] = None,
) -> dict:
    """
    Benchmark multiple compression schemes against direct quantisation.

    Each scheme is an encode function with signature:
        encode_fn(X: np.ndarray, step: float) -> dict
    The returned dict must have keys: 'Z' (reconstruction), 'bits' (float).

    Parameters
    ----------
    X           : zero-centred image to compress
    schemes     : {'scheme name': encode_fn, ...}
    ref_step    : step size for the direct quantisation reference
    plot        : show side-by-side reconstructions and error images
    images      : additional images to test (list of ndarrays)
    image_names : names for additional images (list of str)
    figsize     : figure size override

    Returns
    -------
    results : {
        'reference': {'rms': float, 'bits': float},
        scheme_name: {'rms': float, 'bits': float, 'CR': float, 'Z': ndarray, 'step': float},
        ...
    }
    """
    # Reference
    Xq_ref, ref_rms, ref_bits = direct_quantisation_reference(X, ref_step)
    results = {
        '__reference__': {
            'rms': ref_rms,
            'bits': ref_bits,
            'CR': 1.0,
            'Z': Xq_ref,
            'step': ref_step,
        }
    }

    # Each scheme: find the step that matches ref_rms
    for name, encode_fn in schemes.items():
        def _obj(step, _fn=encode_fn):
            r = _fn(X, step)
            return abs(r['rms'] - ref_rms)

        opt = scipy.optimize.minimize_scalar(
            _obj, bounds=(0.01, 500.0), method='bounded',
            options={'xatol': 0.01},
        )
        best_step = float(opt.x)
        enc = encode_fn(X, best_step)
        results[name] = {
            'rms': enc['rms'],
            'bits': enc['bits'],
            'CR': ref_bits / enc['bits'],
            'Z': enc['Z'],
            'step': best_step,
        }

    # Print table
    print(f"{'Scheme':<22} {'Step':>8} {'Bits':>10} {'CR':>7} {'RMS':>8}")
    print('-' * 60)
    print(f"{'Direct (ref)':<22} {ref_step:>8.2f} {ref_bits:>10.0f} {'1.000':>7} {ref_rms:>8.4f}")
    for name, r in results.items():
        if name == '__reference__':
            continue
        print(f"{name:<22} {r['step']:>8.2f} {r['bits']:>10.0f} {r['CR']:>7.3f} {r['rms']:>8.4f}")

    if plot:
        _plot_comparison(X, results, figsize=figsize)

    return results


def _plot_comparison(
    X: np.ndarray,
    results: dict,
    figsize: Optional[Tuple] = None,
) -> None:
    """Plot original, all scheme reconstructions, and error images."""
    scheme_names = [k for k in results if k != '__reference__']
    n_schemes = len(scheme_names)
    n_cols = n_schemes + 2  # original + ref + each scheme

    if figsize is None:
        figsize = (3.5 * n_cols, 7)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize)

    vmin, vmax = X.min(), X.max()
    ref = results['__reference__']

    all_imgs = [X, ref['Z']] + [results[k]['Z'] for k in scheme_names]
    all_titles = [
        'Original',
        f"Direct (ref)\nRMS={ref['rms']:.2f}",
    ] + [
        f"{k}\nCR={results[k]['CR']:.2f} RMS={results[k]['rms']:.2f}"
        for k in scheme_names
    ]

    errors = [img - X for img in all_imgs[1:]]
    err_lim = max(np.max(np.abs(e)) for e in errors)

    for col, (img, title) in enumerate(zip(all_imgs, all_titles)):
        axes[0, col].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis('off')

    axes[1, 0].axis('off')
    for col, err in enumerate(errors):
        axes[1, col + 1].imshow(err, cmap='gray', vmin=-err_lim, vmax=err_lim)
        axes[1, col + 1].set_title('error', fontsize=8)
        axes[1, col + 1].axis('off')

    plt.suptitle('Scheme comparison (matched RMS)', fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Huffman coding without DCT step (needed for DWT competition encoder)
# ---------------------------------------------------------------------------

def huffman_encode_blocks(
    Yq_int: np.ndarray,
    M: int,
    opthuff: bool = False,
    dcbits: int = 8,
) -> Tuple[np.ndarray, object]:
    """
    Apply JPEG-style run-length/Huffman coding to pre-quantised integer blocks.

    Unlike jpegenc, this function skips the DCT step — Yq_int must already be
    integer-quantised coefficients arranged in M×M blocks (use dwtgroup first
    for DWT data).

    Returns (vlc, hufftab).
    """
    scan = diagscan(M)
    dhufftab = huffdflt(1)
    _, ehuf = huffgen(dhufftab)
    huffhist = np.zeros(16 ** 2)
    H, W = Yq_int.shape

    def _encode_pass(ehuf_in):
        vlist = []
        hh = np.zeros(16 ** 2)
        for r in range(0, H, M):
            for c in range(0, W, M):
                yqflat = Yq_int[r:r+M, c:c+M].flatten('F').astype(np.int64)
                dccoef = int(yqflat[0]) + 2 ** (dcbits - 1)
                if dccoef not in range(2 ** dcbits):
                    raise ValueError('DC coefficient too large for dcbits')
                vlist.append(np.array([[dccoef, dcbits]]))
                vlist.append(huffenc(hh, runampl(yqflat[scan]), ehuf_in))
        return np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlist), hh

    vlc, huffhist = _encode_pass(ehuf)

    if not opthuff:
        return vlc, dhufftab

    dhufftab = huffdes(huffhist)
    _, ehuf2 = huffgen(dhufftab)
    vlc, _ = _encode_pass(ehuf2)
    return vlc, dhufftab


def huffman_decode_blocks(
    vlc: np.ndarray,
    M: int,
    hufftab=None,
    H: int = 256,
    W: int = 256,
    dcbits: int = 8,
) -> np.ndarray:
    """
    Decode a VLC array produced by huffman_encode_blocks back to integer blocks.

    Returns integer coefficient array of shape (H, W), NOT dequantised.
    """
    if hufftab is None:
        hufftab = huffdflt(1)

    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    huffcode_arr, _ = huffgen(hufftab)
    eob = np.array([huffgen(hufftab)[1][0]])
    run16_code = huffgen(hufftab)[1][15 * 16]
    scan = diagscan(M)
    k2 = 2 ** np.arange(17)

    # Rebuild eob / run16 as (val, bits) pairs for comparison
    from cued_sf2_lab.jpeg import huffgen as _hg
    hc, ehuf = _hg(hufftab)
    eob_pair = ehuf[0]
    run16_pair = ehuf[15 * 16]

    i = 0
    Zq = np.zeros((H, W), dtype=float)

    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M * M)
            cf = 0
            yq[cf] = vlc[i, 0] - 2 ** (dcbits - 1)
            i += 1

            while np.any(vlc[i] != eob_pair):
                run = 0
                while np.all(vlc[i] == run16_pair):
                    run += 16
                    i += 1
                start = huffstart[vlc[i, 1] - 1]
                res = hufftab.huffval[start + vlc[i, 0] - hc[start]]
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1
                ampl = vlc[i, 0]
                thr = k2[si - 1]
                yq[scan[cf - 1]] = ampl - (ampl < thr) * (2 * thr - 1)
                i += 1

            i += 1  # consume EOB
            Zq[r:r+M, c:c+M] = yq.reshape((M, M)).T

    return Zq


# ---------------------------------------------------------------------------
# JPEG Huffman coding for DWT (dwtgroup + huffman_encode_blocks)
# ---------------------------------------------------------------------------

def jpeg_encode_dwt(
    X: np.ndarray,
    n: int = 4,
    dwtstep: Optional[np.ndarray] = None,
    qstep_base: float = 17.0,
    opthuff: bool = False,
    dcbits: int = 8,
) -> dict:
    """
    Encode X with n-level DWT + JPEG run-length/Huffman coding.

    Uses dwtgroup to rearrange 2^n × 2^n coefficient blocks, then applies
    Huffman coding (without an extra DCT step, unlike jpegenc).

    dwtstep: (3, n+1) per-band step matrix (uses uniform qstep_base if None).

    Returns dict with 'vlc', 'hufftab', 'bits', 'rms', 'Z', 'n', 'dwtstep'.
    """
    H, W = X.shape
    N_block = 2 ** n

    Y = nlevdwt(X, n)

    if dwtstep is None:
        dwtstep = np.full((3, n + 1), qstep_base)

    # Integer quantise each band (standard uniform rise1=step/2, matching jpegenc convention)
    Yq_int = np.zeros_like(Y)
    for level in range(n):
        r = H >> level
        c = W >> level
        r2, c2 = r >> 1, c >> 1
        for k, rs, cs in [
            (0, slice(0, r2), slice(c2, c)),
            (1, slice(r2, r), slice(0, c2)),
            (2, slice(r2, r), slice(c2, c)),
        ]:
            s = dwtstep[k, level]
            if np.isfinite(s):
                Yq_int[rs, cs] = quant1(Y[rs, cs], s)  # default rise1=s/2
    r_lp, c_lp = H >> n, W >> n
    s_lp = dwtstep[0, n]
    if np.isfinite(s_lp):
        Yq_int[:r_lp, :c_lp] = quant1(Y[:r_lp, :c_lp], s_lp)

    # Group into N_block × N_block spatial blocks
    Yrg = dwtgroup(Yq_int.astype(int), n)

    vlc, hufftab = huffman_encode_blocks(Yrg, N_block, opthuff=opthuff, dcbits=dcbits)
    bits = int(vlc[:, 1].sum())

    # Reconstruct to measure RMS
    Z = jpeg_decode_dwt(vlc, n=n, dwtstep=dwtstep, hufftab=hufftab, H=H, W=W, dcbits=dcbits)

    return {
        'vlc': vlc,
        'hufftab': hufftab,
        'bits': bits,
        'rms': rms_error(X, Z),
        'Z': Z,
        'n': n,
        'dwtstep': dwtstep,
        'qstep_base': qstep_base,
    }


def jpeg_decode_dwt(
    vlc: np.ndarray,
    n: int = 4,
    dwtstep: Optional[np.ndarray] = None,
    qstep_base: float = 17.0,
    hufftab=None,
    H: int = 256,
    W: int = 256,
    dcbits: int = 8,
) -> np.ndarray:
    """Decode a bitstream produced by jpeg_encode_dwt()."""
    N_block = 2 ** n
    if dwtstep is None:
        dwtstep = np.full((3, n + 1), qstep_base)

    Yrg_int = huffman_decode_blocks(vlc, N_block, hufftab=hufftab, H=H, W=W, dcbits=dcbits)
    Yq_int = dwtgroup(Yrg_int.astype(int), -n)

    Yq = np.zeros_like(Yq_int, dtype=float)
    for level in range(n):
        r = H >> level
        c = W >> level
        r2, c2 = r >> 1, c >> 1
        for k, rs, cs in [
            (0, slice(0, r2), slice(c2, c)),
            (1, slice(r2, r), slice(0, c2)),
            (2, slice(r2, r), slice(c2, c)),
        ]:
            s = dwtstep[k, level]
            if np.isfinite(s):
                Yq[rs, cs] = quant2(Yq_int[rs, cs].astype(float), s)  # default rise1=s/2
    r_lp, c_lp = H >> n, W >> n
    s_lp = dwtstep[0, n]
    if np.isfinite(s_lp):
        Yq[:r_lp, :c_lp] = quant2(Yq_int[:r_lp, :c_lp].astype(float), s_lp)

    return nlevidwt(Yq, n)


# ---------------------------------------------------------------------------
# Competition: compress to a target bit budget
# ---------------------------------------------------------------------------

def compress_to_bits(
    X: np.ndarray,
    huff_encode_fn: Callable,
    target_bits: int,
    step_bounds: Tuple[float, float] = (1.0, 400.0),
    tol_frac: float = 0.02,
) -> dict:
    """
    Find the step size whose Huffman bit count is closest to target_bits.

    huff_encode_fn(X, step) must return a dict with 'bits' and 'Z' and 'rms'.
    Returns the encode dict at the optimal step.

    Bits are monotonically decreasing with step, so we binary search.
    """
    lo, hi = step_bounds

    # Ensure lo gives more bits than target and hi gives fewer
    enc_lo = huff_encode_fn(X, lo)
    enc_hi = huff_encode_fn(X, hi)

    if enc_lo['bits'] <= target_bits:
        return enc_lo  # can't hit target even at finest step
    if enc_hi['bits'] >= target_bits:
        return enc_hi  # can't hit target even at coarsest step

    best = enc_hi
    for _ in range(50):
        mid = (lo + hi) / 2.0
        enc = huff_encode_fn(X, mid)
        if abs(enc['bits'] - target_bits) < target_bits * tol_frac:
            return enc
        if enc['bits'] > target_bits:
            lo = mid
        else:
            hi = mid
        if abs(enc['bits'] - target_bits) < abs(best['bits'] - target_bits):
            best = enc

    return best


def compare_at_bits(
    X: np.ndarray,
    huff_schemes: Dict[str, Callable],
    target_bits_list: list,
    plot: bool = True,
    figsize: Optional[Tuple] = None,
) -> dict:
    """
    Compare Huffman-coded schemes at equal bit budgets — the competition metric.

    huff_schemes: {'name': encode_fn} where encode_fn(X, step) -> dict with
                  'bits', 'Z', 'rms'.
    target_bits_list: list of bit budgets to compare at.

    For each budget, finds the step that hits it for each scheme, then reports
    the RMS error and displays reconstructed images.
    """
    results = {}
    for target in target_bits_list:
        results[target] = {}
        for name, enc_fn in huff_schemes.items():
            enc = compress_to_bits(X, enc_fn, target)
            results[target][name] = {
                'bits': enc['bits'],
                'rms': enc['rms'],
                'Z': enc['Z'],
            }

    # Print summary table
    scheme_names = list(huff_schemes.keys())
    header = f"{'Budget':>10}  " + "  ".join(f"{n[:14]:>16}" for n in scheme_names)
    print(header)
    print('-' * len(header))
    for target in target_bits_list:
        row = f"{target:>10}  "
        for name in scheme_names:
            r = results[target][name]
            row += f"  {r['rms']:>6.3f}({r['bits']:>7})"
        print(row)
    print("Values: RMS(actual_bits)")

    if plot:
        _plot_compare_at_bits(X, results, huff_schemes, target_bits_list, figsize)

    return results


def _plot_compare_at_bits(X, results, huff_schemes, target_bits_list, figsize):
    """Plot reconstructions and RMS-vs-bits curves for the competition comparison."""
    scheme_names = list(huff_schemes.keys())
    n_schemes = len(scheme_names)
    n_budgets = len(target_bits_list)

    # --- RMS vs bits curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_schemes))
    for color, name in zip(colors, scheme_names):
        bits_pts = [results[t][name]['bits'] for t in target_bits_list]
        rms_pts  = [results[t][name]['rms']  for t in target_bits_list]
        ax.plot(bits_pts, rms_pts, 'o-', color=color, label=name, linewidth=1.8)
    ax.set_xlabel('Actual Huffman bits')
    ax.set_ylabel('RMS error (lower is better)')
    ax.set_title('Competition comparison: RMS vs actual bits')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # --- Image grid at each budget ---
    for target in target_bits_list:
        n_cols = n_schemes + 1
        fig_w = figsize[0] if figsize else 3.5 * n_cols
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 4))

        vmin, vmax = X.min(), X.max()
        axes[0].imshow(X, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title('Original', fontsize=9)
        axes[0].axis('off')

        for i, name in enumerate(scheme_names):
            r = results[target][name]
            axes[i + 1].imshow(r['Z'], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i + 1].set_title(
                f"{name}\n{r['bits']} bits  RMS={r['rms']:.2f}", fontsize=8
            )
            axes[i + 1].axis('off')

        plt.suptitle(f'Target budget: {target} bits', fontsize=10)
        plt.tight_layout()
        plt.show()
