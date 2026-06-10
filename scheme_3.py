"""
Scheme 3: 4-level DWT with equal-MSE per-subband steps + HH suppression
"""

import numpy as np
import scipy.optimize

from helpers import (
    nlevdwt, nlevidwt, quantdwt, dwt_total_bits, dwt_equal_mse_steps,
    bpp, rms_error, direct_quantisation_reference,
    compare_schemes, load_image,
)


_N_LEVELS = 4
_SUPPRESS_HH = True

_cached_ratios = {}


def _get_ratios(shape: tuple) -> np.ndarray:
    if shape not in _cached_ratios:
        rows, cols = shape
        n = _N_LEVELS
        energies = np.zeros((3, n + 1))

        for level in range(n):
            r = rows >> level
            c = cols >> level
            r2 = r >> 1
            c2 = c >> 1
            bands = [
                (0, level, slice(0, r2), slice(c2, c)),
                (1, level, slice(r2, r), slice(0, c2)),
                (2, level, slice(r2, r), slice(c2, c)),
            ]
            for k, lev, rs, cs in bands:
                Yimp = np.zeros((rows, cols))
                Yimp[(rs.start + rs.stop) // 2, (cs.start + cs.stop) // 2] = 100.0
                Zimp = nlevidwt(Yimp, n)
                energies[k, lev] = np.sum(Zimp ** 2)

        r_lp = rows >> n
        c_lp = cols >> n
        Yimp = np.zeros((rows, cols))
        Yimp[r_lp // 2, c_lp // 2] = 100.0
        energies[0, n] = np.sum(nlevidwt(Yimp, n) ** 2)

        ref = energies[0, n]
        ratios = np.sqrt(ref / np.maximum(energies, 1e-12))
        _cached_ratios[shape] = ratios

    return _cached_ratios[shape]


def encode(X: np.ndarray, step: float) -> dict:
    n = _N_LEVELS
    ratios = _get_ratios(X.shape)
    dwtstep = step * ratios

    if _SUPPRESS_HH:
        dwtstep[2, 0] = np.inf

    Y = nlevdwt(X, n)
    Yq, dwtent = quantdwt(Y, dwtstep, rise1=None)
    Z = nlevidwt(Yq, n)
    bits = dwt_total_bits(dwtent, X.shape, n)

    return {
        'Yq': Yq,
        'dwtent': dwtent,
        'dwtstep': dwtstep,
        'Z': Z,
        'bits': bits,
        'rms': rms_error(X, Z),
        'step': step,
        'n': n,
    }


def decode(enc_result: dict) -> np.ndarray:
    return nlevidwt(enc_result['Yq'], enc_result['n'])


if __name__ == '__main__':
    X = load_image('lighthouse.mat')
    _, ref_rms, ref_bits = direct_quantisation_reference(X, 17.0)

    print('Scheme 3: 4-level DWT, equal-MSE steps, HH-suppressed, rise1=step')
    print(f'Reference RMS={ref_rms:.3f}  bits={ref_bits:.0f}')
    print()

    result = compare_schemes(
        X,
        {'DWT 4-lvl eq-MSE': encode},
        ref_step=17.0,
        plot=True,
    )
