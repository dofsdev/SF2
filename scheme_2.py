"""
Scheme 2: DCT N=8 with per-subband equal-MSE weighting + high-freq suppression
"""

import numpy as np
import scipy.optimize

from helpers import (
    dct_ii, colxfm, regroup,
    quantise, bpp, dctbpp,
    rms_error, direct_quantisation_reference,
    compare_schemes, load_image,
)

_N = 8
_CODING_N = 8


def _dct_basis_energies(N: int) -> np.ndarray:
    C = dct_ii(N)
    energies = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            imp = np.zeros((N, N))
            imp[u, v] = 1.0
            basis = C.T @ imp @ C
            energies[u, v] = np.sum(basis ** 2)
    return energies


def _build_step_matrix(base_step: float, suppress_corner: bool = True) -> np.ndarray:
    N = _N
    energies = _dct_basis_energies(N)
    ratios = np.sqrt(energies[0, 0] / np.maximum(energies, 1e-12))
    mat = base_step * ratios

    if suppress_corner:
        for u in range(N - 2, N):
            for v in range(N - 2, N):
                mat[u, v] = np.inf
    return mat


def encode(X: np.ndarray, step: float) -> dict:
    N = _N
    C = dct_ii(N)

    Y = colxfm(colxfm(X, C).T, C).T
    Yr = regroup(Y, N)

    step_matrix = _build_step_matrix(step, suppress_corner=True)

    H, W = X.shape
    sub_h = H // N
    sub_w = W // N

    Yq_r = np.zeros_like(Yr)
    for u in range(N):
        for v in range(N):
            s = step_matrix[u, v]
            if not np.isfinite(s):
                continue
            sub = Yr[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w]
            Yq_r[u * sub_h:(u + 1) * sub_h, v * sub_w:(v + 1) * sub_w] = quantise(sub, s, s)

    Yq = regroup(Yq_r, [sub_h, sub_w])
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)

    bits = dctbpp(Yq_r, _CODING_N)

    return {
        'Yq': Yq,
        'Yq_r': Yq_r,
        'Z': Z,
        'bits': bits,
        'rms': rms_error(X, Z),
        'step': step,
        'step_matrix': step_matrix,
        'N': N,
    }


def decode(enc_result: dict) -> np.ndarray:
    N = enc_result['N']
    C = dct_ii(N)
    return colxfm(colxfm(enc_result['Yq'].T, C.T).T, C.T)


if __name__ == '__main__':
    X = load_image('lighthouse.mat')
    _, ref_rms, ref_bits = direct_quantisation_reference(X, 17.0)

    print('Scheme 2: DCT N=8, equal-MSE weights, corner suppression, rise1=step')
    print(f'Reference RMS={ref_rms:.3f}  bits={ref_bits:.0f}')
    print()

    result = compare_schemes(
        X,
        {'DCT N=8 eq-MSE': encode},
        ref_step=17.0,
        plot=True,
    )
