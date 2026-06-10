"""
Scheme 1: LBT N=4, s=sqrt(2), M=16, rise1=step
"""

import numpy as np
from helpers import (
    lbt_encode, lbt_decode,
    jpeg_encode_lbt, jpeg_decode_lbt,
    rms_error, direct_quantisation_reference,
    compare_schemes, load_image,
)

_N = 4
_S = np.sqrt(2)
_M = 16


def encode(X: np.ndarray, step: float) -> dict:
    result = lbt_encode(X, N=_N, s=_S, step=step, rise1=step, coding_N=_M)
    result['step'] = step
    return result


def decode(enc_result: dict) -> np.ndarray:
    return lbt_decode(enc_result['Yq'], N=_N, s=_S, Pr=enc_result['Pr'])


def encode_huffman(X: np.ndarray, qstep: float, opthuff: bool = True) -> dict:
    result = jpeg_encode_lbt(X, qstep, N=_N, s=_S, M=_M, opthuff=opthuff)
    return result


def decode_huffman(enc_result: dict, W: int = 256, H: int = 256) -> np.ndarray:
    return jpeg_decode_lbt(
        enc_result['vlc'], enc_result['qstep'],
        N=_N, s=_S, M=_M,
        hufftab=enc_result.get('hufftab'),
        Pr=enc_result.get('Pr'),
        W=W, H=H,
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = load_image('lighthouse.mat')
    _, ref_rms, ref_bits = direct_quantisation_reference(X, 17.0)

    print('Scheme 1: LBT N=4, s=sqrt(2), rise1=step')
    print(f'Reference RMS={ref_rms:.3f}  bits={ref_bits:.0f}')
    print()

    result = compare_schemes(
        X,
        {'LBT N=4 s=√2': encode},
        ref_step=17.0,
        plot=True,
    )

    import scipy.optimize

    def _obj(step):
        enc = encode(X, step)
        return abs(enc['rms'] - ref_rms)
    opt = scipy.optimize.minimize_scalar(_obj, bounds=(0.01, 100), method='bounded')
    best_step = float(opt.x)

    enc_h = encode_huffman(X, best_step, opthuff=True)
    Z_h = decode_huffman(enc_h)
    print(f'\nHuffman actual bits: {enc_h["bits"]}  '
          f'RMS: {rms_error(X, Z_h):.4f}  '
          f'CR: {ref_bits / enc_h["bits"]:.3f}')
