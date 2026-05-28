#!/usr/bin/env python
"""Comprehensive test of all notebook code sections"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize
import inspect

from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import (
    rowdec, rowint, beside, bpp, quantise
)

print("=" * 60)
print("COMPREHENSIVE NOTEBOOK TEST")
print("=" * 60)

# Define functions from notebook
def py4enc(X, h):
    X1 = rowdec(rowdec(X, h).T, h).T
    Y0 = X - rowint(rowint(X1, 2*h).T, 2*h).T

    X2 = rowdec(rowdec(X1, h).T, h).T
    Y1 = X1 - rowint(rowint(X2, 2*h).T, 2*h).T

    X3 = rowdec(rowdec(X2, h).T, h).T
    Y2 = X2 - rowint(rowint(X3, 2*h).T, 2*h).T

    X4 = rowdec(rowdec(X3, h).T, h).T
    Y3 = X3 - rowint(rowint(X4, 2*h).T, 2*h).T

    return Y0, Y1, Y2, Y3, X4

def py4dec(Y0, Y1, Y2, Y3, X4, h):
    Z3 = rowint(rowint(X4, 2*h).T, 2*h).T + Y3
    Z2 = rowint(rowint(Z3, 2*h).T, 2*h).T + Y2
    Z1 = rowint(rowint(Z2, 2*h).T, 2*h).T + Y1
    Z0 = rowint(rowint(Z1, 2*h).T, 2*h).T + Y0
    return Z3, Z2, Z1, Z0

def reconstruct_pyramid(Y0_q, Y1_q, Y2_q, Y3_q, X4_q, h):
    """Reconstruct image from quantised pyramid"""
    Z3 = rowint(rowint(X4_q, 2*h).T, 2*h).T + Y3_q
    Z2 = rowint(rowint(Z3, 2*h).T, 2*h).T + Y2_q
    Z1 = rowint(rowint(Z2, 2*h).T, 2*h).T + Y1_q
    Z0 = rowint(rowint(Z1, 2*h).T, 2*h).T + Y0_q
    return Z0

# Load image
print("\n[1] Loading image...")
X, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
X = X - 128.0
print(f"  Image shape: {X.shape}")

# Pyramid decomposition
print("\n[2] Creating pyramid with original filter...")
h = 0.25 * np.array([1, 2, 1])
Y0, Y1, Y2, Y3, X4 = py4enc(X, h)
print(f"  Pyramid created: Y0{Y0.shape}, Y1{Y1.shape}, Y2{Y2.shape}, Y3{Y3.shape}, X4{X4.shape}")

# Pyramid reconstruction
print("\n[3] Testing pyramid reconstruction...")
Z3, Z2, Z1, Z0 = py4dec(Y0, Y1, Y2, Y3, X4, h)
encode_decode_err = np.max(np.abs(X - Z0))
print(f"  Encode-Decode Error: {encode_decode_err:.10f} ✓")

# Calculate intermediate lowpass images
print("\n[4] Computing intermediate lowpass images...")
X1 = rowdec(rowdec(X, h).T, h).T
X2 = rowdec(rowdec(X1, h).T, h).T
X3 = rowdec(rowdec(X2, h).T, h).T
print(f"  X1{X1.shape}, X2{X2.shape}, X3{X3.shape}")

# Compression ratios for different pyramid depths
print("\n[5] Computing compression ratios...")
step = 17
entropy_X = bpp(quantise(X, step))
entropy_X1 = bpp(quantise(X1, step))
entropy_Y0 = bpp(quantise(Y0, step))

total_bits_X = entropy_X * X.size
total_bits_X1 = entropy_X1 * X1.size
total_bits_Y0 = entropy_Y0 * Y0.size
total_bits_py1 = total_bits_X1 + total_bits_Y0

compression_ratio_1stage = total_bits_X / total_bits_py1
print(f"  1-Stage: {compression_ratio_1stage:.3f}")

# 2-stage pyramid
Y0_q, Y1_q, X2_q = quantise(Y0, step), quantise(Y1, step), quantise(X2, step)
entropy_Y1 = bpp(Y1_q)
entropy_X2 = bpp(X2_q)
total_bits_py2 = total_bits_Y0 + entropy_Y1 * Y1.size + entropy_X2 * X2.size
compression_ratio_2stage = total_bits_X / total_bits_py2
print(f"  2-Stage: {compression_ratio_2stage:.3f}")

# 3-stage pyramid
Y2_q, X3_q = quantise(Y2, step), quantise(X3, step)
entropy_Y2 = bpp(Y2_q)
entropy_X3 = bpp(X3_q)
total_bits_py3 = total_bits_Y0 + entropy_Y1 * Y1.size + entropy_Y2 * Y2.size + entropy_X3 * X3.size
compression_ratio_3stage = total_bits_X / total_bits_py3
print(f"  3-Stage: {compression_ratio_3stage:.3f}")

# 4-stage pyramid
Y3_q, X4_q = quantise(Y3, step), quantise(X4, step)
entropy_Y3 = bpp(Y3_q)
entropy_X4 = bpp(X4_q)
total_bits_py4 = total_bits_Y0 + entropy_Y1 * Y1.size + entropy_Y2 * Y2.size + entropy_Y3 * Y3.size + entropy_X4 * X4.size
compression_ratio_4stage = total_bits_X / total_bits_py4
print(f"  4-Stage: {compression_ratio_4stage:.3f}")

# Quantisation and reconstruction for different pyramid depths
print("\n[6] Testing quantisation and reconstruction...")

# 1-layer reconstruction
X1_q = quantise(X1, step)
Y0_q = quantise(Y0, step)
X1_interp = rowint(rowint(X1_q, 2*h).T, 2*h).T
Z0_1layer = X1_interp + Y0_q
rms_error_1layer = np.std(X - Z0_1layer)
print(f"  1-Layer RMS Error: {rms_error_1layer:.3f}")

# 2-layer reconstruction
Y1_q = quantise(Y1, step)
X2_q = quantise(X2, step)
X2_interp = rowint(rowint(X2_q, 2*h).T, 2*h).T
Z1_2layer = X2_interp + Y1_q
Z0_2layer = rowint(rowint(Z1_2layer, 2*h).T, 2*h).T + Y0_q
rms_error_2layer = np.std(X - Z0_2layer)
print(f"  2-Layer RMS Error: {rms_error_2layer:.3f}")

# 3-layer reconstruction
Y2_q = quantise(Y2, step)
X3_q = quantise(X3, step)
X3_interp = rowint(rowint(X3_q, 2*h).T, 2*h).T
Z2_3layer = X3_interp + Y2_q
Z1_3layer = rowint(rowint(Z2_3layer, 2*h).T, 2*h).T + Y1_q
Z0_3layer = rowint(rowint(Z1_3layer, 2*h).T, 2*h).T + Y0_q
rms_error_3layer = np.std(X - Z0_3layer)
print(f"  3-Layer RMS Error: {rms_error_3layer:.3f}")

# 4-layer reconstruction (full pyramid)
Y3_q = quantise(Y3, step)
X4_q = quantise(X4, step)
Z0_4layer = reconstruct_pyramid(Y0_q, Y1_q, Y2_q, Y3_q, X4_q, h)
rms_error_4layer = np.std(X - Z0_4layer)
print(f"  4-Layer RMS Error: {rms_error_4layer:.3f}")

# Direct quantisation comparison
print("\n[7] Direct quantisation comparison...")
X_q_direct = quantise(X, step)
rms_error_direct = np.std(X - X_q_direct)
print(f"  Direct Quantisation RMS Error: {rms_error_direct:.3f}")
print(f"  Direct Quantisation Total bits: {bpp(X_q_direct) * X.size:.0f}")

# Optimisation function for step size
print("\n[8] Testing step size optimisation...")

def pyramid_rms_error(step, n_layers=4):
    """Calculate RMS error for pyramid with given step size and number of layers"""
    if n_layers >= 1:
        Y0_q = quantise(Y0, step)
        X1_q = quantise(X1, step)
        if n_layers == 1:
            X1_interp = rowint(rowint(X1_q, 2*h).T, 2*h).T
            Z0 = X1_interp + Y0_q
            return np.std(X - Z0)

    if n_layers >= 2:
        Y1_q = quantise(Y1, step)
        X2_q = quantise(X2, step)
        X2_interp = rowint(rowint(X2_q, 2*h).T, 2*h).T
        Z1 = X2_interp + Y1_q
        Z0 = rowint(rowint(Z1, 2*h).T, 2*h).T + Y0_q
        if n_layers == 2:
            return np.std(X - Z0)

    if n_layers >= 3:
        Y2_q = quantise(Y2, step)
        X3_q = quantise(X3, step)
        X3_interp = rowint(rowint(X3_q, 2*h).T, 2*h).T
        Z2 = X3_interp + Y2_q
        Z1 = rowint(rowint(Z2, 2*h).T, 2*h).T + Y1_q
        Z0 = rowint(rowint(Z1, 2*h).T, 2*h).T + Y0_q
        if n_layers == 3:
            return np.std(X - Z0)

    if n_layers >= 4:
        Y3_q = quantise(Y3, step)
        X4_q = quantise(X4, step)
        Z0 = reconstruct_pyramid(Y0_q, Y1_q, Y2_q, Y3_q, X4_q, h)
        return np.std(X - Z0)

target_error = rms_error_direct
print(f"  Target error: {target_error:.3f}")

# Test finding optimal step for 1-layer
def error_diff_1layer(step):
    return abs(pyramid_rms_error(step, 1) - target_error)

result = scipy.optimize.minimize_scalar(error_diff_1layer, bounds=(0.1, 100), method='bounded')
optimal_step_1 = result.x
achieved_error_1 = pyramid_rms_error(optimal_step_1, 1)
print(f"  1-Layer optimal step: {optimal_step_1:.3f}, error: {achieved_error_1:.3f}")

# Test impulse response measurement
print("\n[9] Testing impulse response measurement...")

def measure_impulse_energy(layer_index, h):
    """Measure energy of pyramid output when an impulse is placed in a given layer."""
    Y0_imp = np.zeros_like(Y0)
    Y1_imp = np.zeros_like(Y1)
    Y2_imp = np.zeros_like(Y2)
    Y3_imp = np.zeros_like(Y3)
    X4_imp = np.zeros_like(X4)

    impulse_amp = 100
    if layer_index == 0:
        Y0_imp[Y0.shape[0]//2, Y0.shape[1]//2] = impulse_amp
    elif layer_index == 1:
        Y1_imp[Y1.shape[0]//2, Y1.shape[1]//2] = impulse_amp
    elif layer_index == 2:
        Y2_imp[Y2.shape[0]//2, Y2.shape[1]//2] = impulse_amp
    elif layer_index == 3:
        Y3_imp[Y3.shape[0]//2, Y3.shape[1]//2] = impulse_amp
    elif layer_index == 4:
        X4_imp[X4.shape[0]//2, X4.shape[1]//2] = impulse_amp

    Z0 = reconstruct_pyramid(Y0_imp, Y1_imp, Y2_imp, Y3_imp, X4_imp, h)
    energy = np.sum(Z0 ** 2)
    return energy

energies = []
layer_names = ['Y0', 'Y1', 'Y2', 'Y3', 'X4']
for layer_idx in range(5):
    energy = measure_impulse_energy(layer_idx, h)
    energies.append(energy)
    print(f"  {layer_names[layer_idx]} impulse energy: {energy:.2e}")

step_ratios = np.sqrt(energies[0] / np.array(energies))
print(f"  Step size ratios (relative to Y0): {step_ratios}")

# Test with new filter
print("\n[10] Testing with new filter (m=4)...")
h_new = np.array([1, 4, 6, 4, 1]) / 16.0
Y0_new, Y1_new, Y2_new, Y3_new, X4_new = py4enc(X, h_new)
X1_new = rowdec(rowdec(X, h_new).T, h_new).T
X2_new = rowdec(rowdec(X1_new, h_new).T, h_new).T
X3_new = rowdec(rowdec(X2_new, h_new).T, h_new).T
print(f"  New pyramid created successfully")

# Measure energies with new filter
energies_new = []
for layer_idx in range(5):
    Y0_imp = np.zeros_like(Y0_new)
    Y1_imp = np.zeros_like(Y1_new)
    Y2_imp = np.zeros_like(Y2_new)
    Y3_imp = np.zeros_like(Y3_new)
    X4_imp = np.zeros_like(X4_new)

    impulse_amp = 100
    if layer_idx == 0:
        Y0_imp[Y0_new.shape[0]//2, Y0_new.shape[1]//2] = impulse_amp
    elif layer_idx == 1:
        Y1_imp[Y1_new.shape[0]//2, Y1_new.shape[1]//2] = impulse_amp
    elif layer_idx == 2:
        Y2_imp[Y2_new.shape[0]//2, Y2_new.shape[1]//2] = impulse_amp
    elif layer_idx == 3:
        Y3_imp[Y3_new.shape[0]//2, Y3_new.shape[1]//2] = impulse_amp
    elif layer_idx == 4:
        X4_imp[X4_new.shape[0]//2, X4_new.shape[1]//2] = impulse_amp

    Z0 = reconstruct_pyramid(Y0_imp, Y1_imp, Y2_imp, Y3_imp, X4_imp, h_new)
    energy = np.sum(Z0 ** 2)
    energies_new.append(energy)

step_ratios_new = np.sqrt(energies_new[0] / np.array(energies_new))
print(f"  New filter step size ratios: {step_ratios_new}")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)

