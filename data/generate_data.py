#!/usr/bin/env python3
"""
Raw Numerical Dataset Generator and Analytical Reference for Distributed Vector Processing
Parallel and GPU Computing Laboratory - Team 5
"""

import argparse
import sys
import numpy as np

def generate_analytical_reference(N, alpha=2.5, beta=1.5):
    """
    Computes exact analytical results for validation against C/MPI implementations.
    Uses vectorization with NumPy to verify outputs.
    """
    indices = np.arange(N, dtype=np.float64) % 1000 * 0.01
    X = np.sin(indices) + 1.5
    Y = np.cos(indices) + 2.0

    # Transformations
    Z = alpha * X + beta * Y
    W = np.sqrt(X**2 + Y**2) + np.sin(X) + np.cos(Y)

    # Reductions
    dot_product = np.dot(X, Y)
    l2_norm = np.linalg.norm(X)
    z_sum = np.sum(Z)
    w_min = np.min(W)
    w_max = np.max(W)

    print("=================================================================")
    print(" Python / NumPy Ground Truth Reference")
    print("=================================================================")
    print(f" Vector Size (N)   : {N:,} elements")
    print(f" Working Memory    : {(N * 8 * 4) / (1024 * 1024):.2f} MB")
    print(f" Scalar Alpha/Beta : {alpha:.2f} / {beta:.2f}")
    print("-----------------------------------------------------------------")
    print(f" Verification Samples:")
    print(f"   Z[0]     = {Z[0]:.8f}, Z[N-1] = {Z[-1]:.8f}")
    print(f"   W[0]     = {W[0]:.8f}, W[N-1] = {W[-1]:.8f}")
    print(f"   Dot Product = {dot_product:.8f}")
    print(f"   L2 Norm (X) = {l2_norm:.8f}")
    print(f"   Z Sum       = {z_sum:.8f}")
    print(f"   W Min / Max = {w_min:.8f} / {w_max:.8f}")
    print("=================================================================")

def main():
    parser = argparse.ArgumentParser(description="Raw Vector Dataset Tool")
    parser.add_argument("--size", "-n", type=int, default=10000000, help="Vector size N")
    parser.add_argument("--alpha", type=float, default=2.5, help="Scalar alpha")
    parser.add_argument("--beta", type=float, default=1.5, help="Scalar beta")
    parser.add_argument("--verify-only", action="store_true", help="Print reference analytical metrics")
    args = parser.parse_args()

    generate_analytical_reference(args.size, args.alpha, args.beta)

if __name__ == "__main__":
    main()
