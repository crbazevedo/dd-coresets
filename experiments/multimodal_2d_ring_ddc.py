#!/usr/bin/env python
"""
2D multimodal experiment: 3 Gaussians + ring, visual DDC demo.

Usage:
    python experiments/multimodal_2d_ring_ddc.py
"""

import numpy as np
import matplotlib.pyplot as plt

from dd_coresets.ddc import fit_ddc_coreset


# ---------- Geração dos dados 2D (3 Gaussians + ring) ---------- #

def generate_2d_multimodal(n=8000, random_state=0):
    rng = np.random.default_rng(random_state)

    # 3 Gaussian blobs
    means = np.array([[0, 0], [4, 0], [0, 4]])
    covs = [
        np.array([[1.0, 0.2], [0.2, 0.5]]),
        np.array([[0.5, -0.1], [-0.1, 0.8]]),
        np.array([[0.7, 0.0], [0.0, 0.7]]),
    ]
    n_blob = n // 2
    X_blobs = []
    for m, C in zip(means, covs):
        X_blobs.append(rng.multivariate_normal(m, C, size=n_blob // 3))
    X_blobs = np.vstack(X_blobs)

    # Noisy ring
    n_ring = n - len(X_blobs)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_ring)
    radius = 5.0 + rng.normal(0.0, 0.2, size=n_ring)
    X_ring = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)],
        axis=1,
    )

    X = np.vstack([X_blobs, X_ring])
    rng.shuffle(X)
    return X


# ---------- Main ---------- #

def main():
    # Dados
    X = generate_2d_multimodal(n=8000, random_state=0)

    # Fit DDC coreset
    S, w, info = fit_ddc_coreset(
        X,
        k=80,
        n0=None,          # usa todos os pontos
        m_neighbors=25,
        alpha=0.3,
        gamma=1.0,
        refine_iters=1,
        reweight_full=False,  # aqui só precisa do working sample
        random_state=13,
    )

    # --- Plot 1: dados + reps (tamanho ∝ peso) ---
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=4, alpha=0.2, label="data")
    sizes = 200 * (w / w.max())
    plt.scatter(
        S[:, 0],
        S[:, 1],
        s=sizes,
        edgecolors="black",
        label="DDC reps",
    )
    plt.legend()
    plt.title("2D multimodal data + DDC representatives")
    plt.tight_layout()

    # --- Plot 2: marginais ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for dim, ax in enumerate(axes):
        ax.hist(X[:, dim], bins=40, density=True, alpha=0.4, label="full")
        ax.hist(
            S[:, dim],
            bins=40,
            weights=w,
            density=True,
            histtype="step",
            linewidth=2,
            label="DDC",
        )
        ax.set_title(f"Marginal dim {dim}")
        ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
