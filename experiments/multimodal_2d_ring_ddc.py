#!/usr/bin/env python
"""
2D multimodal experiment: 3 Gaussians + ring, visual DDC demo.
Compares DDC vs Random coresets.

Usage:
    python experiments/multimodal_2d_ring_ddc.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from dd_coresets.ddc import fit_ddc_coreset, fit_random_coreset


# ---------------- Metrics ---------------- #

def weighted_mean(S, w):
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    return (S * w[:, None]).sum(axis=0)


def weighted_cov(S, w):
    """Weighted covariance given support S and weights w (sum to 1)."""
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    mu = weighted_mean(S, w)
    Xc = S - mu
    cov = (Xc * w[:, None]).T @ Xc
    return cov


def corr_from_cov(cov):
    cov = np.asarray(cov, dtype=float)
    d = cov.shape[0]
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv_std = 1.0 / std
    C = cov * inv_std[:, None] * inv_std[None, :]
    return C


def wasserstein_1d_approx(X_dim, S_dim, w, n_samples=5000, random_state=None):
    """Approximate 1D Wasserstein-1 between X_dim and weighted S_dim."""
    rng = np.random.default_rng(random_state)
    X_dim = np.asarray(X_dim, dtype=float)
    S_dim = np.asarray(S_dim, dtype=float)
    w = np.asarray(w, dtype=float)

    n = len(X_dim)
    k = len(S_dim)

    idx_x = rng.integers(0, n, size=n_samples)
    x_samp = X_dim[idx_x]

    idx_s = rng.choice(k, size=n_samples, p=w)
    y_samp = S_dim[idx_s]

    x_sorted = np.sort(x_samp)
    y_sorted = np.sort(y_samp)
    return float(np.mean(np.abs(x_sorted - y_sorted)))


def ks_1d_approx(X_dim, S_dim, w, n_grid=512):
    """Approximate Kolmogorov–Smirnov statistic between full X and discrete S."""
    X_dim = np.asarray(X_dim, dtype=float)
    S_dim = np.asarray(S_dim, dtype=float)
    w = np.asarray(w, dtype=float)

    lo = min(X_dim.min(), S_dim.min())
    hi = max(X_dim.max(), S_dim.max())
    grid = np.linspace(lo, hi, n_grid)

    X_sorted = np.sort(X_dim)
    F_X = np.searchsorted(X_sorted, grid, side="right") / len(X_dim)

    F_S = np.array([w[(S_dim <= t)].sum() for t in grid])

    return float(np.max(np.abs(F_X - F_S)))


def evaluate_representation(name, S, w, X_full, metrics_random_seeds=(101, 102)):
    """Compute metrics comparing coreset (S, w) to full data X_full."""
    X_full = np.asarray(X_full, dtype=float)
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)

    mu_full = X_full.mean(axis=0)
    cov_full = np.cov(X_full, rowvar=False)

    mu_core = weighted_mean(S, w)
    cov_core = weighted_cov(S, w)

    mean_err_l2 = float(np.linalg.norm(mu_full - mu_core))
    cov_err_fro = float(np.linalg.norm(cov_full - cov_core, ord="fro"))

    corr_full = corr_from_cov(cov_full)
    corr_core = corr_from_cov(cov_core)
    corr_err_fro = float(np.linalg.norm(corr_full - corr_core, ord="fro"))

    d = X_full.shape[1]
    W1_dims, KS_dims = [], []
    for dim, seed in zip(range(d), metrics_random_seeds):
        W1 = wasserstein_1d_approx(X_full[:, dim], S[:, dim], w,
                                   n_samples=5000, random_state=seed)
        KS = ks_1d_approx(X_full[:, dim], S[:, dim], w, n_grid=512)
        W1_dims.append(float(W1))
        KS_dims.append(float(KS))

    res = {
        "method": name,
        "mean_err_l2": mean_err_l2,
        "cov_err_fro": cov_err_fro,
        "corr_err_fro": corr_err_fro,
        "W1_mean": float(np.mean(W1_dims)),
        "W1_max": float(np.max(W1_dims)),
        "KS_mean": float(np.mean(KS_dims)),
        "KS_max": float(np.max(KS_dims)),
    }

    for i, (w1, ks) in enumerate(zip(W1_dims, KS_dims)):
        res[f"W1_dim{i}"] = w1
        res[f"KS_dim{i}"] = ks

    return res


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
    # Output directory
    output_dir = Path(__file__).parent.parent / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dados
    X = generate_2d_multimodal(n=8000, random_state=0)
    k = 80

    # Fit DDC coreset
    S_ddc, w_ddc, info_ddc = fit_ddc_coreset(
        X,
        k=k,
        n0=None,          # usa todos os pontos
        m_neighbors=25,
        alpha=0.3,
        gamma=1.0,
        refine_iters=1,
        reweight_full=False,
        random_state=13,
    )

    # Fit Random coreset (mesmos parâmetros)
    S_rnd, w_rnd, info_rnd = fit_random_coreset(
        X,
        k=k,
        n0=None,          # usa todos os pontos
        gamma=1.0,
        reweight_full=False,
        random_state=42,
    )

    # --- Plot 1: Comparação DDC vs Random (scatter) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # DDC
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.2, c="gray", label="data")
    sizes_ddc = 200 * (w_ddc / w_ddc.max())
    ax.scatter(
        S_ddc[:, 0],
        S_ddc[:, 1],
        s=sizes_ddc,
        c="red",
        edgecolors="black",
        linewidth=0.5,
        label="DDC reps",
        zorder=10,
    )
    ax.set_title("DDC Coreset", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    # Random
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=0.2, c="gray", label="data")
    sizes_rnd = 200 * (w_rnd / w_rnd.max())
    ax.scatter(
        S_rnd[:, 0],
        S_rnd[:, 1],
        s=sizes_rnd,
        c="blue",
        edgecolors="black",
        linewidth=0.5,
        label="Random reps",
        zorder=10,
    )
    ax.set_title("Random Coreset", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("2D Multimodal Data: DDC vs Random Coresets (k=80)", 
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "ddc_vs_random_scatter.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'ddc_vs_random_scatter.png'}")
    plt.close()

    # --- Plot 2: Marginais DDC ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for dim, ax in enumerate(axes):
        ax.hist(X[:, dim], bins=40, density=True, alpha=0.4, 
                label="Full data", color="gray")
        ax.hist(
            S_ddc[:, dim],
            bins=40,
            weights=w_ddc,
            density=True,
            histtype="step",
            linewidth=2,
            label="DDC coreset",
            color="red",
        )
        ax.set_title(f"Marginal dim {dim}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("DDC: Distributional Approximation (Marginals)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_dir / "ddc_marginals.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'ddc_marginals.png'}")
    plt.close()

    # --- Plot 3: Marginais Random ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for dim, ax in enumerate(axes):
        ax.hist(X[:, dim], bins=40, density=True, alpha=0.4, 
                label="Full data", color="gray")
        ax.hist(
            S_rnd[:, dim],
            bins=40,
            weights=w_rnd,
            density=True,
            histtype="step",
            linewidth=2,
            label="Random coreset",
            color="blue",
        )
        ax.set_title(f"Marginal dim {dim}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Random: Distributional Approximation (Marginals)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_dir / "random_marginals.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'random_marginals.png'}")
    plt.close()

    # --- Compute metrics ---
    print("\nComputing metrics...")
    res_ddc = evaluate_representation("DDC", S_ddc, w_ddc, X, metrics_random_seeds=(101, 102))
    res_rnd = evaluate_representation("Random", S_rnd, w_rnd, X, metrics_random_seeds=(101, 102))
    
    # Create comparison table
    comparison_df = pd.DataFrame([res_ddc, res_rnd])
    
    # Select key metrics for display
    display_cols = [
        "method",
        "mean_err_l2",
        "cov_err_fro",
        "corr_err_fro",
        "W1_mean",
        "W1_max",
        "KS_mean",
        "KS_max",
    ]
    
    print("\n" + "="*70)
    print("DDC vs Random: Quantitative Comparison (2D Multimodal Dataset)")
    print("="*70)
    print(comparison_df[display_cols].to_string(index=False))
    print("="*70)
    
    # Save comparison table to CSV
    csv_path = output_dir / "comparison_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to: {csv_path}")
    
    # Save formatted table as text
    txt_path = output_dir / "comparison_metrics.txt"
    with open(txt_path, "w") as f:
        f.write("DDC vs Random: Quantitative Comparison\n")
        f.write("="*70 + "\n")
        f.write("Dataset: 2D Multimodal (3 Gaussians + Ring, n=8000)\n")
        f.write("Parameters: k=80, n0=None\n")
        f.write("="*70 + "\n\n")
        f.write(comparison_df[display_cols].to_string(index=False))
        f.write("\n\n")
        f.write("Metrics:\n")
        f.write("  - mean_err_l2: L2 norm of mean difference\n")
        f.write("  - cov_err_fro: Frobenius norm of covariance difference\n")
        f.write("  - corr_err_fro: Frobenius norm of correlation difference\n")
        f.write("  - W1_mean: Mean Wasserstein-1 distance across dimensions\n")
        f.write("  - W1_max: Maximum Wasserstein-1 distance across dimensions\n")
        f.write("  - KS_mean: Mean Kolmogorov-Smirnov statistic across dimensions\n")
        f.write("  - KS_max: Maximum Kolmogorov-Smirnov statistic across dimensions\n")
    print(f"✅ Formatted table saved to: {txt_path}")
    
    print("\n✅ All figures and metrics saved successfully!")
    
    return comparison_df


if __name__ == "__main__":
    main()
