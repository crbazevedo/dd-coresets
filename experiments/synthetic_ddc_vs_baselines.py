#!/usr/bin/env python
"""
Synthetic experiment for Density–Diversity Coresets (DDC)
vs. Random and Stratified baselines on a 5D Gaussian mixture.

Usage:
    python experiments/synthetic_ddc_vs_baselines.py

Author: Carlos R. B. Azevedo
Date: 2025-11-11
License: MIT
Version: 1.0.0
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dd_coresets.ddc import (
    fit_ddc_coreset,
    fit_random_coreset,
    fit_stratified_coreset,
)


# ---------------- Metrics ---------------- #

def weighted_mean(S, w):
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    return (S * w[:, None]).sum(axis=0)


def weighted_cov(S, w):
    """
    Weighted covariance given support S and weights w (sum to 1).
    """
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
    """
    Approximate 1D Wasserstein-1 between X_dim and weighted S_dim.
    """
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
    """
    Approximate Kolmogorov–Smirnov statistic between full X and discrete S.
    """
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


def evaluate_representation(name, S, w, X_full, metrics_random_seeds=(101, 102, 103, 104, 105)):
    """
    Compute metrics comparing coreset (S, w) to full data X_full.
    """
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


# ---------------- Synthetic data ---------------- #

def generate_synthetic_mixture(n=50000, d=5, random_state=7):
    rng = np.random.default_rng(random_state)
    n_components = 4
    mix_weights = np.array([0.3, 0.3, 0.2, 0.2])
    mix_weights /= mix_weights.sum()

    means = np.array([
        np.zeros(d),
        np.ones(d) * 3.0,
        np.concatenate([np.array([5.0, -3.0]), np.zeros(d - 2)]),
        np.concatenate([np.array([-4.0, 4.0]), np.ones(d - 2)])
    ])

    covs = [
        np.diag([1.0, 0.5, 1.5, 0.8, 1.2]),
        np.diag([0.7, 1.2, 0.6, 1.0, 0.9]),
        np.diag([1.5, 0.7, 0.7, 0.7, 1.0]),
        np.diag([0.5, 0.5, 1.3, 1.3, 0.6])
    ]

    comp_idx = rng.choice(n_components, size=n, p=mix_weights)
    X = np.zeros((n, d), dtype=float)

    for k_comp in range(n_components):
        idx = np.where(comp_idx == k_comp)[0]
        size = len(idx)
        if size == 0:
            continue
        X[idx] = rng.multivariate_normal(means[k_comp], covs[k_comp], size=size)

    return X, comp_idx


# ---------------- Main experiment ---------------- #

def run_experiment():
    X, comp_idx = generate_synthetic_mixture(n=50000, d=5, random_state=7)

    configs = [
        {"k": 50, "alpha": 0.3},
        {"k": 100, "alpha": 0.3},
    ]

    rows = []

    for cfg in configs:
        k_rep = cfg["k"]
        alpha = cfg["alpha"]

        # ---- DDC ----
        t0 = time.time()
        S_ddc, w_ddc, info_ddc = fit_ddc_coreset(
            X,
            k=k_rep,
            n0=20000,
            m_neighbors=32,
            alpha=alpha,
            gamma=1.0,
            refine_iters=1,
            reweight_full=True,
            random_state=13,
        )
        t1 = time.time()

        res_ddc = evaluate_representation(
            f"DDC(k={k_rep},α={alpha})", S_ddc, w_ddc, X
        )
        res_ddc["runtime_sec"] = t1 - t0
        rows.append(res_ddc)

        # ---- Random ----
        t0 = time.time()
        S_rnd, w_rnd, info_rnd = fit_random_coreset(
            X,
            k=k_rep,
            n0=20000,
            gamma=1.0,
            reweight_full=True,
            random_state=123,
        )
        t1 = time.time()

        res_rnd = evaluate_representation(
            f"Random(k={k_rep})", S_rnd, w_rnd, X
        )
        res_rnd["runtime_sec"] = t1 - t0
        rows.append(res_rnd)

        # ---- Stratified ----
        t0 = time.time()
        S_strat, w_strat, info_strat = fit_stratified_coreset(
            X,
            strata=comp_idx,
            k=k_rep,
            n0=20000,
            gamma=1.0,
            reweight_full=True,
            random_state=321,
        )
        t1 = time.time()

        res_strat = evaluate_representation(
            f"Stratified(k={k_rep})", S_strat, w_strat, X
        )
        res_strat["runtime_sec"] = t1 - t0
        rows.append(res_strat)

    results_df = pd.DataFrame(rows)
    print("\n=== Coreset comparison: DDC vs Random vs Stratified ===\n")
    print(results_df[[
        "method",
        "mean_err_l2",
        "cov_err_fro",
        "corr_err_fro",
        "W1_mean",
        "W1_max",
        "KS_mean",
        "KS_max",
        "runtime_sec"
    ]].to_string(index=False))

    # Simple plots for quick inspection
    plt.figure(figsize=(7, 4))
    x = np.arange(len(results_df))
    plt.bar(x, results_df["W1_mean"].values)
    plt.xticks(x, results_df["method"].values, rotation=30, ha="right")
    plt.ylabel("W1_mean")
    plt.title("W1_mean by method/config")
    plt.tight_layout()

    plt.figure(figsize=(6, 5))
    plt.scatter(results_df["mean_err_l2"].values, results_df["cov_err_fro"].values)
    for i, txt in enumerate(results_df["method"].values):
        plt.annotate(txt,
                     (results_df["mean_err_l2"].values[i],
                      results_df["cov_err_fro"].values[i]),
                     fontsize=8)
    plt.xlabel("Mean error (L2)")
    plt.ylabel("Cov error (Frobenius)")
    plt.title("Moment errors: mean vs covariance")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_experiment()
