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
from sklearn.neighbors import NearestNeighbors


# ---------------- Basic utilities ---------------- #

def pairwise_sq_dists(X, Y=None):
    """
    Squared Euclidean distances between rows of X and Y.
    Returns matrix D2 of shape (n_X, n_Y).
    """
    X = np.asarray(X, dtype=float)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)

    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    D2 = XX + YY - 2.0 * (X @ Y.T)
    return np.maximum(D2, 0.0)


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


# ---------------- DDC components ---------------- #

def density_knn(X, m_neighbors=32):
    """
    kNN-based local density proxy.
    p_i ∝ 1 / r_k(x_i)^d, normalised to sum to 1.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    m = min(m_neighbors + 1, max(2, n))  # +1 to include self

    nn = NearestNeighbors(n_neighbors=m, algorithm="ball_tree")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)

    rk = dists[:, -1]
    rk = np.maximum(rk, 1e-12)
    p = 1.0 / (rk ** d)
    p /= p.sum()
    return p


def select_reps_greedy(X, p, k, alpha=0.3, random_state=None):
    """
    Greedy density–diversity selection in O(k * n * d).

    X: (n, d) working sample
    p: density scores, sum to 1
    k: number of representatives
    alpha: density–diversity trade-off (0 ≈ diversity, 1 ≈ density)
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)

    n, d = X.shape
    if k >= n:
        return np.arange(n, dtype=int)

    selected = np.empty(k, dtype=int)

    # First representative: highest density
    j0 = int(np.argmax(p))
    selected[0] = j0

    diff = X - X[j0]
    min_dist = np.linalg.norm(diff, axis=1)

    for t in range(1, k):
        last = selected[t - 1]
        diff = X - X[last]
        new_dist = np.linalg.norm(diff, axis=1)
        min_dist = np.minimum(min_dist, new_dist)

        scores = min_dist * (p ** alpha)
        scores[selected[:t]] = -np.inf  # avoid reselection
        j_next = int(np.argmax(scores))
        selected[t] = j_next

    return selected


def soft_assign_weights(X, S, gamma=1.0):
    """
    Soft assignments via a Gaussian kernel and resulting weights.

    X: (n, d) data
    S: (k, d) representatives
    gamma: multiplier for median distance scale
    Returns: weights w (k,), assignments A (n, k)
    """
    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)
    D2 = pairwise_sq_dists(X, S)

    med = np.median(D2)
    if med <= 0:
        med = 1.0
    sigma2 = gamma * med

    K = np.exp(-D2 / (2.0 * sigma2))
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    A = K / row_sums

    w = A.mean(axis=0)
    w = np.maximum(w, 1e-18)
    w = w / w.sum()
    return w, A


def medoid_refinement(X, selected_idx, A, max_iters=1):
    """
    One or a few medoid refinement iterations.

    X: (n0, d)
    selected_idx: (k,) indices of representatives in X
    A: (n0, k) assignments from soft_assign_weights
    """
    X = np.asarray(X, dtype=float)
    selected_idx = np.asarray(selected_idx, dtype=int)
    n, d = X.shape
    k = len(selected_idx)

    for _ in range(max_iters):
        # Hard cluster from soft assignments
        C = np.argmax(A, axis=1)
        changed = False

        for j in range(k):
            idx_cluster = np.where(C == j)[0]
            if idx_cluster.size == 0:
                continue

            Xc = X[idx_cluster]
            D2_local = pairwise_sq_dists(Xc)
            mean_dist = np.sqrt(D2_local).mean(axis=1)

            best_local = int(np.argmin(mean_dist))
            new_idx = idx_cluster[best_local]

            if new_idx != selected_idx[j]:
                changed = True
            selected_idx[j] = new_idx

        S = X[selected_idx]
        w, A = soft_assign_weights(X, S)
        if not changed:
            break

    S = X[selected_idx]
    return selected_idx, S, w, A


# ---------------- Metrics ---------------- #

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


def build_working_sample(X, comp_idx, n0=20000, random_state=0):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx_work = rng.choice(n, size=n0, replace=False)
    X0 = X[idx_work]
    comp0 = comp_idx[idx_work]
    return X0, comp0


# ---------------- Baselines ---------------- #

def build_random_coreset(X0, X_full, k, gamma=1.0, random_state=123):
    rng = np.random.default_rng(random_state)
    n0 = X0.shape[0]
    idx_rnd = rng.choice(n0, size=k, replace=False)
    S_rnd = X0[idx_rnd]
    w_rnd, _ = soft_assign_weights(X_full, S_rnd, gamma=gamma)
    return S_rnd, w_rnd


def build_stratified_coreset(X0, comp0, X_full, k,
                             n_components=4, gamma=1.0, random_state=321):
    rng = np.random.default_rng(random_state)
    X0 = np.asarray(X0, dtype=float)
    comp0 = np.asarray(comp0, dtype=int)

    counts = np.bincount(comp0, minlength=n_components).astype(float)
    props = counts / counts.sum()

    alloc = np.floor(props * k).astype(int)

    # Ajuste por arredondamento
    while alloc.sum() < k:
        residuals = (props * k) - np.floor(props * k)
        j = int(np.argmax(residuals))
        alloc[j] += 1
    while alloc.sum() > k:
        j = int(np.argmax(alloc))
        alloc[j] -= 1

    chosen_idx = []
    for c in range(n_components):
        pool = np.where(comp0 == c)[0]
        if alloc[c] > 0 and len(pool) > 0:
            pick = rng.choice(pool, size=min(alloc[c], len(pool)), replace=False)
            chosen_idx.append(pick)

    if len(chosen_idx) > 0:
        idx_strat = np.concatenate(chosen_idx)
    else:
        # fallback: se der algum problema, vira random
        idx_strat = rng.choice(len(X0), size=k, replace=False)

    if len(idx_strat) < k:
        extra = np.setdiff1d(np.arange(len(X0)), idx_strat, assume_unique=False)
        need = k - len(idx_strat)
        add = rng.choice(extra, size=need, replace=False)
        idx_strat = np.concatenate([idx_strat, add])

    S_strat = X0[idx_strat]
    w_strat, _ = soft_assign_weights(X_full, S_strat, gamma=gamma)
    return S_strat, w_strat


# ---------------- Main experiment ---------------- #

def run_experiment():
    X, comp_idx = generate_synthetic_mixture(n=50000, d=5, random_state=7)
    X0, comp0 = build_working_sample(X, comp_idx, n0=20000, random_state=0)

    p0 = density_knn(X0, m_neighbors=32)

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
        selected_idx = select_reps_greedy(X0, p0, k_rep, alpha=alpha, random_state=13)
        S0 = X0[selected_idx]
        w0, A0 = soft_assign_weights(X0, S0, gamma=1.0)
        _, S_ref, _, A_ref = medoid_refinement(X0, selected_idx, A0, max_iters=1)
        w_full, _ = soft_assign_weights(X, S_ref, gamma=1.0)
        t1 = time.time()

        res_ddc = evaluate_representation(
            f"DDC(k={k_rep},α={alpha})", S_ref, w_full, X
        )
        res_ddc["runtime_sec"] = t1 - t0
        rows.append(res_ddc)

        # ---- Random ----
        t0 = time.time()
        S_rnd, w_rnd = build_random_coreset(X0, X, k_rep, gamma=1.0, random_state=123)
        t1 = time.time()

        res_rnd = evaluate_representation(
            f"Random(k={k_rep})", S_rnd, w_rnd, X
        )
        res_rnd["runtime_sec"] = t1 - t0
        rows.append(res_rnd)

        # ---- Stratified ----
        t0 = time.time()
        S_strat, w_strat = build_stratified_coreset(
            X0, comp0, X, k_rep,
            n_components=4, gamma=1.0, random_state=321
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
