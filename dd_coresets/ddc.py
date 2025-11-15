"""
dd_coresets.ddc
---------------
Core implementation of Density–Diversity Coresets (DDC)
and simple baselines (random, stratified).

API principal:

    S, w, info = fit_ddc_coreset(...)

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import OAS, LedoitWolf


# --------- Dataclass de retorno (opcional, mas útil) --------- #

@dataclass
class CoresetInfo:
    method: str
    k: int
    n: int
    n0: int
    working_indices: np.ndarray
    selected_indices: np.ndarray
    alpha: Optional[float] = None
    m_neighbors: Optional[int] = None
    gamma: Optional[float] = None
    refine_iters: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # arrays não são JSON-friendly, mas isso já ajuda em debug interno
        return d


# --------- Helpers internos (não-exportados) --------- #

def _pairwise_sq_dists(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)
    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    D2 = XX + YY - 2.0 * (X @ Y.T)
    return np.maximum(D2, 0.0)


def _density_knn_euclidean(X: np.ndarray, m_neighbors: int = 32) -> np.ndarray:
    """
    kNN-based local density proxy (Euclidean).

    p_i ∝ 1 / r_k(x_i)^d, normalised to sum to 1.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    m = min(m_neighbors + 1, max(2, n))  # +1 inclui self

    nn = NearestNeighbors(n_neighbors=m, algorithm="ball_tree")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)

    rk = dists[:, -1]
    rk = np.maximum(rk, 1e-12)
    p = 1.0 / (rk ** d)
    p /= p.sum()
    return p


# Backward compatibility alias
_density_knn = _density_knn_euclidean


def _select_reps_greedy(
    X: np.ndarray,
    p: np.ndarray,
    k: int,
    alpha: float = 0.3,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Greedy density–diversity selection in O(k * n * d).
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)

    n, d = X.shape
    if k >= n:
        return np.arange(n, dtype=int)

    selected = np.empty(k, dtype=int)

    # Primeiro representante: maior densidade
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
        scores[selected[:t]] = -np.inf
        j_next = int(np.argmax(scores))
        selected[t] = j_next

    return selected


def _soft_assign_weights(
    X: np.ndarray,
    S: np.ndarray,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft assignments via Gaussian kernel and resulting weights.

    Returns
    -------
    w : (k,)
        Weights (sum to 1).
    A : (n, k)
        Assignment matrix (rows sum to 1).
    """
    X = np.asarray(X, dtype=float)
    S = np.asarray(S, dtype=float)

    D2 = _pairwise_sq_dists(X, S)
    med = float(np.median(D2))
    if med <= 0.0:
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


def _medoid_refinement(
    X: np.ndarray,
    selected_idx: np.ndarray,
    A: np.ndarray,
    max_iters: int = 1,
    gamma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Few medoid refinement iterations.

    Returns
    -------
    selected_idx_ref : (k,)
    S_ref : (k, d)
    w_ref : (k,)
    A_ref : (n, k)
    """
    X = np.asarray(X, dtype=float)
    selected_idx = np.asarray(selected_idx, dtype=int)
    n, d = X.shape
    k = len(selected_idx)

    # Initialize
    S = X[selected_idx]
    w, A = _soft_assign_weights(X, S, gamma=gamma)

    for _ in range(max_iters):
        C = np.argmax(A, axis=1)  # hard cluster
        changed = False

        for j in range(k):
            idx_cluster = np.where(C == j)[0]
            if idx_cluster.size == 0:
                continue

            Xc = X[idx_cluster]
            D2_local = _pairwise_sq_dists(Xc)
            mean_dist = np.sqrt(D2_local).mean(axis=1)

            best_local = int(np.argmin(mean_dist))
            new_idx = idx_cluster[best_local]

            if new_idx != selected_idx[j]:
                changed = True
            selected_idx[j] = new_idx

        S = X[selected_idx]
        w, A = _soft_assign_weights(X, S, gamma=gamma)
        if not changed:
            break

    S = X[selected_idx]
    return selected_idx, S, w, A


# --------- Adaptive density estimation --------- #

def _estimate_density_adaptive(
    X: np.ndarray,
    nbrs_idx: np.ndarray,  # (n0, m+1) euclidean neighbors (first is self)
    *,
    m_neighbors: int,
    iterations: int,
    shrinkage: str,  # "oas" | "lw" | "none"
    reg_eps: float,
) -> np.ndarray:
    """
    Local Mahalanobis density estimation.

    Parameters
    ----------
    X : (n0, d_eff) array
        Working sample data.
    nbrs_idx : (n0, m+1) array
        Euclidean neighbor indices (first column is self).
    m_neighbors : int
        Number of neighbors (excluding self).
    iterations : int
        Number of refinement iterations.
    shrinkage : str
        "oas" | "lw" | "none"
    reg_eps : float
        Regularization epsilon.

    Returns
    -------
    p : (n0,) array
        Density estimates (sum to 1).
    """
    X = np.asarray(X, dtype=float)
    n0, d_eff = X.shape

    # Initialize with Euclidean (use passed neighbor indices)
    # nbrs_idx[i, m_neighbors] is the k-th neighbor (0-indexed, so m_neighbors is k+1-th)
    rk_euclidean = np.zeros(n0)
    for i in range(n0):
        kth_nbr_idx = nbrs_idx[i, m_neighbors]
        rk_euclidean[i] = np.linalg.norm(X[i] - X[kth_nbr_idx])
    rk_euclidean = np.maximum(rk_euclidean, 1e-12)
    p = m_neighbors / (rk_euclidean ** d_eff + 1e-10)

    # Iterative refinement
    for iteration in range(iterations):
        new_p = np.zeros(n0)

        for i in range(n0):
            # Get neighbors (exclude self)
            neighbor_indices = nbrs_idx[i, 1 : m_neighbors + 1]
            neighbors = X[neighbor_indices]

            if len(neighbors) <= d_eff:
                new_p[i] = p[i]
                continue

            # Local mean
            mu_local = neighbors.mean(axis=0)

            # Local covariance with shrinkage
            if shrinkage == "oas":
                oas = OAS()
                oas.fit(neighbors)
                C = oas.covariance_
            elif shrinkage == "lw":
                lw = LedoitWolf()
                lw.fit(neighbors)
                C = lw.covariance_
            else:  # "none"
                C = np.cov(neighbors.T)

            # Regularization
            C += np.eye(d_eff) * reg_eps

            try:
                # Cholesky decomposition
                L = np.linalg.cholesky(C)
                detC = (np.diag(L).prod()) ** 2

                # Mahalanobis distances to neighbors only
                D = neighbors - mu_local
                Y = np.linalg.solve(L, D.T)  # (d_eff, m_neighbors)
                dM = np.sqrt((Y * Y).sum(axis=0))  # (m_neighbors,)

                # k-th Mahalanobis neighbor
                rk_mahal = np.partition(dM, m_neighbors - 1)[m_neighbors - 1]
                rk_mahal = np.maximum(rk_mahal, 1e-12)

                # Adaptive density
                new_p[i] = m_neighbors / (rk_mahal ** d_eff * np.sqrt(detC) + 1e-12)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to Euclidean
                new_p[i] = p[i]

        p = new_p

    # Normalize
    p = np.maximum(p, 1e-18)
    p /= p.sum()
    return p


# --------- Config resolution --------- #

def _resolve_config(
    mode: str,
    preset: str,
    distance_cfg: Optional[Dict[str, Any]],
    pipeline_cfg: Optional[Dict[str, Any]],
    legacy_kwargs: Dict[str, Any],
    d: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Resolve configuration from presets, user overrides, and legacy kwargs.

    Returns
    -------
    distance_cfg_resolved : dict
    pipeline_cfg_resolved : dict
    meta_info : dict
        {"mode": str, "preset": str, "fallbacks": list, "deprecations": list}
    """
    from .pipelines import PRESETS

    deprecations = []
    fallbacks = []

    # Start with preset defaults (or empty if manual)
    if preset == "manual":
        base_distance_cfg = {}
        base_pipeline_cfg = {}
    elif preset not in PRESETS:
        raise ValueError(f"preset must be one of {list(PRESETS.keys())} or 'manual', got {preset}")
    else:
        base_distance_cfg = PRESETS[preset]["distance_cfg"].copy()
        base_pipeline_cfg = PRESETS[preset]["pipeline_cfg"].copy()

    # Override with user configs
    if distance_cfg is not None:
        base_distance_cfg.update(distance_cfg)
    if pipeline_cfg is not None:
        base_pipeline_cfg.update(pipeline_cfg)

    # Map legacy kwargs
    legacy_mappings = {
        "use_adaptive_distance": ("mode", lambda x: "adaptive" if x else "euclidean"),
        "m_neighbors": ("distance_cfg", "m_neighbors"),
        "adaptive_iterations": ("distance_cfg", "iterations"),
        "dim_threshold_adaptive": ("pipeline_cfg", "dim_threshold_adaptive"),
        "pca_n_components": ("pipeline_cfg", "cap_components"),
        "shrinkage": ("distance_cfg", "shrinkage"),
        "reg_eps": ("distance_cfg", "reg_eps"),
    }

    for legacy_key, value in legacy_kwargs.items():
        if legacy_key in legacy_mappings:
            deprecations.append(legacy_key)
            mapping = legacy_mappings[legacy_key]
            if callable(mapping[1]):
                # Transform function
                if legacy_key == "use_adaptive_distance":
                    mode = mapping[1](value)
            else:
                # Direct mapping
                cfg_key = mapping[0]
                param_key = mapping[1]
                if cfg_key == "distance_cfg":
                    base_distance_cfg[param_key] = value
                elif cfg_key == "pipeline_cfg":
                    base_pipeline_cfg[param_key] = value

    # Emit deprecation warnings
    for dep_key in deprecations:
        warnings.warn(
            f"Parameter '{dep_key}' is deprecated. Use 'mode', 'preset', or '*_cfg' dicts instead.",
            DeprecationWarning,
            stacklevel=3,
        )

    # Validate mode
    if mode not in {"euclidean", "adaptive", "auto"}:
        raise ValueError(f"mode must be one of {{'euclidean', 'adaptive', 'auto'}}, got {mode}")

    # Fill defaults for manual preset if missing
    if preset == "manual":
        # Default distance config
        if "m_neighbors" not in base_distance_cfg:
            base_distance_cfg["m_neighbors"] = 32
        if "iterations" not in base_distance_cfg:
            base_distance_cfg["iterations"] = 1
        if "shrinkage" not in base_distance_cfg:
            base_distance_cfg["shrinkage"] = "oas"
        if "reg_eps" not in base_distance_cfg:
            base_distance_cfg["reg_eps"] = 1e-6
        
        # Default pipeline config
        if "dim_threshold_adaptive" not in base_pipeline_cfg:
            base_pipeline_cfg["dim_threshold_adaptive"] = 30
        if "reduce" not in base_pipeline_cfg:
            base_pipeline_cfg["reduce"] = "auto"
        if "retain_variance" not in base_pipeline_cfg:
            base_pipeline_cfg["retain_variance"] = 0.95
        if "cap_components" not in base_pipeline_cfg:
            base_pipeline_cfg["cap_components"] = 50

    meta_info = {
        "mode": mode,
        "preset": preset,
        "fallbacks": fallbacks,
        "deprecations": deprecations,
    }

    return base_distance_cfg, base_pipeline_cfg, meta_info


# --------- API pública --------- #

def fit_ddc_coreset(
    X: np.ndarray,
    k: int,
    *,
    n0: Optional[int] = None,
    alpha: float = 0.3,
    gamma: float = 1.0,
    refine_iters: int = 1,
    reweight_full: bool = True,
    random_state: Optional[int] = None,
    # New surface:
    mode: str = "euclidean",  # "euclidean" | "adaptive" | "auto"
    preset: str = "balanced",  # "fast" | "balanced" | "robust" | "manual"
    distance_cfg: Optional[Dict[str, Any]] = None,  # used if preset="manual"
    pipeline_cfg: Optional[Dict[str, Any]] = None,  # used if preset="manual"
    **legacy_kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit a Density–Diversity Coreset (DDC) on X.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Preprocessed data.
    k : int
        Number of representatives.
    n0 : int or None, default=None
        Working sample size. If None or >= n, all data are used.
    alpha : float, default=0.3
        Density–diversity trade-off (0 ≈ diversity, 1 ≈ density).
    gamma : float, default=1.0
        Kernel scale multiplier for soft assignments.
    refine_iters : int, default=1
        Number of medoid refinement iterations.
    reweight_full : bool, default=True
        If True, recompute weights using full X; else, use working sample.
    random_state : int or None, default=None
        RNG seed.
    mode : str, default="euclidean"
        Distance mode: "euclidean" (default, backward compatible),
        "adaptive" (Mahalanobis), or "auto" (choose based on d).
    preset : str, default="balanced"
        Preset configuration: "fast", "balanced", "robust", or "manual".
    distance_cfg : dict or None, default=None
        Override distance config (used if preset="manual").
    pipeline_cfg : dict or None, default=None
        Override pipeline config (used if preset="manual").
    **legacy_kwargs
        Legacy parameters (deprecated, emit warnings).

    Returns
    -------
    S : (k, d) array
        Representatives (always in original feature space).
    w : (k,) array
        Weights (sum to 1).
    info : dict
        Metadata including resolved configs, pipeline info, fallbacks.
    """
    from .pipelines import choose_pipeline, reduce_dimensionality_if_needed

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Default n0 for backward compatibility
    if n0 is None:
        n0 = 20000

    # Resolve configuration
    distance_cfg_resolved, pipeline_cfg_resolved, meta_info = _resolve_config(
        mode=mode,
        preset=preset,
        distance_cfg=distance_cfg,
        pipeline_cfg=pipeline_cfg,
        legacy_kwargs=legacy_kwargs,
        d=d,
    )

    m_neighbors = distance_cfg_resolved["m_neighbors"]
    dim_threshold_adaptive = pipeline_cfg_resolved["dim_threshold_adaptive"]

    # Choose pipeline strategy
    pipeline_decision = choose_pipeline(
        d=d,
        m_neighbors=m_neighbors,
        mode=mode,
        dim_threshold_adaptive=dim_threshold_adaptive,
    )

    # Reduce dimensionality if needed
    X_eff = X
    pca_info = {"pca_model": None, "n_components": d, "explained_variance_ratio": None}
    if pipeline_decision["do_pca"]:
        X_eff, pca_info = reduce_dimensionality_if_needed(
            X,
            reduce=pipeline_cfg_resolved["reduce"],
            retain_variance=pipeline_cfg_resolved["retain_variance"],
            cap_components=pipeline_cfg_resolved["cap_components"],
        )
    d_eff = X_eff.shape[1]

    # Working sample
    if n0 >= n:
        idx_work = np.arange(n, dtype=int)
    else:
        idx_work = rng.choice(n, size=n0, replace=False)
    X0 = X_eff[idx_work]
    n0_eff = X0.shape[0]

    # Build Euclidean kNN (always, for neighbor graph)
    # Adjust m_neighbors if working sample is too small
    m_eff = min(m_neighbors + 1, max(2, n0_eff))
    nn = NearestNeighbors(n_neighbors=m_eff, metric="euclidean")
    nn.fit(X0)
    dists_euclidean, nbrs_idx = nn.kneighbors(X0, return_distance=True)
    # Adjust m_neighbors for density estimation if needed
    m_neighbors_eff = min(m_neighbors, n0_eff - 1)

    # Compute density
    use_adaptive = pipeline_decision["adaptive"] and (m_neighbors_eff > d_eff)
    if use_adaptive:
        p0 = _estimate_density_adaptive(
            X0,
            nbrs_idx,
            m_neighbors=m_neighbors_eff,
            iterations=distance_cfg_resolved["iterations"],
            shrinkage=distance_cfg_resolved["shrinkage"],
            reg_eps=distance_cfg_resolved["reg_eps"],
        )
    else:
        if pipeline_decision["adaptive"] and not (m_neighbors_eff > d_eff):
            meta_info["fallbacks"].append(
                f"Adaptive requested but m_neighbors ({m_neighbors_eff}) <= d_eff ({d_eff}), using Euclidean"
            )
        p0 = _density_knn_euclidean(X0, m_neighbors=m_neighbors_eff)

    # Greedy selection on working sample (in effective space)
    selected_idx0 = _select_reps_greedy(
        X0, p0, k, alpha=alpha, random_state=random_state
    )
    S0_eff = X0[selected_idx0]

    # Map back to original space if PCA was used
    if pca_info["pca_model"] is not None:
        S0 = pca_info["pca_model"].inverse_transform(S0_eff)
    else:
        S0 = S0_eff

    # Soft assign + medoid refinement on working sample (use original space for distances)
    X0_orig = X[idx_work]
    # selected_idx0 are indices relative to X0 (working sample in effective space)
    # For refinement, we need to work in original space, so we use S0 (already mapped back)
    # and refine by finding closest points in X0_orig to S0
    w0, A0 = _soft_assign_weights(X0_orig, S0, gamma=gamma)
    # Refinement: find medoids in original space based on soft assignments
    selected_idx_ref0, S_ref0, w_ref0, A_ref0 = _medoid_refinement(
        X0_orig, selected_idx0, A0, max_iters=refine_iters, gamma=gamma
    )

    # Reweight on full data if requested
    if reweight_full:
        S = S_ref0
        w_full, A_full = _soft_assign_weights(X, S, gamma=gamma)
        w = w_full
    else:
        S = S_ref0
        w = w_ref0

    # Map selected_idx_ref0 (índices relativos ao working sample) para X
    selected_global = idx_work[selected_idx_ref0]

    # Build info dict (extended)
    info = {
        "method": "ddc",
        "k": k,
        "n": n,
        "n0": n0_eff,
        "working_indices": idx_work,
        "selected_indices": selected_global,
        "alpha": alpha,
        "gamma": gamma,
        "refine_iters": refine_iters,
        "pipeline": {
            "mode": meta_info["mode"],
            "preset": meta_info["preset"],
            "adaptive": use_adaptive,
            "pca_used": pca_info["pca_model"] is not None,
            "d_original": d,
            "d_effective": d_eff,
            "fallbacks": meta_info["fallbacks"],
        },
        "config": {
            "distance_cfg": distance_cfg_resolved,
            "pipeline_cfg": pipeline_cfg_resolved,
        },
        "pca": pca_info,
        "deprecations": meta_info["deprecations"],
        "random_state": random_state,
    }

    return S, w, info


def fit_random_coreset(
    X: np.ndarray,
    k: int,
    n0: Optional[int] = 20000,
    gamma: float = 1.0,
    reweight_full: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, CoresetInfo]:
    """
    Random coreset baseline: uniform sample + soft weights.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if (n0 is None) or (n0 >= n):
        idx_work = np.arange(n, dtype=int)
    else:
        idx_work = rng.choice(n, size=n0, replace=False)
    X0 = X[idx_work]
    n0_eff = X0.shape[0]

    # Sample k from working sample
    idx_local = rng.choice(n0_eff, size=min(k, n0_eff), replace=False)
    S0 = X0[idx_local]

    if reweight_full:
        w, A = _soft_assign_weights(X, S0, gamma=gamma)
    else:
        w, A = _soft_assign_weights(X0, S0, gamma=gamma)

    selected_global = idx_work[idx_local]

    info = CoresetInfo(
        method="random",
        k=len(S0),
        n=n,
        n0=n0_eff,
        working_indices=idx_work,
        selected_indices=selected_global,
    )
    return S0, w, info


def fit_stratified_coreset(
    X: np.ndarray,
    strata: np.ndarray,
    k: int,
    n0: Optional[int] = 20000,
    gamma: float = 1.0,
    reweight_full: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, CoresetInfo]:
    """
    Stratified coreset baseline: allocate k_g reps per stratum ∝ frequency,
    sample uniformly within stratum, then apply soft weighting.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    strata = np.asarray(strata, dtype=int)
    n, d = X.shape
    assert strata.shape[0] == n, "strata must have length n"

    # Working sample (preserva rótulos)
    if (n0 is None) or (n0 >= n):
        idx_work = np.arange(n, dtype=int)
    else:
        idx_work = rng.choice(n, size=n0, replace=False)
    X0 = X[idx_work]
    strata0 = strata[idx_work]
    n0_eff = X0.shape[0]

    unique = np.unique(strata0)
    G = len(unique)

    counts = np.array([np.sum(strata0 == g) for g in unique], dtype=float)
    props = counts / counts.sum()

    alloc = np.floor(props * k).astype(int)
    # Ajusta arredondamento
    while alloc.sum() < k:
        residuals = (props * k) - np.floor(props * k)
        j = int(np.argmax(residuals))
        alloc[j] += 1
    while alloc.sum() > k:
        j = int(np.argmax(alloc))
        alloc[j] -= 1

    chosen_local = []
    for g, kg in zip(unique, alloc):
        if kg <= 0:
            continue
        pool = np.where(strata0 == g)[0]
        if len(pool) == 0:
            continue
        k_eff = min(kg, len(pool))
        pick = rng.choice(pool, size=k_eff, replace=False)
        chosen_local.append(pick)

    if len(chosen_local) == 0:
        # fallback: vira random
        return fit_random_coreset(
            X, k=k, n0=n0, gamma=gamma, reweight_full=reweight_full,
            random_state=random_state
        )

    idx_local = np.concatenate(chosen_local)
    if len(idx_local) > k:
        idx_local = rng.choice(idx_local, size=k, replace=False)

    S0 = X0[idx_local]

    if reweight_full:
        w, A = _soft_assign_weights(X, S0, gamma=gamma)
    else:
        w, A = _soft_assign_weights(X0, S0, gamma=gamma)

    selected_global = idx_work[idx_local]

    info = CoresetInfo(
        method="stratified",
        k=len(S0),
        n=n,
        n0=n0_eff,
        working_indices=idx_work,
        selected_indices=selected_global,
    )
    return S0, w, info


def fit_kmedoids_coreset(
    X: np.ndarray,
    k: int,
    n0: Optional[int] = 20000,
    gamma: float = 1.0,
    reweight_full: bool = True,
    max_iters: int = 10,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, CoresetInfo]:
    """
    K-medoids baseline: selects k medoids (real data points) that minimize
    the sum of distances to nearest medoid. Uses PAM-like algorithm with
    working sample for efficiency.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    # Working sample
    if (n0 is None) or (n0 >= n):
        idx_work = np.arange(n, dtype=int)
    else:
        idx_work = rng.choice(n, size=n0, replace=False)
    X0 = X[idx_work]
    n0_eff = X0.shape[0]

    # Initialize medoids using k-means++ style initialization
    medoid_idx = np.zeros(k, dtype=int)
    medoid_idx[0] = rng.integers(0, n0_eff)
    
    # Compute distances from all points to first medoid (more efficient than pairwise)
    min_dists = np.sum((X0 - X0[medoid_idx[0]])**2, axis=1)
    
    for i in range(1, k):
        # Probability proportional to squared distance
        probs = min_dists / (min_dists.sum() + 1e-10)
        
        # Filter out already-selected medoids to ensure uniqueness
        available_idx = np.setdiff1d(np.arange(n0_eff), medoid_idx[:i])
        if len(available_idx) == 0:
            # Fallback: if all points are selected, break early
            medoid_idx = medoid_idx[:i]
            break
        
        # Map probabilities to available indices
        probs_available = probs[available_idx]
        probs_available = probs_available / (probs_available.sum() + 1e-10)
        
        selected_available = rng.choice(len(available_idx), p=probs_available)
        medoid_idx[i] = available_idx[selected_available]
        
        # Update minimum distances
        new_dists = _pairwise_sq_dists(X0, X0[medoid_idx[i:i+1]])[:, 0]
        min_dists = np.minimum(min_dists, new_dists)

    # PAM-like swap optimization
    for _ in range(max_iters):
        # Assign each point to nearest medoid (compute distances on-demand)
        medoid_dists = _pairwise_sq_dists(X0, X0[medoid_idx])
        assignments = np.argmin(medoid_dists, axis=1)
        
        changed = False
        for j in range(k):
            cluster_mask = (assignments == j)
            if cluster_mask.sum() == 0:
                continue
            
            cluster_points = X0[cluster_mask]
            
            # Current cost: sum of distances to medoid j
            current_medoid_dists = _pairwise_sq_dists(cluster_points, X0[medoid_idx[j:j+1]])[:, 0]
            current_cost = np.sqrt(current_medoid_dists).sum()
            
            # Try swapping with each non-medoid in cluster
            candidates = np.where(cluster_mask)[0]
            candidates = candidates[~np.isin(candidates, medoid_idx)]
            
            if len(candidates) == 0:
                continue
            
            best_candidate = None
            best_cost = current_cost
            
            for candidate in candidates:
                # Cost if we swap medoid j with candidate
                candidate_dists = _pairwise_sq_dists(cluster_points, X0[candidate:candidate+1])[:, 0]
                new_cost = np.sqrt(candidate_dists).sum()
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_candidate = candidate
            
            if best_candidate is not None:
                medoid_idx[j] = best_candidate
                changed = True
        
        if not changed:
            break

    S0 = X0[medoid_idx]

    if reweight_full:
        w, A = _soft_assign_weights(X, S0, gamma=gamma)
    else:
        w, A = _soft_assign_weights(X0, S0, gamma=gamma)

    selected_global = idx_work[medoid_idx]

    info = CoresetInfo(
        method="kmedoids",
        k=len(S0),
        n=n,
        n0=n0_eff,
        working_indices=idx_work,
        selected_indices=selected_global,
    )
    return S0, w, info
