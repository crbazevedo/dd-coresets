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

import numpy as np
from sklearn.neighbors import NearestNeighbors


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


def _density_knn(X: np.ndarray, m_neighbors: int = 32) -> np.ndarray:
    """
    kNN-based local density proxy.

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
        w, A = _soft_assign_weights(X, S)
        if not changed:
            break

    S = X[selected_idx]
    return selected_idx, S, w, A


# --------- API pública --------- #

def fit_ddc_coreset(
    X: np.ndarray,
    k: int,
    n0: Optional[int] = 20000,
    m_neighbors: int = 32,
    alpha: float = 0.3,
    gamma: float = 1.0,
    refine_iters: int = 1,
    reweight_full: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, CoresetInfo]:
    """
    Fit a Density–Diversity Coreset (DDC) on X.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Preprocessed data.
    k : int
        Number of representatives.
    n0 : int or None
        Working sample size. If None or >= n, all data are used.
    m_neighbors : int
        Number of neighbors for kNN density proxy.
    alpha : float
        Density–diversity trade-off (0 ≈ diversity, 1 ≈ density).
    gamma : float
        Kernel scale multiplier for soft assignments.
    refine_iters : int
        Number of medoid refinement iterations.
    reweight_full : bool
        If True, recompute weights using full X; else, use working sample.
    random_state : int or None
        RNG seed.

    Returns
    -------
    S : (k, d)
        Representatives.
    w : (k,)
        Weights (sum to 1).
    info : CoresetInfo
        Metadata (indices, parameters, etc.).
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

    # Density proxy
    p0 = _density_knn(X0, m_neighbors=m_neighbors)

    # Greedy selection on working sample
    selected_idx0 = _select_reps_greedy(
        X0, p0, k, alpha=alpha, random_state=random_state
    )
    S0 = X0[selected_idx0]

    # Soft assign + medoid refinement on working sample
    w0, A0 = _soft_assign_weights(X0, S0, gamma=gamma)
    selected_idx_ref0, S_ref0, w_ref0, A_ref0 = _medoid_refinement(
        X0, selected_idx0, A0, max_iters=refine_iters
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

    info = CoresetInfo(
        method="ddc",
        k=k,
        n=n,
        n0=n0_eff,
        working_indices=idx_work,
        selected_indices=selected_global,
        alpha=alpha,
        m_neighbors=m_neighbors,
        gamma=gamma,
        refine_iters=refine_iters,
    )

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
