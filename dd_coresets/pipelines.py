"""
dd_coresets.pipelines
---------------------
Presets, dimensionality policies, and label-wise wrapper for DDC.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import OAS, LedoitWolf


# --------- Presets (single source of truth) --------- #

PRESETS = {
    "fast": {
        "distance_cfg": {
            "m_neighbors": 24,
            "iterations": 1,
            "shrinkage": "oas",
            "reg_eps": 1e-6,
        },
        "pipeline_cfg": {
            "dim_threshold_adaptive": 30,
            "reduce": "auto",
            "retain_variance": 0.95,
            "cap_components": 30,
        },
    },
    "balanced": {
        "distance_cfg": {
            "m_neighbors": 32,
            "iterations": 1,
            "shrinkage": "oas",
            "reg_eps": 1e-6,
        },
        "pipeline_cfg": {
            "dim_threshold_adaptive": 30,
            "reduce": "auto",
            "retain_variance": 0.95,
            "cap_components": 50,
        },
    },
    "robust": {
        "distance_cfg": {
            "m_neighbors": 64,
            "iterations": 2,
            "shrinkage": "oas",
            "reg_eps": 1e-5,
        },
        "pipeline_cfg": {
            "dim_threshold_adaptive": 30,
            "reduce": "auto",
            "retain_variance": 0.98,
            "cap_components": 60,
        },
    },
}


# --------- Policy functions --------- #

def choose_pipeline(
    d: int, m_neighbors: int, mode: str, dim_threshold_adaptive: int
) -> Dict[str, Any]:
    """
    Choose pipeline strategy based on dimensionality and mode.

    Parameters
    ----------
    d : int
        Original dimensionality.
    m_neighbors : int
        Number of neighbors for density estimation.
    mode : str
        "euclidean" | "adaptive" | "auto"
    dim_threshold_adaptive : int
        Threshold above which to consider PCA.

    Returns
    -------
    dict
        {"adaptive": bool, "do_pca": bool, "fallback_reason": str | None}
    """
    if mode == "euclidean":
        return {"adaptive": False, "do_pca": False, "fallback_reason": None}

    if mode == "adaptive":
        if m_neighbors > d:
            return {"adaptive": True, "do_pca": False, "fallback_reason": None}
        else:
            return {
                "adaptive": False,
                "do_pca": False,
                "fallback_reason": f"m_neighbors ({m_neighbors}) <= d ({d})",
            }

    # mode == "auto"
    if d < 20:
        return {"adaptive": False, "do_pca": False, "fallback_reason": None}

    if 20 <= d < dim_threshold_adaptive:
        if m_neighbors > d:
            return {"adaptive": True, "do_pca": False, "fallback_reason": None}
        else:
            return {
                "adaptive": False,
                "do_pca": False,
                "fallback_reason": f"m_neighbors ({m_neighbors}) <= d ({d})",
            }

    # d >= dim_threshold_adaptive
    return {"adaptive": True, "do_pca": True, "fallback_reason": None}


def reduce_dimensionality_if_needed(
    X: np.ndarray,
    reduce: str,
    retain_variance: float,
    cap_components: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reduce dimensionality using PCA if needed.

    Parameters
    ----------
    X : (n, d) array
        Input data.
    reduce : str
        "auto" | "pca" | "none"
    retain_variance : float
        Target variance to retain (0-1).
    cap_components : int
        Maximum number of components.

    Returns
    -------
    X_reduced : (n, d_eff) array
        Reduced data (or original if no reduction).
    info : dict
        {"pca_model": PCA | None, "n_components": int, "explained_variance_ratio": array | None}
    """
    if reduce not in {"auto", "pca"}:
        return X, {"pca_model": None, "n_components": X.shape[1], "explained_variance_ratio": None}

    n, d = X.shape
    pca = PCA(n_components=min(cap_components, d), random_state=42)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = np.searchsorted(cumvar, retain_variance) + 1
    n_comp = min(n_comp, cap_components, d)

    pca_final = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca_final.fit_transform(X)

    return X_reduced, {
        "pca_model": pca_final,
        "n_components": n_comp,
        "explained_variance_ratio": pca_final.explained_variance_ratio_,
    }


# --------- Label-wise wrapper --------- #

def fit_ddc_coreset_by_label(
    X: np.ndarray,
    y: np.ndarray,
    k_total: int,
    **ddc_kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit DDC coreset preserving class proportions.

    Parameters
    ----------
    X : (n, d) array
        Input data.
    y : (n,) array
        Class labels.
    k_total : int
        Total number of representatives.
    **ddc_kwargs
        Arguments passed to fit_ddc_coreset.

    Returns
    -------
    S : (k_total, d) array
        Representatives.
    w : (k_total,) array
        Weights (sum to 1).
    info : dict
        Metadata including class allocations.
    """
    from .ddc import fit_ddc_coreset

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, d = X.shape
    assert y.shape[0] == n, "y must have length n"

    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # Compute class proportions
    counts = np.array([np.sum(y == c) for c in unique_classes], dtype=float)
    props = counts / counts.sum()

    # Allocate k per class
    alloc = np.floor(props * k_total).astype(int)
    while alloc.sum() < k_total:
        residuals = (props * k_total) - np.floor(props * k_total)
        j = int(np.argmax(residuals))
        alloc[j] += 1
    while alloc.sum() > k_total:
        j = int(np.argmax(alloc))
        alloc[j] -= 1

    # Fit DDC per class
    S_list = []
    w_list = []
    info_list = []

    for c, k_c in zip(unique_classes, alloc):
        if k_c <= 0:
            continue

        mask = y == c
        X_c = X[mask]
        if len(X_c) == 0:
            continue

        k_eff = min(k_c, len(X_c))
        S_c, w_c, info_c = fit_ddc_coreset(X_c, k=k_eff, **ddc_kwargs)

        S_list.append(S_c)
        w_list.append(w_c)
        info_list.append(info_c)

    if len(S_list) == 0:
        raise ValueError("No valid classes found")

    S = np.vstack(S_list)
    w = np.concatenate(w_list)
    w = w / w.sum()  # Renormalize

    # Build info
    info = {
        "method": "ddc_by_label",
        "classes": unique_classes.tolist(),
        "k_per_class": alloc.tolist(),
        "n_per_class": counts.tolist(),
        "info_per_class": info_list,
    }

    return S, w, info

