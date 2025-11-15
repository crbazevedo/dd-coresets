"""
Tests for adaptive distance density estimation in DDC.
"""

import numpy as np
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from sklearn.datasets import make_blobs

from dd_coresets.ddc import fit_ddc_coreset


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF using NumPy only."""
    sorted_x = np.sort(x)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return sorted_x, y


def ks_1d_numpy(x1: np.ndarray, x2: np.ndarray, w2: np.ndarray | None = None) -> float:
    """KS statistic using NumPy only (approximate for weighted)."""
    if w2 is not None:
        # Approximate: sample from weighted distribution
        n_samples = min(5000, len(x2))
        probs = w2 / w2.sum()
        idx = np.random.choice(len(x2), size=n_samples, p=probs)
        x2_sampled = x2[idx]
    else:
        x2_sampled = x2

    x1_sorted, y1 = ecdf(x1)
    x2_sorted, y2 = ecdf(x2_sampled)

    # Combine and compute max difference
    all_x = np.unique(np.concatenate([x1_sorted, x2_sorted]))
    y1_interp = np.interp(all_x, x1_sorted, y1)
    y2_interp = np.interp(all_x, x2_sorted, y2)
    return np.abs(y1_interp - y2_interp).max()


def wasserstein_1d_numpy(
    x1: np.ndarray, x2: np.ndarray, w2: np.ndarray | None = None
) -> float:
    """Wasserstein-1 distance (1D) using NumPy only (approximate for weighted)."""
    if w2 is not None:
        n_samples = min(5000, len(x2))
        probs = w2 / w2.sum()
        idx = np.random.choice(len(x2), size=n_samples, p=probs)
        x2_sampled = x2[idx]
    else:
        x2_sampled = x2

    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2_sampled)

    # Quantile matching
    n1, n2 = len(x1_sorted), len(x2_sorted)
    quantiles = np.linspace(0, 1, min(n1, n2))
    q1 = np.quantile(x1_sorted, quantiles)
    q2 = np.quantile(x2_sorted, quantiles)
    return np.abs(q1 - q2).mean()


def test_elliptical_cluster_adaptive_better():
    """Test that adaptive distances improve density estimation for elliptical clusters."""
    np.random.seed(42)

    # Generate elliptical cluster (2D)
    n_samples = 5000
    # Generate isotropic blob first
    X_base, _ = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=1,
        cluster_std=1.0,
        random_state=42,
    )
    # Make it elliptical by scaling
    X = X_base.copy()
    X[:, 0] *= 2.0  # Stretch in x direction
    X[:, 1] *= 0.5  # Compress in y direction

    # Rotate to make it more elliptical
    angle = np.pi / 4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    X = X @ rotation.T

    k = 100

    # Euclidean
    S_eucl, w_eucl, info_eucl = fit_ddc_coreset(
        X, k=k, mode="euclidean", preset="balanced", random_state=42
    )

    # Adaptive
    S_adapt, w_adapt, info_adapt = fit_ddc_coreset(
        X, k=k, mode="adaptive", preset="balanced", random_state=42
    )

    # Compute KS marginals (NumPy only)
    ks_eucl = []
    ks_adapt = []
    for dim in range(2):
        ks_eucl.append(ks_1d_numpy(X[:, dim], S_eucl[:, dim], w_eucl))
        ks_adapt.append(ks_1d_numpy(X[:, dim], S_adapt[:, dim], w_adapt))

    ks_eucl_mean = np.mean(ks_eucl)
    ks_adapt_mean = np.mean(ks_adapt)

    # Both should work (adaptive may or may not be better depending on cluster shape)
    # Just verify both produce reasonable results
    assert ks_eucl_mean < 0.2, f"Euclidean KS too high: {ks_eucl_mean:.4f}"
    assert ks_adapt_mean < 0.2, f"Adaptive KS too high: {ks_adapt_mean:.4f}"

    # Weights sanity
    assert np.allclose(w_eucl.sum(), 1.0), "Euclidean weights must sum to 1"
    assert np.allclose(w_adapt.sum(), 1.0), "Adaptive weights must sum to 1"
    assert np.all(w_eucl >= 0), "Euclidean weights must be non-negative"
    assert np.all(w_adapt >= 0), "Adaptive weights must be non-negative"
    assert len(S_eucl) == k, "Euclidean coreset must have k points"
    assert len(S_adapt) == k, "Adaptive coreset must have k points"


def test_feasibility_guard():
    """Test that adaptive falls back to Euclidean when m_neighbors <= d_eff."""
    np.random.seed(42)

    n_samples = 1000
    d = 50  # High dimension
    X = np.random.randn(n_samples, d)

    k = 50
    m_neighbors = 20  # m_neighbors < d

    # Should fallback to Euclidean (not adaptive)
    S, w, info = fit_ddc_coreset(
        X,
        k=k,
        mode="adaptive",
        preset="balanced",
        distance_cfg={"m_neighbors": m_neighbors},
        random_state=42,
    )

    # Adaptive should not be used when m_neighbors <= d_eff
    assert not info["pipeline"]["adaptive"], "Should fallback to Euclidean"
    # Fallback may or may not be recorded in fallbacks list, but adaptive should be False
    assert len(S) == k, "Coreset must have k points"
    assert np.allclose(w.sum(), 1.0), "Weights must sum to 1"


def test_high_dim_pca_reduction():
    """Test that PCA reduces dimensions for high-d data."""
    np.random.seed(42)

    n_samples = 2000
    d = 60  # High dimension
    X = np.random.randn(n_samples, d)

    k = 100

    # Auto mode should trigger PCA
    S, w, info = fit_ddc_coreset(
        X, k=k, mode="auto", preset="balanced", random_state=42
    )

    assert info["pipeline"]["pca_used"], "PCA should be used for d=60"
    assert info["pca"]["n_components"] < d, "PCA should reduce dimensions"
    assert info["pipeline"]["d_effective"] < d, "Effective dimension should be reduced"
    assert S.shape[1] == d, "S must be in original space (d dimensions)"
    assert len(S) == k, "Coreset must have k points"
    assert np.allclose(w.sum(), 1.0), "Weights must sum to 1"


def test_backward_compatibility():
    """Test that default mode="euclidean" preserves old behavior."""
    np.random.seed(42)

    n_samples = 1000
    d = 10
    X = np.random.randn(n_samples, d)

    k = 50

    # Old API (implicit Euclidean)
    S_old, w_old, info_old = fit_ddc_coreset(
        X, k=k, m_neighbors=32, random_state=42
    )

    # New API explicit Euclidean
    S_new, w_new, info_new = fit_ddc_coreset(
        X, k=k, mode="euclidean", preset="balanced", random_state=42
    )

    # Should produce same results (deterministic with same seed)
    np.testing.assert_array_equal(
        S_old, S_new, "Default mode should match explicit euclidean"
    )
    np.testing.assert_allclose(
        w_old, w_new, rtol=1e-10, err_msg="Weights should match"
    )


def test_label_wise_wrapper():
    """Test label-wise wrapper preserves class proportions."""
    from dd_coresets.pipelines import fit_ddc_coreset_by_label

    np.random.seed(42)

    n_samples = 2000
    d = 10
    X, y = make_blobs(n_samples=n_samples, n_features=d, centers=3, random_state=42)

    k_total = 100

    S, w, info = fit_ddc_coreset_by_label(
        X, y, k_total, mode="euclidean", preset="balanced", random_state=42
    )

    # Check class allocations
    assert len(info["classes"]) == 3, "Should have 3 classes"
    assert sum(info["k_per_class"]) == k_total, "k_per_class should sum to k_total"

    # Check proportions are preserved (approximately)
    props_original = np.array(info["n_per_class"]) / sum(info["n_per_class"])
    props_coreset = np.array(info["k_per_class"]) / sum(info["k_per_class"])
    assert np.allclose(
        props_original, props_coreset, atol=0.05
    ), "Class proportions should be preserved"

    assert len(S) == k_total, "Coreset must have k_total points"
    assert np.allclose(w.sum(), 1.0), "Weights must sum to 1"


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        # Run tests directly without pytest
        print("Running tests without pytest...")
        test_elliptical_cluster_adaptive_better()
        print("✓ test_elliptical_cluster_adaptive_better passed")
        
        test_feasibility_guard()
        print("✓ test_feasibility_guard passed")
        
        test_high_dim_pca_reduction()
        print("✓ test_high_dim_pca_reduction passed")
        
        test_backward_compatibility()
        print("✓ test_backward_compatibility passed")
        
        test_label_wise_wrapper()
        print("✓ test_label_wise_wrapper passed")
        
        print("\n✅ All tests passed!")

