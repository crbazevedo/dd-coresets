"""
Tests for edge cases and error handling.
"""

import numpy as np
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from dd_coresets.ddc import fit_ddc_coreset
from dd_coresets.pipelines import fit_ddc_coreset_by_label


def test_fit_ddc_coreset_k_larger_than_n():
    """Test when k >= n."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    
    S, w, info = fit_ddc_coreset(X, k=100, mode="euclidean", random_state=42)
    
    # Should return at most n points
    assert len(S) <= 50
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_k_equals_1():
    """Test with k=1."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    S, w, info = fit_ddc_coreset(X, k=1, mode="euclidean", random_state=42)
    
    assert len(S) == 1
    assert np.allclose(w.sum(), 1.0)
    assert w[0] == 1.0


def test_fit_ddc_coreset_very_small_dataset():
    """Test with very small dataset."""
    np.random.seed(42)
    X = np.random.randn(10, 3)
    
    # Use smaller m_neighbors for small dataset
    S, w, info = fit_ddc_coreset(
        X, k=5, mode="euclidean", preset="manual",
        distance_cfg={"m_neighbors": 5, "iterations": 1, "shrinkage": "oas", "reg_eps": 1e-6},
        pipeline_cfg={"dim_threshold_adaptive": 30, "reduce": "none", "retain_variance": 0.95, "cap_components": 30},
        random_state=42
    )
    
    assert len(S) == 5
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_single_feature():
    """Test with d=1."""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    
    S, w, info = fit_ddc_coreset(X, k=20, mode="euclidean", random_state=42)
    
    assert len(S) == 20
    assert S.shape[1] == 1
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_invalid_mode():
    """Test invalid mode raises error."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    try:
        fit_ddc_coreset(X, k=20, mode="invalid", random_state=42)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_fit_ddc_coreset_invalid_preset():
    """Test invalid preset raises error."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    try:
        fit_ddc_coreset(X, k=20, preset="invalid", random_state=42)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_fit_ddc_coreset_alpha_zero():
    """Test with alpha=0 (pure diversity)."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    S, w, info = fit_ddc_coreset(X, k=20, alpha=0.0, mode="euclidean", random_state=42)
    
    assert len(S) == 20
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_alpha_one():
    """Test with alpha=1 (pure density)."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    S, w, info = fit_ddc_coreset(X, k=20, alpha=1.0, mode="euclidean", random_state=42)
    
    assert len(S) == 20
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_gamma_variations():
    """Test different gamma values."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    for gamma in [0.1, 1.0, 10.0]:
        S, w, info = fit_ddc_coreset(
            X, k=20, gamma=gamma, mode="euclidean", random_state=42
        )
        assert len(S) == 20
        assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_refine_iters_zero():
    """Test with refine_iters=0."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    S, w, info = fit_ddc_coreset(
        X, k=20, refine_iters=0, mode="euclidean", random_state=42
    )
    
    assert len(S) == 20
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_constant_data():
    """Test with constant data (all points same)."""
    np.random.seed(42)
    X = np.ones((100, 5)) * 0.5
    
    S, w, info = fit_ddc_coreset(X, k=20, mode="euclidean", random_state=42)
    
    assert len(S) == 20
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_duplicate_points():
    """Test with duplicate points."""
    np.random.seed(42)
    X_base = np.random.randn(50, 5)
    X = np.vstack([X_base, X_base])  # Duplicate all points
    
    S, w, info = fit_ddc_coreset(X, k=20, mode="euclidean", random_state=42)
    
    assert len(S) == 20
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_by_label_imbalanced():
    """Test label-wise with highly imbalanced classes."""
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    
    # Very imbalanced: 90%, 9%, 1%
    X1, _ = make_blobs(n_samples=900, n_features=5, centers=1, random_state=42)
    X2, _ = make_blobs(n_samples=90, n_features=5, centers=1, random_state=43)
    X3, _ = make_blobs(n_samples=10, n_features=5, centers=1, random_state=44)
    
    X = np.vstack([X1, X2, X3])
    y = np.concatenate([np.zeros(900), np.ones(90), np.full(10, 2)])
    
    S, w, info = fit_ddc_coreset_by_label(
        X, y, k_total=100, mode="euclidean", random_state=42
    )
    
    assert len(S) == 100
    assert np.allclose(w.sum(), 1.0)
    # Smallest class should still get at least 1 point
    assert min(info["k_per_class"]) >= 1


def test_fit_ddc_coreset_by_label_single_sample_class():
    """Test with class that has only 1 sample."""
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    
    X1, _ = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)
    X2 = np.random.randn(1, 5)  # Single point
    
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(100), np.ones(1)])
    
    S, w, info = fit_ddc_coreset_by_label(
        X, y, k_total=20, mode="euclidean", random_state=42
    )
    
    assert len(S) == 20
    assert np.allclose(w.sum(), 1.0)


def test_fit_ddc_coreset_adaptive_m_neighbors_too_small():
    """Test adaptive with m_neighbors too small."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    # m_neighbors < d should fallback
    S, w, info = fit_ddc_coreset(
        X,
        k=30,
        mode="adaptive",
        preset="manual",
        distance_cfg={"m_neighbors": 5, "iterations": 1, "shrinkage": "oas", "reg_eps": 1e-6},  # < 10
        pipeline_cfg={"dim_threshold_adaptive": 30, "reduce": "none", "retain_variance": 0.95, "cap_components": 30},
        random_state=42,
    )
    
    assert len(S) == 30
    assert not info["pipeline"]["adaptive"]  # Should fallback


def test_fit_ddc_coreset_pca_retain_all_variance():
    """Test PCA with retain_variance=1.0."""
    np.random.seed(42)
    X = np.random.randn(200, 50)
    
    S, w, info = fit_ddc_coreset(
        X,
        k=30,
        mode="auto",
        preset="manual",
        pipeline_cfg={"retain_variance": 1.0, "cap_components": 50},
        random_state=42,
    )
    
    assert len(S) == 30
    # Should use all components or cap
    assert info["pca"]["n_components"] <= 50


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("Running tests without pytest...")
        test_fit_ddc_coreset_k_larger_than_n()
        print("✓ test_fit_ddc_coreset_k_larger_than_n passed")
        
        test_fit_ddc_coreset_k_equals_1()
        print("✓ test_fit_ddc_coreset_k_equals_1 passed")
        
        test_fit_ddc_coreset_very_small_dataset()
        print("✓ test_fit_ddc_coreset_very_small_dataset passed")
        
        test_fit_ddc_coreset_single_feature()
        print("✓ test_fit_ddc_coreset_single_feature passed")
        
        test_fit_ddc_coreset_invalid_mode()
        print("✓ test_fit_ddc_coreset_invalid_mode passed")
        
        test_fit_ddc_coreset_invalid_preset()
        print("✓ test_fit_ddc_coreset_invalid_preset passed")
        
        test_fit_ddc_coreset_alpha_zero()
        print("✓ test_fit_ddc_coreset_alpha_zero passed")
        
        test_fit_ddc_coreset_alpha_one()
        print("✓ test_fit_ddc_coreset_alpha_one passed")
        
        test_fit_ddc_coreset_gamma_variations()
        print("✓ test_fit_ddc_coreset_gamma_variations passed")
        
        test_fit_ddc_coreset_refine_iters_zero()
        print("✓ test_fit_ddc_coreset_refine_iters_zero passed")
        
        test_fit_ddc_coreset_constant_data()
        print("✓ test_fit_ddc_coreset_constant_data passed")
        
        test_fit_ddc_coreset_duplicate_points()
        print("✓ test_fit_ddc_coreset_duplicate_points passed")
        
        test_fit_ddc_coreset_by_label_imbalanced()
        print("✓ test_fit_ddc_coreset_by_label_imbalanced passed")
        
        test_fit_ddc_coreset_by_label_single_sample_class()
        print("✓ test_fit_ddc_coreset_by_label_single_sample_class passed")
        
        test_fit_ddc_coreset_adaptive_m_neighbors_too_small()
        print("✓ test_fit_ddc_coreset_adaptive_m_neighbors_too_small passed")
        
        test_fit_ddc_coreset_pca_retain_all_variance()
        print("✓ test_fit_ddc_coreset_pca_retain_all_variance passed")
        
        print("\n✅ All tests passed!")

