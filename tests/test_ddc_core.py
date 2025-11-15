"""
Tests for core DDC functions and edge cases.
"""

import numpy as np
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from dd_coresets.ddc import (
    fit_ddc_coreset,
    fit_random_coreset,
    fit_stratified_coreset,
    fit_kmedoids_coreset,
    _density_knn_euclidean,
    _select_reps_greedy,
    _soft_assign_weights,
    _medoid_refinement,
)


def test_density_knn_euclidean_basic():
    """Test basic Euclidean density estimation."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    p = _density_knn_euclidean(X, m_neighbors=10)
    
    assert len(p) == 100
    assert np.allclose(p.sum(), 1.0)
    assert np.all(p >= 0)


def test_density_knn_euclidean_small_n():
    """Test with n < m_neighbors."""
    np.random.seed(42)
    X = np.random.randn(5, 2)
    
    p = _density_knn_euclidean(X, m_neighbors=10)
    
    assert len(p) == 5
    assert np.allclose(p.sum(), 1.0)


def test_select_reps_greedy_basic():
    """Test greedy selection."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    p = np.ones(100) / 100
    
    selected = _select_reps_greedy(X, p, k=10, alpha=0.3, random_state=42)
    
    assert len(selected) == 10
    assert len(np.unique(selected)) == 10  # All unique
    assert np.all(selected >= 0)
    assert np.all(selected < 100)


def test_select_reps_greedy_k_equals_n():
    """Test when k >= n."""
    np.random.seed(42)
    X = np.random.randn(10, 5)
    p = np.ones(10) / 10
    
    selected = _select_reps_greedy(X, p, k=20, alpha=0.3, random_state=42)
    
    assert len(selected) == 10  # Should return all points


def test_soft_assign_weights_basic():
    """Test soft assignment."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    S = X[:10]  # 10 representatives
    
    w, A = _soft_assign_weights(X, S, gamma=1.0)
    
    assert len(w) == 10
    assert np.allclose(w.sum(), 1.0)
    assert np.all(w >= 0)
    assert A.shape == (100, 10)
    assert np.allclose(A.sum(axis=1), 1.0)  # Rows sum to 1


def test_medoid_refinement_basic():
    """Test medoid refinement."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    selected_idx = np.array([0, 10, 20, 30, 40])
    
    # Create dummy assignment matrix
    A = np.random.rand(100, 5)
    A = A / A.sum(axis=1, keepdims=True)
    
    selected_ref, S_ref, w_ref, A_ref = _medoid_refinement(
        X, selected_idx, A, max_iters=2
    )
    
    assert len(selected_ref) == 5
    assert S_ref.shape == (5, 5)
    assert len(w_ref) == 5
    assert np.allclose(w_ref.sum(), 1.0)


def test_fit_ddc_coreset_all_data():
    """Test with n0=None (use all data)."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    
    S, w, info = fit_ddc_coreset(X, k=20, n0=None, mode="euclidean", random_state=42)
    
    assert len(S) == 20
    assert np.allclose(w.sum(), 1.0)
    assert info["n0"] == 100  # Should use all data


def test_fit_ddc_coreset_n0_larger_than_n():
    """Test with n0 > n."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    
    S, w, info = fit_ddc_coreset(X, k=10, n0=1000, mode="euclidean", random_state=42)
    
    assert len(S) == 10
    assert info["n0"] == 50  # Should use all data


def test_fit_ddc_coreset_different_presets():
    """Test different presets."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    for preset in ["fast", "balanced", "robust"]:
        S, w, info = fit_ddc_coreset(
            X, k=30, mode="euclidean", preset=preset, random_state=42
        )
        assert len(S) == 30
        assert np.allclose(w.sum(), 1.0)
        assert info["pipeline"]["preset"] == preset


def test_fit_ddc_coreset_manual_config():
    """Test manual configuration."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    # Manual preset requires explicit configs
    S, w, info = fit_ddc_coreset(
        X,
        k=30,
        mode="euclidean",
        preset="manual",
        distance_cfg={"m_neighbors": 20, "iterations": 1, "shrinkage": "oas", "reg_eps": 1e-6},
        pipeline_cfg={"dim_threshold_adaptive": 25, "reduce": "auto", "retain_variance": 0.95, "cap_components": 30},
        random_state=42,
    )
    
    assert len(S) == 30
    assert info["config"]["distance_cfg"]["m_neighbors"] == 20
    assert info["config"]["pipeline_cfg"]["dim_threshold_adaptive"] == 25


def test_fit_ddc_coreset_legacy_kwargs():
    """Test legacy kwargs mapping."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    import warnings
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        S, w, info = fit_ddc_coreset(
            X, k=30, m_neighbors=24, mode="euclidean", random_state=42
        )
        
        # Should emit deprecation warning
        assert len(w_list) > 0
        assert any("deprecated" in str(warning.message).lower() for warning in w_list)


def test_fit_ddc_coreset_reweight_full_false():
    """Test with reweight_full=False."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    S, w, info = fit_ddc_coreset(
        X, k=30, mode="euclidean", reweight_full=False, random_state=42
    )
    
    assert len(S) == 30
    assert np.allclose(w.sum(), 1.0)


def test_fit_random_coreset_basic():
    """Test random coreset."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    S, w, info = fit_random_coreset(X, k=30, random_state=42)
    
    assert len(S) == 30
    assert np.allclose(w.sum(), 1.0)
    assert info.method == "random"


def test_fit_stratified_coreset_basic():
    """Test stratified coreset."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    strata = np.random.randint(0, 3, size=200)
    
    S, w, info = fit_stratified_coreset(X, strata=strata, k=30, random_state=42)
    
    assert len(S) == 30
    assert np.allclose(w.sum(), 1.0)
    assert info.method == "stratified"


def test_fit_kmedoids_coreset_basic():
    """Test k-medoids coreset."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    S, w, info = fit_kmedoids_coreset(X, k=30, random_state=42)
    
    assert len(S) == 30
    assert np.allclose(w.sum(), 1.0)
    assert info.method == "kmedoids"


def test_fit_ddc_coreset_high_dim_pca():
    """Test high-dimensional data triggers PCA."""
    np.random.seed(42)
    X = np.random.randn(200, 60)
    
    S, w, info = fit_ddc_coreset(X, k=30, mode="auto", preset="balanced", random_state=42)
    
    assert len(S) == 30
    assert S.shape[1] == 60  # Should be in original space
    assert info["pipeline"]["pca_used"]
    assert info["pipeline"]["d_effective"] < 60


def test_fit_ddc_coreset_adaptive_shrinkage_methods():
    """Test different shrinkage methods."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    for shrinkage in ["oas", "lw", "none"]:
        S, w, info = fit_ddc_coreset(
            X,
            k=30,
            mode="adaptive",
            preset="manual",
            distance_cfg={"shrinkage": shrinkage},
            random_state=42,
        )
        assert len(S) == 30
        assert info["config"]["distance_cfg"]["shrinkage"] == shrinkage


def test_fit_ddc_coreset_adaptive_iterations():
    """Test different iteration counts."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    
    for iterations in [1, 2, 3]:
        S, w, info = fit_ddc_coreset(
            X,
            k=30,
            mode="adaptive",
            preset="manual",
            distance_cfg={"iterations": iterations},
            random_state=42,
        )
        assert len(S) == 30
        assert info["config"]["distance_cfg"]["iterations"] == iterations


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("Running tests without pytest...")
        test_density_knn_euclidean_basic()
        print("✓ test_density_knn_euclidean_basic passed")
        
        test_density_knn_euclidean_small_n()
        print("✓ test_density_knn_euclidean_small_n passed")
        
        test_select_reps_greedy_basic()
        print("✓ test_select_reps_greedy_basic passed")
        
        test_select_reps_greedy_k_equals_n()
        print("✓ test_select_reps_greedy_k_equals_n passed")
        
        test_soft_assign_weights_basic()
        print("✓ test_soft_assign_weights_basic passed")
        
        test_medoid_refinement_basic()
        print("✓ test_medoid_refinement_basic passed")
        
        test_fit_ddc_coreset_all_data()
        print("✓ test_fit_ddc_coreset_all_data passed")
        
        test_fit_ddc_coreset_n0_larger_than_n()
        print("✓ test_fit_ddc_coreset_n0_larger_than_n passed")
        
        test_fit_ddc_coreset_different_presets()
        print("✓ test_fit_ddc_coreset_different_presets passed")
        
        test_fit_ddc_coreset_manual_config()
        print("✓ test_fit_ddc_coreset_manual_config passed")
        
        test_fit_ddc_coreset_legacy_kwargs()
        print("✓ test_fit_ddc_coreset_legacy_kwargs passed")
        
        test_fit_ddc_coreset_reweight_full_false()
        print("✓ test_fit_ddc_coreset_reweight_full_false passed")
        
        test_fit_random_coreset_basic()
        print("✓ test_fit_random_coreset_basic passed")
        
        test_fit_stratified_coreset_basic()
        print("✓ test_fit_stratified_coreset_basic passed")
        
        test_fit_kmedoids_coreset_basic()
        print("✓ test_fit_kmedoids_coreset_basic passed")
        
        test_fit_ddc_coreset_high_dim_pca()
        print("✓ test_fit_ddc_coreset_high_dim_pca passed")
        
        test_fit_ddc_coreset_adaptive_shrinkage_methods()
        print("✓ test_fit_ddc_coreset_adaptive_shrinkage_methods passed")
        
        test_fit_ddc_coreset_adaptive_iterations()
        print("✓ test_fit_ddc_coreset_adaptive_iterations passed")
        
        print("\n✅ All tests passed!")

