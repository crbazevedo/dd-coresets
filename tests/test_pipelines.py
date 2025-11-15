"""
Tests for pipeline functions and presets.
"""

import numpy as np
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from sklearn.datasets import make_blobs

from dd_coresets.pipelines import (
    PRESETS,
    choose_pipeline,
    reduce_dimensionality_if_needed,
    fit_ddc_coreset_by_label,
)


def test_presets_exist():
    """Test that all presets are defined."""
    assert "fast" in PRESETS
    assert "balanced" in PRESETS
    assert "robust" in PRESETS
    
    for preset_name, preset_config in PRESETS.items():
        assert "distance_cfg" in preset_config
        assert "pipeline_cfg" in preset_config
        assert "m_neighbors" in preset_config["distance_cfg"]
        assert "iterations" in preset_config["distance_cfg"]
        assert "shrinkage" in preset_config["distance_cfg"]
        assert "reg_eps" in preset_config["distance_cfg"]


def test_choose_pipeline_euclidean():
    """Test pipeline choice for euclidean mode."""
    # Euclidean mode should always return euclidean
    for d in [5, 20, 50, 100]:
        decision = choose_pipeline(d=d, m_neighbors=32, mode="euclidean", dim_threshold_adaptive=30)
        assert not decision["adaptive"]
        assert not decision["do_pca"]


def test_choose_pipeline_adaptive():
    """Test pipeline choice for adaptive mode."""
    # Low d, feasible
    decision = choose_pipeline(d=10, m_neighbors=32, mode="adaptive", dim_threshold_adaptive=30)
    assert decision["adaptive"]
    assert not decision["do_pca"]
    
    # Low d, not feasible (m_neighbors <= d)
    decision = choose_pipeline(d=50, m_neighbors=32, mode="adaptive", dim_threshold_adaptive=30)
    assert not decision["adaptive"]
    assert "fallback_reason" in decision


def test_choose_pipeline_auto():
    """Test pipeline choice for auto mode."""
    # d < 20: Euclidean
    decision = choose_pipeline(d=10, m_neighbors=32, mode="auto", dim_threshold_adaptive=30)
    assert not decision["adaptive"]
    assert not decision["do_pca"]
    
    # 20 <= d < 30: Adaptive if feasible
    decision = choose_pipeline(d=25, m_neighbors=32, mode="auto", dim_threshold_adaptive=30)
    assert decision["adaptive"]
    assert not decision["do_pca"]
    
    # d >= 30: PCA + Adaptive
    decision = choose_pipeline(d=60, m_neighbors=32, mode="auto", dim_threshold_adaptive=30)
    assert decision["adaptive"]
    assert decision["do_pca"]


def test_reduce_dimensionality_none():
    """Test that reduce='none' doesn't reduce."""
    X = np.random.randn(100, 50)
    X_reduced, info = reduce_dimensionality_if_needed(
        X, reduce="none", retain_variance=0.95, cap_components=30
    )
    assert X_reduced.shape == X.shape
    assert info["pca_model"] is None


def test_reduce_dimensionality_pca():
    """Test PCA reduction."""
    # Create data with clear structure
    X, _ = make_blobs(n_samples=200, n_features=50, centers=3, random_state=42)
    
    X_reduced, info = reduce_dimensionality_if_needed(
        X, reduce="pca", retain_variance=0.95, cap_components=30
    )
    
    assert X_reduced.shape[0] == X.shape[0]
    assert X_reduced.shape[1] < X.shape[1]
    assert X_reduced.shape[1] <= 30
    assert info["pca_model"] is not None
    assert info["n_components"] <= 30
    assert info["explained_variance_ratio"] is not None


def test_reduce_dimensionality_auto():
    """Test auto reduction (should use PCA for high-d)."""
    X = np.random.randn(100, 60)
    X_reduced, info = reduce_dimensionality_if_needed(
        X, reduce="auto", retain_variance=0.95, cap_components=30
    )
    
    # Should reduce
    assert X_reduced.shape[1] < X.shape[1]
    assert info["pca_model"] is not None


def test_fit_ddc_coreset_by_label_basic():
    """Test basic label-wise wrapper."""
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, n_features=5, centers=3, random_state=42)
    
    S, w, info = fit_ddc_coreset_by_label(
        X, y, k_total=30, mode="euclidean", preset="balanced", random_state=42
    )
    
    assert len(S) == 30
    assert np.allclose(w.sum(), 1.0)
    assert len(info["classes"]) == 3
    assert sum(info["k_per_class"]) == 30


def test_fit_ddc_coreset_by_label_preserves_proportions():
    """Test that label-wise wrapper preserves class proportions."""
    np.random.seed(42)
    # Create imbalanced dataset
    X1, y1 = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)
    X2, y2 = make_blobs(n_samples=200, n_features=5, centers=1, random_state=43)
    X3, y3 = make_blobs(n_samples=50, n_features=5, centers=1, random_state=44)
    
    X = np.vstack([X1, X2, X3])
    y = np.concatenate([np.zeros(100), np.ones(200), np.full(50, 2)])
    
    # Original proportions: 100/350, 200/350, 50/350
    props_original = np.array([100, 200, 50]) / 350
    
    S, w, info = fit_ddc_coreset_by_label(
        X, y, k_total=100, mode="euclidean", preset="balanced", random_state=42
    )
    
    props_coreset = np.array(info["k_per_class"]) / sum(info["k_per_class"])
    
    # Should be approximately preserved (within 5%)
    assert np.allclose(props_original, props_coreset, atol=0.05)


def test_fit_ddc_coreset_by_label_single_class():
    """Test label-wise wrapper with single class."""
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, n_features=5, centers=1, random_state=42)
    y = np.zeros(len(y))
    
    S, w, info = fit_ddc_coreset_by_label(
        X, y, k_total=20, mode="euclidean", random_state=42
    )
    
    assert len(S) == 20
    assert len(info["classes"]) == 1
    assert info["k_per_class"][0] == 20


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("Running tests without pytest...")
        test_presets_exist()
        print("✓ test_presets_exist passed")
        
        test_choose_pipeline_euclidean()
        print("✓ test_choose_pipeline_euclidean passed")
        
        test_choose_pipeline_adaptive()
        print("✓ test_choose_pipeline_adaptive passed")
        
        test_choose_pipeline_auto()
        print("✓ test_choose_pipeline_auto passed")
        
        test_reduce_dimensionality_none()
        print("✓ test_reduce_dimensionality_none passed")
        
        test_reduce_dimensionality_pca()
        print("✓ test_reduce_dimensionality_pca passed")
        
        test_reduce_dimensionality_auto()
        print("✓ test_reduce_dimensionality_auto passed")
        
        test_fit_ddc_coreset_by_label_basic()
        print("✓ test_fit_ddc_coreset_by_label_basic passed")
        
        test_fit_ddc_coreset_by_label_preserves_proportions()
        print("✓ test_fit_ddc_coreset_by_label_preserves_proportions passed")
        
        test_fit_ddc_coreset_by_label_single_class()
        print("✓ test_fit_ddc_coreset_by_label_single_class passed")
        
        print("\n✅ All tests passed!")

