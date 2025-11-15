#!/usr/bin/env python3
"""
Generate basic_tabular.ipynb - Basic usage with simple tabular data.

This notebook demonstrates:
- Simple tabular data (Gaussian mixture, 2-3 clusters, 5-10 features)
- Basic fit_ddc_coreset usage with default parameters
- Comparison: DDC vs Random vs Stratified
- Metrics: Mean/Cov/Corr errors, Wasserstein-1 marginals
- Visualizations: 2D scatter (UMAP), marginal histograms, metrics table
"""

import json

# Helper functions for metrics (will be included in notebook)
WASSERSTEIN_1D_CODE = '''def wasserstein_1d_approx(x1, x2, w2=None, n_samples=5000):
    """Approximate Wasserstein-1 distance for 1D distributions."""
    if w2 is not None:
        probs = w2 / w2.sum()
        idx = np.random.choice(len(x2), size=n_samples, p=probs, replace=True)
        x2_sampled = x2[idx]
    else:
        x2_sampled = x2
    
    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2_sampled)
    n = min(len(x1_sorted), len(x2_sorted))
    quantiles = np.linspace(0, 1, n)
    q1 = np.quantile(x1_sorted, quantiles)
    q2 = np.quantile(x2_sorted, quantiles)
    return np.abs(q1 - q2).mean()


def ks_1d_approx(x1, x2, w2=None, n_grid=512):
    """Approximate KS statistic for 1D distributions."""
    x_min = min(x1.min(), x2.min())
    x_max = max(x1.max(), x2.max())
    grid = np.linspace(x_min, x_max, n_grid)
    
    F_X = np.array([np.mean(x1 <= x) for x in grid])
    
    if w2 is not None:
        F_S = np.array([np.sum(w2[x2 <= x]) for x in grid])
    else:
        F_S = np.array([np.mean(x2 <= x) for x in grid])
    
    return float(np.max(np.abs(F_X - F_S)))


def weighted_mean(S, w):
    """Compute weighted mean."""
    return (S * w[:, None]).sum(axis=0)


def weighted_cov(S, w):
    """Compute weighted covariance matrix."""
    mu = weighted_mean(S, w)
    Xc = S - mu
    return (Xc * w[:, None]).T @ Xc


def compute_metrics(X_full, S, w, method_name):
    """Compute all metrics comparing coreset to full data."""
    # Joint distribution metrics
    mu_full = X_full.mean(axis=0)
    cov_full = np.cov(X_full, rowvar=False)
    
    mu_coreset = weighted_mean(S, w)
    cov_coreset = weighted_cov(S, w)
    
    mean_err = np.linalg.norm(mu_full - mu_coreset)
    cov_err = np.linalg.norm(cov_full - cov_coreset, ord='fro')
    
    # Correlation matrices
    std_full = np.sqrt(np.diag(cov_full))
    std_core = np.sqrt(np.diag(cov_coreset))
    corr_full = cov_full / (std_full[:, None] * std_full[None, :] + 1e-12)
    corr_core = cov_coreset / (std_core[:, None] * std_core[None, :] + 1e-12)
    corr_err = np.linalg.norm(corr_full - corr_core, ord='fro')
    
    # Marginal distribution metrics
    d = X_full.shape[1]
    W1_dims = []
    KS_dims = []
    
    for dim in range(d):
        W1 = wasserstein_1d_approx(X_full[:, dim], S[:, dim], w)
        KS = ks_1d_approx(X_full[:, dim], S[:, dim], w)
        W1_dims.append(W1)
        KS_dims.append(KS)
    
    return {
        'method': method_name,
        'mean_err_l2': mean_err,
        'cov_err_fro': cov_err,
        'corr_err_fro': corr_err,
        'W1_mean': np.mean(W1_dims),
        'W1_max': np.max(W1_dims),
        'KS_mean': np.mean(KS_dims),
        'KS_max': np.max(KS_dims),
    }
'''

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Basic Tabular Data: DDC vs Random vs Stratified\n",
                "\n",
                "This notebook demonstrates the basic usage of `dd-coresets` on simple tabular data. ",
                "We'll compare **Density-Diversity Coresets (DDC)** with **Random** and **Stratified** sampling.\n",
                "\n",
                "## What You'll Learn\n",
                "\n",
                "- How to install and use `dd-coresets`\n",
                "- Basic API: `fit_ddc_coreset`, `fit_random_coreset`, `fit_stratified_coreset`\n",
                "- Understanding distributional metrics (Mean, Covariance, Wasserstein-1)\n",
                "- When DDC is better than Random (clustered data)\n",
                "\n",
                "## The Dataset\n",
                "\n",
                "We'll use a simple **Gaussian mixture** with 3 clusters and 8 features. ",
                "This structure is common in real-world data and demonstrates DDC's advantage."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install dd-coresets (uncomment if needed)\n",
                "# !pip install dd-coresets\n",
                "\n",
                "# For Kaggle/Colab, you may need:\n",
                "# !pip install dd-coresets --quiet"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.datasets import make_blobs\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.decomposition import PCA\n",
                "\n",
                "# Try importing UMAP, fallback to PCA if not available\n",
                "try:\n",
                "    import umap\n",
                "    HAS_UMAP = True\n",
                "except ImportError:\n",
                "    HAS_UMAP = False\n",
                "    print(\"UMAP not available, using PCA for visualization\")\n",
                "\n",
                "from dd_coresets import (\n",
                "    fit_ddc_coreset,\n",
                "    fit_random_coreset,\n",
                "    fit_stratified_coreset,\n",
                ")\n",
                "\n",
                "# Set random seed for reproducibility\n",
                "RANDOM_STATE = 42\n",
                "np.random.seed(RANDOM_STATE)\n",
                "\n",
                "# Set plotting style\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette(\"husl\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Generate Dataset\n",
                "\n",
                "We'll create a Gaussian mixture with **3 clusters** and **8 features**. ",
                "The clusters are well-separated, which is where DDC typically excels."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate Gaussian mixture\n",
                "n_samples = 10000\n",
                "n_features = 8\n",
                "n_clusters = 3\n",
                "\n",
                "X, cluster_labels = make_blobs(\n",
                "    n_samples=n_samples,\n",
                "    n_features=n_features,\n",
                "    centers=n_clusters,\n",
                "    cluster_std=1.5,\n",
                "    center_box=(-10, 10),\n",
                "    random_state=RANDOM_STATE,\n",
                ")\n",
                "\n",
                "# Standardize features\n",
                "scaler = StandardScaler()\n",
                "X = scaler.fit_transform(X)\n",
                "\n",
                "print(f\"Dataset shape: {X.shape}\")\n",
                "print(f\"Number of clusters: {n_clusters}\")\n",
                "print(f\"Cluster sizes: {np.bincount(cluster_labels)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Visualize Original Data\n",
                "\n",
                "Let's visualize the data in 2D using UMAP (or PCA if UMAP is not available)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Project to 2D for visualization\n",
                "if HAS_UMAP:\n",
                "    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)\n",
                "    X_2d = reducer.fit_transform(X)\n",
                "else:\n",
                "    reducer = PCA(n_components=2, random_state=RANDOM_STATE)\n",
                "    X_2d = reducer.fit_transform(X)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, \n",
                "                      cmap='viridis', alpha=0.5, s=10)\n",
                "plt.colorbar(scatter, label='Cluster')\n",
                "plt.title('Original Data (2D Projection)', fontsize=14, fontweight='bold')\n",
                "plt.xlabel('Component 1')\n",
                "plt.ylabel('Component 2')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Fit Coresets\n",
                "\n",
                "Now we'll create coresets using three methods:\n",
                "\n",
                "1. **DDC**: Density-Diversity Coreset (unsupervised)\n",
                "2. **Random**: Uniform random sampling\n",
                "3. **Stratified**: Stratified sampling by cluster\n",
                "\n",
                "We'll use `k=200` representatives (2% of the data)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "k = 200  # Number of representatives\n",
                "\n",
                "print(\"Fitting coresets...\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# DDC (using default mode='euclidean', simplest preset)\n",
                "S_ddc, w_ddc, info_ddc = fit_ddc_coreset(\n",
                "    X, k=k, mode='euclidean', random_state=RANDOM_STATE\n",
                ")\n",
                "print(f\"✓ DDC: {len(S_ddc)} representatives\")\n",
                "print(f\"  Pipeline: {info_ddc['pipeline']}\")\n",
                "\n",
                "# Random\n",
                "S_random, w_random, info_random = fit_random_coreset(\n",
                "    X, k=k, random_state=RANDOM_STATE + 1\n",
                ")\n",
                "print(f\"✓ Random: {len(S_random)} representatives\")\n",
                "\n",
                "# Stratified (by cluster)\n",
                "S_strat, w_strat, info_strat = fit_stratified_coreset(\n",
                "    X, strata=cluster_labels, k=k, random_state=RANDOM_STATE + 2\n",
                ")\n",
                "print(f\"✓ Stratified: {len(S_strat)} representatives\")\n",
                "print(\"=\" * 60)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Compute Metrics\n",
                "\n",
                "We'll compute distributional metrics to compare how well each coreset preserves the original distribution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                WASSERSTEIN_1D_CODE
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Compute metrics for all methods\n",
                "metrics_ddc = compute_metrics(X, S_ddc, w_ddc, 'DDC')\n",
                "metrics_random = compute_metrics(X, S_random, w_random, 'Random')\n",
                "metrics_strat = compute_metrics(X, S_strat, w_strat, 'Stratified')\n",
                "\n",
                "# Create comparison table\n",
                "results_df = pd.DataFrame([metrics_ddc, metrics_random, metrics_strat])\n",
                "results_df = results_df.set_index('method')\n",
                "\n",
                "print(\"\\nDistributional Metrics Comparison:\")\n",
                "print(\"=\" * 60)\n",
                "print(results_df.round(4))\n",
                "\n",
                "# Compute relative improvement\n",
                "print(\"\\nDDC Improvement over Random:\")\n",
                "print(\"=\" * 60)\n",
                "for metric in ['mean_err_l2', 'cov_err_fro', 'corr_err_fro', 'W1_mean', 'KS_mean']:\n",
                "    random_val = metrics_random[metric]\n",
                "    ddc_val = metrics_ddc[metric]\n",
                "    improvement = (1 - ddc_val / random_val) * 100\n",
                "    print(f\"{metric:20s}: {improvement:6.1f}% better\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Visualizations\n",
                "\n",
                "Let's visualize the coresets and compare their distributional fidelity."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Project coresets to 2D using same reducer\n",
                "S_ddc_2d = reducer.transform(S_ddc)\n",
                "S_random_2d = reducer.transform(S_random)\n",
                "S_strat_2d = reducer.transform(S_strat)\n",
                "\n",
                "# Get cluster labels for coreset points\n",
                "from sklearn.neighbors import NearestNeighbors\n",
                "nn = NearestNeighbors(n_neighbors=1)\n",
                "nn.fit(X)\n",
                "\n",
                "_, idx_ddc = nn.kneighbors(S_ddc)\n",
                "_, idx_random = nn.kneighbors(S_random)\n",
                "_, idx_strat = nn.kneighbors(S_strat)\n",
                "\n",
                "labels_ddc = cluster_labels[idx_ddc.flatten()]\n",
                "labels_random = cluster_labels[idx_random.flatten()]\n",
                "labels_strat = cluster_labels[idx_strat.flatten()]\n",
                "\n",
                "# Plot spatial coverage\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "for ax, S_2d, labels, title, w in zip(\n",
                "    axes, [S_ddc_2d, S_random_2d, S_strat_2d],\n",
                "    [labels_ddc, labels_random, labels_strat],\n",
                "    ['DDC', 'Random', 'Stratified'],\n",
                "    [w_ddc, w_random, w_strat]\n",
                "):\n",
                "    scatter = ax.scatter(S_2d[:, 0], S_2d[:, 1], c=labels, \n",
                "                         cmap='viridis', s=w*1000, alpha=0.7, edgecolors='black', linewidths=0.5)\n",
                "    ax.set_title(f'{title} Coreset (k={len(S_2d)})', fontsize=12, fontweight='bold')\n",
                "    ax.set_xlabel('Component 1')\n",
                "    ax.set_ylabel('Component 2')\n",
                "    ax.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot marginal distributions for first 4 features\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for dim in range(4):\n",
                "    ax = axes[dim]\n",
                "    \n",
                "    # Full data histogram\n",
                "    ax.hist(X[:, dim], bins=50, alpha=0.3, color='gray', label='Full Data', density=True)\n",
                "    \n",
                "    # Weighted histograms for coresets\n",
                "    for S, w, label, color in [\n",
                "        (S_ddc, w_ddc, 'DDC', 'blue'),\n",
                "        (S_random, w_random, 'Random', 'orange'),\n",
                "        (S_strat, w_strat, 'Stratified', 'green')\n",
                "    ]:\n",
                "        # Sample from weighted distribution\n",
                "        n_samples = 5000\n",
                "        probs = w / w.sum()\n",
                "        idx = np.random.choice(len(S), size=n_samples, p=probs, replace=True)\n",
                "        ax.hist(S[idx, dim], bins=30, alpha=0.5, label=label, \n",
                "                color=color, density=True, histtype='step', linewidth=2)\n",
                "    \n",
                "    ax.set_title(f'Feature {dim+1} Marginal Distribution', fontweight='bold')\n",
                "    ax.set_xlabel('Value')\n",
                "    ax.set_ylabel('Density')\n",
                "    ax.legend()\n",
                "    ax.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Bar chart comparing metrics\n",
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "metrics_to_plot = ['mean_err_l2', 'cov_err_fro', 'W1_mean', 'KS_mean']\n",
                "methods = ['DDC', 'Random', 'Stratified']\n",
                "colors = ['blue', 'orange', 'green']\n",
                "\n",
                "x = np.arange(len(metrics_to_plot))\n",
                "width = 0.25\n",
                "\n",
                "for i, method in enumerate(methods):\n",
                "    metrics = [results_df.loc[method, m] for m in metrics_to_plot]\n",
                "    axes[0].bar(x + i*width, metrics, width, label=method, color=colors[i], alpha=0.7)\n",
                "\n",
                "axes[0].set_xlabel('Metric')\n",
                "axes[0].set_ylabel('Error (lower is better)')\n",
                "axes[0].set_title('Distributional Metrics Comparison', fontweight='bold')\n",
                "axes[0].set_xticks(x + width)\n",
                "axes[0].set_xticklabels(metrics_to_plot, rotation=45, ha='right')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True, alpha=0.3, axis='y')\n",
                "\n",
                "# Improvement percentages\n",
                "improvements = []\n",
                "for metric in metrics_to_plot:\n",
                "    random_val = results_df.loc['Random', metric]\n",
                "    ddc_val = results_df.loc['DDC', metric]\n",
                "    improvement = (1 - ddc_val / random_val) * 100\n",
                "    improvements.append(improvement)\n",
                "\n",
                "axes[1].bar(range(len(metrics_to_plot)), improvements, color='blue', alpha=0.7)\n",
                "axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)\n",
                "axes[1].set_xlabel('Metric')\n",
                "axes[1].set_ylabel('Improvement (%)')\n",
                "axes[1].set_title('DDC Improvement over Random', fontweight='bold')\n",
                "axes[1].set_xticks(range(len(metrics_to_plot)))\n",
                "axes[1].set_xticklabels(metrics_to_plot, rotation=45, ha='right')\n",
                "axes[1].grid(True, alpha=0.3, axis='y')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Key Takeaways\n",
                "\n",
                "### When DDC is Better\n",
                "\n",
                "- **Clustered data**: DDC preserves cluster structure better than Random\n",
                "- **Spatial coverage**: DDC ensures all clusters are represented\n",
                "- **Distributional fidelity**: DDC better preserves marginal distributions\n",
                "\n",
                "### When Random Might Be Better\n",
                "\n",
                "- **Very large datasets** (n >> k) with complex non-Gaussian structure\n",
                "- **Preserving exact global covariance** is critical\n",
                "- **High-dimensional sparse data**\n",
                "\n",
                "### Next Steps\n",
                "\n",
                "- Try `multimodal_clusters.ipynb` for more complex cluster structures\n",
                "- Try `adaptive_distances.ipynb` for advanced features (presets, adaptive distances)\n",
                "- See `docs/DDC_ADVANTAGE_CASES.md` for comprehensive analysis"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

if __name__ == "__main__":
    output_path = "examples/basic_tabular.ipynb"
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"✓ Created {output_path}")

