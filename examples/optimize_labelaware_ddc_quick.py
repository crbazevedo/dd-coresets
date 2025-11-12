#!/usr/bin/env python3
"""
Quick optimization test (reduced grid) for Label-Aware DDC parameters.
"""

import numpy as np
import pandas as pd
import sys
import os
import json
from pathlib import Path
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from dd_coresets import fit_ddc_coreset

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def weighted_mean(S, w):
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    return (S * w[:, None]).sum(axis=0)


def weighted_cov(S, w):
    S = np.asarray(S, dtype=float)
    w = np.asarray(w, dtype=float)
    mu = weighted_mean(S, w)
    Xc = S - mu
    cov = (Xc * w[:, None]).T @ Xc
    return cov


def corr_from_cov(cov):
    cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv_std = 1.0 / std
    C = cov * inv_std[:, None] * inv_std[None, :]
    return C


def evaluate_labelaware_ddc_params(X_train, y_train, k_reps, alpha, gamma, m_neighbors, refine_iters, X_val=None, y_val=None):
    S_labelaware_list = []
    w_labelaware_list = []
    y_labelaware_list = []
    
    for class_label in np.unique(y_train):
        class_mask = (y_train == class_label)
        X_class = X_train[class_mask]
        p_class = np.sum(class_mask) / len(y_train)
        k_class = max(1, int(round(k_reps * p_class)))
        n0_class = min(20_000, len(X_class))
        m_neighbors_class = max(5, min(m_neighbors, len(X_class) // 10))
        
        try:
            S_class, w_class, _ = fit_ddc_coreset(
                X_class, k=k_class, n0=n0_class,
                alpha=alpha, m_neighbors=m_neighbors_class,
                gamma=gamma, refine_iters=refine_iters,
                reweight_full=True, random_state=RANDOM_STATE + class_label,
            )
            w_class_scaled = w_class * p_class
            S_labelaware_list.append(S_class)
            w_labelaware_list.append(w_class_scaled)
            y_labelaware_list.append(np.full(len(S_class), class_label))
        except:
            return {'cov_err_fro': np.inf, 'corr_err_fro': np.inf, 'downstream_auc': 0.0}
    
    S_labelaware = np.vstack(S_labelaware_list)
    w_labelaware = np.concatenate(w_labelaware_list)
    w_labelaware = w_labelaware / w_labelaware.sum()
    y_labelaware = np.concatenate(y_labelaware_list)
    
    mu_full = X_train.mean(axis=0)
    cov_full = np.cov(X_train, rowvar=False)
    corr_full = corr_from_cov(cov_full)
    
    mu_coreset = weighted_mean(S_labelaware, w_labelaware)
    cov_coreset = weighted_cov(S_labelaware, w_labelaware)
    corr_coreset = corr_from_cov(cov_coreset)
    
    cov_err = np.linalg.norm(cov_full - cov_coreset, ord='fro')
    corr_err = np.linalg.norm(corr_full - corr_coreset, ord='fro')
    
    auc = 0.0
    if X_val is not None and y_val is not None:
        try:
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=None)
            sample_weights = w_labelaware * len(X_train)
            lr.fit(S_labelaware, y_labelaware, sample_weight=sample_weights)
            y_pred_proba = lr.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
        except:
            pass
    
    return {'cov_err_fro': cov_err, 'corr_err_fro': corr_err, 'downstream_auc': auc}


# Quick test
X, y = make_classification(n_samples=10_000, n_features=10, n_informative=5, n_redundant=2,
                          n_clusters_per_class=2, weights=[0.75, 0.25], random_state=RANDOM_STATE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

k_reps = 500  # Smaller for speed

# Reduced grid
param_grid = {
    'alpha': [0.2, 0.3, 0.4],
    'gamma': [0.5, 1.0, 1.5],
    'm_neighbors': [16, 32],
    'refine_iters': [1, 2],
}

results = []
for alpha, gamma, m_neighbors, refine_iters in product(param_grid['alpha'], param_grid['gamma'], 
                                                        param_grid['m_neighbors'], param_grid['refine_iters']):
    metrics = evaluate_labelaware_ddc_params(X_train, y_train, k_reps, alpha, gamma, m_neighbors, refine_iters, X_val, y_val)
    results.append({
        'alpha': alpha, 'gamma': gamma, 'm_neighbors': m_neighbors, 'refine_iters': refine_iters,
        'cov_err_fro': metrics['cov_err_fro'], 'corr_err_fro': metrics['corr_err_fro'],
        'downstream_auc': metrics['downstream_auc'],
        'composite_score': metrics['cov_err_fro'] + 0.5 * metrics['corr_err_fro'],
    })

results_df = pd.DataFrame(results).sort_values('composite_score')
print("Quick test results (top 5):")
print(results_df.head(5).to_string(index=False))

best = results_df.iloc[0]
best_params = {
    'alpha': float(best['alpha']),
    'gamma': float(best['gamma']),
    'm_neighbors': int(best['m_neighbors']),
    'refine_iters': int(best['refine_iters']),
}

print(f"\nBest (quick test): alpha={best_params['alpha']:.1f}, gamma={best_params['gamma']:.1f}, "
      f"m_neighbors={best_params['m_neighbors']}, refine_iters={best_params['refine_iters']}")

Path(__file__).parent.joinpath("best_parameters_quick.json").write_text(json.dumps(best_params, indent=2))

