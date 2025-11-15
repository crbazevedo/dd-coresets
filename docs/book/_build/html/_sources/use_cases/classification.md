# Classification Problems

## Use Case: Model Prototyping on Large Datasets

### The Problem

You're developing a machine learning model on a large dataset (e.g., 50GB, 10M rows). Training on the full dataset takes 8 hours per iteration, making prototyping slow and expensive. You need:

- A small subset that preserves the distribution
- Class proportions maintained (for supervised learning)
- Fast iteration cycles for hyperparameter tuning

### The Solution: Label-Aware DDC

Label-aware DDC creates a coreset that preserves both distributional properties and class proportions, making it ideal for supervised learning tasks.

## Example: Credit Risk Classification

```python
from dd_coresets import fit_ddc_coreset_by_label
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# Load your dataset
# X, y = load_credit_data()  # 1M rows, binary classification

# For demonstration, generate synthetic data
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=100000,
    n_features=20,
    n_informative=10,
    weights=[0.9, 0.1],  # Imbalanced: 90% class 0, 10% class 1
    random_state=42
)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Create label-aware coreset
S_train, w_train, info = fit_ddc_coreset_by_label(
    X_train, y_train,
    k_total=1000,  # 1000 representatives total
    mode="auto",
    preset="balanced"
)

print(f"Compressed {len(X_train):,} training points to {len(S_train):,}")
print(f"Class proportions: {info['k_per_class']}")
print(f"Original proportions: {info['n_per_class']}")
```

## Why Label-Aware DDC Works

**Preserves Class Proportions**: Unlike global DDC (which may distort class balance), label-aware DDC applies DDC separately within each class, preserving class proportions by design.

**Preserves Within-Class Structure**: Each class gets its own density-diversity coreset, preserving the distribution within each class while maintaining class balance.

**Better Model Performance**: Experiments show that label-aware DDC often leads to better model performance (AUC) than random or stratified sampling, especially when classes have complex within-class structure.

## Training Models on Coresets

```python
# Extract labels for coreset points
# (representatives are from original data, so we can map back)
y_coreset = y_train[info['selected_indices_per_class']]

# Train on full data (baseline)
model_full = LogisticRegression(random_state=42)
model_full.fit(X_train, y_train)
auc_full = roc_auc_score(y_test, model_full.predict_proba(X_test)[:, 1])

# Train on coreset (weighted)
model_coreset = LogisticRegression(random_state=42)
model_coreset.fit(S_train, y_coreset, sample_weight=w_train * len(S_train))
auc_coreset = roc_auc_score(y_test, model_coreset.predict_proba(X_test)[:, 1])

print(f"Full data AUC: {auc_full:.4f}")
print(f"Coreset AUC: {auc_coreset:.4f}")
print(f"Performance difference: {(auc_coreset - auc_full) * 100:+.2f}%")
```

**What to expect**: For well-structured data, coreset performance is typically within 1-3% of full data performance, but training is 10-100× faster.

## Conceptual Note: Why Label-Aware?

Standard DDC is unsupervised—it only looks at features, not labels. This can distort class proportions because:

- DDC selects from dense regions
- If one class is denser, it gets more representatives
- Class proportions change, affecting model training

**Label-aware solution**: By applying DDC separately within each class, we:
1. Preserve class proportions (allocate k per class proportionally)
2. Preserve within-class structure (each class gets its own density-diversity coreset)
3. Maintain distributional fidelity (weighted coreset still approximates full distribution)

This is similar to stratified sampling, but with the added benefit of distributional preservation within each class.

## Performance Comparison

For a dataset with 1M training examples:

- **Full data**: Training takes 2 hours, 8GB memory
- **Random sample (10k)**: Training takes 2 minutes, but may miss rare classes
- **Stratified sample (10k)**: Training takes 2 minutes, preserves proportions but not distribution
- **Label-aware DDC (1k)**: Training takes 30 seconds, preserves both proportions and distribution

## Best Practices

1. **Always use label-aware for supervised tasks**: Use `fit_ddc_coreset_by_label` instead of `fit_ddc_coreset`
2. **Choose k based on class balance**: Ensure `k_total >= 2 × number_of_classes` to allow at least 1-2 points per class
3. **Use weights in training**: Pass `sample_weight=w * len(S)` to sklearn models
4. **Compare baselines**: Always compare with Random and Stratified to understand trade-offs

## Further Reading

- [Label-Aware Classification Tutorial](../tutorials/label_aware_classification.md) - Complete example
- [Weighting](../concepts/weighting.md) - How weights work
- [Understanding Metrics](../guides/understanding_metrics.md) - How to evaluate coresets

