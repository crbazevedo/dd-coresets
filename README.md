# dd-coresets

**Density–Diversity Coresets (DDC)**: a small weighted set of *real* data points that approximates the empirical distribution of a large dataset.

This library exposes a simple API (in the spirit of scikit-learn) to:
- build an **unsupervised** density–diversity coreset (`fit_ddc_coreset`);
- compare against **random** and **stratified** baselines (`fit_random_coreset`, `fit_stratified_coreset`).

The goal is pragmatic: help data scientists work with large datasets using small, distribution-preserving subsets that are easy to simulate, visualise, and explain.

---

## Motivation

In practice, we rarely work on the **full dataset** for everything:

- Exploratory plots and dashboards need **small, interpretable samples**.
- Scenario analysis and simulations need **few representative points** with **weights**.
- Prototyping models and ideas is faster on **coresets** than on full data.

Common approaches:

- **Random sampling**: simple, but can miss important modes or tails.
- **Stratified sampling**: good when you already know the right strata (segments, classes, products), but needs domain knowledge and alignment with stakeholders.
- **Cluster centroids (e.g. k-means)**: minimise reconstruction error, but centroids are not real data points and are not explicitly distributional.

**DDC** sits in between:

- Unsupervised, geometry-aware.
- Selects **real points** (medoids) that cover dense regions and diverse modes.
- Learns **weights** via soft assignments, approximating the empirical distribution.

---

## Installation

```bash
git clone https://github.com/crbazevedo/dd-coresets.git
cd dd-coresets

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
