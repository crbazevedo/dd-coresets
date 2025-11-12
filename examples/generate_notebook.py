#!/usr/bin/env python3
"""
Script to generate the binary_classification_ddc.ipynb notebook.
This creates a complete, well-documented Jupyter notebook.
"""

import json

def create_notebook():
    """Create the complete notebook with all cells."""
    
    cells = []
    
    # ========== TITLE AND INTRODUCTION ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Density–Diversity Coresets (DDC) for Large Tabular Data: Global vs Label-wise\n",
            "\n",
            "This notebook demonstrates how to use the `dd-coresets` library to compress large tabular datasets while preserving distributional properties. We focus on a **binary classification** problem (credit risk) and compare:\n",
            "\n",
            "1. **Global, unsupervised DDC coreset**: Compresses the entire dataset without considering labels (may distort class proportions)\n",
            "2. **Label-wise DDC coreset**: Preserves class proportions by applying DDC separately within each class\n",
            "\n",
            "## The Practical Problem\n",
            "\n",
            "When working with millions of rows, many workflows require small, interpretable subsets:\n",
            "- **Exploratory data analysis** needs representative samples\n",
            "- **Model prototyping** is faster on coresets\n",
            "- **Scenario analysis** requires weighted representative points\n",
            "\n",
            "**Why weights matter**: A coreset is not just a set of points—it's a **weighted set** that approximates the full distribution. The weights allow us to reconstruct statistics (means, covariances, marginals) of the original dataset from just a few hundred points.\n",
            "\n",
            "**The challenge**: `dd-coresets` is **unsupervised**—it only looks at feature distributions, not labels. If used naively on the entire dataset, it can change label proportions, which is problematic for supervised learning tasks."
        ]
    })
    
    # ========== SETUP ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Setup"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install dd-coresets\n",
            "# For Google Colab: uncomment the line below\n",
            "# !pip install dd-coresets\n",
            "\n",
            "# For Kaggle: usually already available or use:\n",
            "# !pip install dd-coresets --quiet"
        ]
    })
    
    cells.append({
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
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, accuracy_score\n",
            "from sklearn.decomposition import PCA\n",
            "\n",
            "from scipy.stats import wasserstein_distance, ks_2samp\n",
            "\n",
            "from dd_coresets import fit_ddc_coreset\n",
            "\n",
            "# Set random seed for reproducibility\n",
            "RANDOM_STATE = 42\n",
            "np.random.seed(RANDOM_STATE)\n",
            "\n",
            "# Set plotting style\n",
            "try:\n",
            "    plt.style.use('seaborn-v0_8')\n",
            "except:\n",
            "    plt.style.use('seaborn')\n",
            "sns.set_palette(\"husl\")\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "plt.rcParams['font.size'] = 10\n",
            "\n",
            "print(\"✅ Imports successful\")"
        ]
    })
    
    # ========== DATA LOADING ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Loading\n",
            "\n",
            "We'll use a **credit risk style dataset** with numeric features and a binary label (default / non-default).\n",
            "\n",
            "**For Kaggle users**: Adapt the path below to your dataset location (e.g., `/kaggle/input/give-me-some-credit-dataset/cs-training.csv`).\n",
            "\n",
            "**For Colab users**: Upload your dataset or use the synthetic fallback below."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============================================\n",
            "# KAGGLE-SPECIFIC: Load from Kaggle input\n",
            "# ============================================\n",
            "# Uncomment and adapt the path below if running on Kaggle:\n",
            "#\n",
            "# import os\n",
            "# kaggle_path = \"/kaggle/input/give-me-some-credit-dataset/cs-training.csv\"\n",
            "# if os.path.exists(kaggle_path):\n",
            "#     df = pd.read_csv(kaggle_path)\n",
            "#     # Assuming the target column is named 'SeriousDlqin2yrs' or similar\n",
            "#     # Rename to 'target' for consistency\n",
            "#     if 'SeriousDlqin2yrs' in df.columns:\n",
            "#         df = df.rename(columns={'SeriousDlqin2yrs': 'target'})\n",
            "#     print(f\"✅ Loaded dataset from Kaggle: {df.shape}\")\n",
            "# else:\n",
            "#     print(\"⚠️  Kaggle path not found, using synthetic data\")\n",
            "#     USE_SYNTHETIC = True\n",
            "\n",
            "# ============================================\n",
            "# FALLBACK: Synthetic imbalanced dataset\n",
            "# ============================================\n",
            "USE_SYNTHETIC = True  # Set to False if you have real data\n",
            "\n",
            "if USE_SYNTHETIC:\n",
            "    from sklearn.datasets import make_classification\n",
            "    \n",
            "    print(\"Generating synthetic credit risk dataset...\")\n",
            "    X, y = make_classification(\n",
            "        n_samples=100_000,\n",
            "        n_features=20,\n",
            "        n_informative=10,\n",
            "        n_redundant=5,\n",
            "        n_clusters_per_class=2,\n",
            "        weights=[0.9, 0.1],  # 90% non-default, 10% default (imbalanced)\n",
            "        random_state=RANDOM_STATE,\n",
            "        class_sep=0.8,  # Moderate separation\n",
            "    )\n",
            "    \n",
            "    df = pd.DataFrame(X, columns=[f\"feature_{i}\" for i in range(X.shape[1])])\n",
            "    df[\"target\"] = y\n",
            "    print(f\"✅ Generated synthetic dataset: {df.shape}\")\n",
            "\n",
            "# Display basic info\n",
            "print(\"\\nDataset shape:\", df.shape)\n",
            "print(\"\\nLabel distribution:\")\n",
            "print(df[\"target\"].value_counts(normalize=True))\n",
            "print(\"\\nFirst few rows:\")\n",
            "df.head()"
        ]
    })
    
    # ========== PREPROCESSING ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Preprocessing\n",
            "\n",
            "We'll select numeric features, handle missing values, and scale the data. DDC requires **preprocessed numerical features**."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Select numeric features (exclude target)\n",
            "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
            "numeric_cols = [col for col in numeric_cols if col != 'target']\n",
            "\n",
            "print(f\"Selected {len(numeric_cols)} numeric features\")\n",
            "\n",
            "# Extract features and target\n",
            "X_raw = df[numeric_cols].copy()\n",
            "y_raw = df['target'].values\n",
            "\n",
            "# Handle missing values (simple mean imputation)\n",
            "if X_raw.isnull().sum().sum() > 0:\n",
            "    print(f\"\\n⚠️  Found missing values. Imputing with mean...\")\n",
            "    X_raw = X_raw.fillna(X_raw.mean())\n",
            "else:\n",
            "    print(\"\\n✅ No missing values\")\n",
            "\n",
            "# Convert to NumPy array\n",
            "X_raw = X_raw.values\n",
            "\n",
            "print(f\"\\nFeature matrix shape: {X_raw.shape}\")\n",
            "print(f\"Target vector shape: {y_raw.shape}\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Scale features (important for DDC, which uses Euclidean distances)\n",
            "scaler = StandardScaler()\n",
            "X_scaled = scaler.fit_transform(X_raw)\n",
            "\n",
            "# Split into train/test (stratified to preserve label proportions)\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X_scaled, y_raw, \n",
            "    test_size=0.3, \n",
            "    stratify=y_raw, \n",
            "    random_state=RANDOM_STATE\n",
            ")\n",
            "\n",
            "print(f\"Training set: {X_train.shape[0]:,} samples\")\n",
            "print(f\"Test set: {X_test.shape[0]:,} samples\")\n",
            "print(f\"\\nTraining label distribution:\")\n",
            "unique, counts = np.unique(y_train, return_counts=True)\n",
            "for label, count in zip(unique, counts):\n",
            "    print(f\"  Class {label}: {count:,} ({count/len(y_train)*100:.2f}%)\")"
        ]
    })
    
    # ========== FULL-DATA BASELINE ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Full-Data Baseline Model\n",
            "\n",
            "We'll train a logistic regression model on the **full training set** to establish a gold standard. This will be our baseline for comparison."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train on full training data\n",
            "lr_full = LogisticRegression(\n",
            "    max_iter=1000,\n",
            "    random_state=RANDOM_STATE,\n",
            "    class_weight=None  # No class balancing\n",
            ")\n",
            "\n",
            "lr_full.fit(X_train, y_train)\n",
            "\n",
            "# Predict on test set\n",
            "y_pred_proba_full = lr_full.predict_proba(X_test)[:, 1]\n",
            "y_pred_full = lr_full.predict(X_test)\n",
            "\n",
            "# Evaluate\n",
            "baseline_auc = roc_auc_score(y_test, y_pred_proba_full)\n",
            "baseline_brier = brier_score_loss(y_test, y_pred_proba_full)\n",
            "baseline_accuracy = accuracy_score(y_test, y_pred_full)\n",
            "\n",
            "print(\"📊 Full-Data Baseline Metrics:\")\n",
            "print(f\"  ROC AUC:  {baseline_auc:.4f}\")\n",
            "print(f\"  Brier Score: {baseline_brier:.4f}\")\n",
            "print(f\"  Accuracy:    {baseline_accuracy:.4f}\")\n",
            "\n",
            "# Store for later comparison\n",
            "baseline_metrics = {\n",
            "    'method': 'Full Data',\n",
            "    'auc': baseline_auc,\n",
            "    'brier': baseline_brier,\n",
            "    'accuracy': baseline_accuracy\n",
            "}"
        ]
    })
    
    # ========== BASELINE SUBSETS ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Baseline Subsets: Random and Stratified\n",
            "\n",
            "Before using DDC, let's establish simple baselines:\n",
            "\n",
            "- **Random subset**: Uniform sampling (may miss rare classes)\n",
            "- **Stratified subset**: Preserves class proportions (common practice in supervised learning)\n",
            "\n",
            "We'll use `k_reps = 1000` representatives (1% of training data if n_train = 100k)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "k_reps = 1000  # Number of representatives\n",
            "print(f\"Target coreset size: {k_reps} representatives ({k_reps/len(X_train)*100:.2f}% of training data)\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Random subset\n",
            "np.random.seed(RANDOM_STATE)\n",
            "random_indices = np.random.choice(len(X_train), size=k_reps, replace=False)\n",
            "X_random = X_train[random_indices]\n",
            "y_random = y_train[random_indices]\n",
            "w_random = np.ones(k_reps) / k_reps  # Uniform weights\n",
            "\n",
            "print(\"✅ Random subset created\")\n",
            "print(f\"  Shape: {X_random.shape}\")\n",
            "print(f\"  Label distribution: {np.bincount(y_random) / len(y_random)}\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Stratified subset (preserves class proportions)\n",
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "# Manual stratified sampling to get exactly k_reps\n",
            "strat_indices = []\n",
            "for class_label in np.unique(y_train):\n",
            "    class_mask = (y_train == class_label)\n",
            "    class_indices = np.where(class_mask)[0]\n",
            "    n_class = int(k_reps * np.sum(class_mask) / len(y_train))\n",
            "    selected = np.random.choice(class_indices, size=n_class, replace=False)\n",
            "    strat_indices.extend(selected)\n",
            "\n",
            "# If we don't have exactly k_reps, adjust\n",
            "if len(strat_indices) < k_reps:\n",
            "    remaining = k_reps - len(strat_indices)\n",
            "    remaining_indices = np.setdiff1d(np.arange(len(X_train)), strat_indices)\n",
            "    strat_indices.extend(np.random.choice(remaining_indices, size=remaining, replace=False))\n",
            "elif len(strat_indices) > k_reps:\n",
            "    strat_indices = np.random.choice(strat_indices, size=k_reps, replace=False)\n",
            "\n",
            "X_strat = X_train[strat_indices]\n",
            "y_strat = y_train[strat_indices]\n",
            "w_strat = np.ones(len(X_strat)) / len(X_strat)  # Uniform weights\n",
            "\n",
            "print(\"✅ Stratified subset created\")\n",
            "print(f\"  Shape: {X_strat.shape}\")\n",
            "print(f\"  Label distribution: {np.bincount(y_strat) / len(y_strat)}\")\n",
            "print(f\"  Original label distribution: {np.bincount(y_train) / len(y_train)}\")"
        ]
    })
    
    # ========== GLOBAL DDC ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Global (Unsupervised) DDC Coreset\n",
            "\n",
            "Now we'll apply DDC to the **entire training set**, ignoring labels. This is the \"naive\" approach that can distort class proportions because DDC only looks at feature distributions."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Global DDC coreset (unsupervised, ignores labels)\n",
            "n0 = min(50_000, X_train.shape[0])  # Working sample size\n",
            "\n",
            "print(f\"Fitting global DDC coreset (k={k_reps}, n0={n0})...\")\n",
            "\n",
            "S_global, w_global, info_global = fit_ddc_coreset(\n",
            "    X_train,\n",
            "    k=k_reps,\n",
            "    n0=n0,\n",
            "    alpha=0.3,  # density-diversity trade-off\n",
            "    m_neighbors=32,  # kNN for density estimation\n",
            "    gamma=1.0,  # kernel scale\n",
            "    refine_iters=1,\n",
            "    reweight_full=True,  # Reweight on full dataset\n",
            "    random_state=RANDOM_STATE,\n",
            ")\n",
            "\n",
            "print(f\"✅ Global DDC coreset created: {S_global.shape}\")\n",
            "print(f\"  Weights sum: {w_global.sum():.6f}\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Recover labels for the selected representatives\n",
            "# info_global contains the indices of selected points\n",
            "selected_indices = info_global.selected_indices\n",
            "y_global = y_train[selected_indices]\n",
            "\n",
            "print(\"📊 Label Distribution Comparison:\")\n",
            "print(f\"  Original training set:\")\n",
            "orig_props = np.bincount(y_train) / len(y_train)\n",
            "for label, prop in enumerate(orig_props):\n",
            "    print(f\"    Class {label}: {prop:.4f} ({np.sum(y_train == label):,} samples)\")\n",
            "\n",
            "print(f\"\\n  Global DDC coreset:\")\n",
            "global_props = np.bincount(y_global) / len(y_global)\n",
            "for label, prop in enumerate(global_props):\n",
            "    print(f\"    Class {label}: {prop:.4f} ({np.sum(y_global == label):,} samples)\")\n",
            "\n",
            "print(f\"\\n⚠️  Class proportion shift:\")\n",
            "for label in range(len(orig_props)):\n",
            "    shift = global_props[label] - orig_props[label]\n",
            "    print(f\"    Class {label}: {shift:+.4f} ({shift/orig_props[label]*100:+.2f}% relative change)\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Key observation**: The global DDC coreset can change label proportions because it's **unsupervised**—it only optimizes for feature distribution preservation, not label balance. This is problematic for supervised learning tasks where class proportions matter."
        ]
    })
    
    # ========== LABEL-WISE DDC ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Label-wise DDC Coreset\n",
            "\n",
            "To preserve class proportions while still benefiting from DDC's distribution-preserving properties, we apply DDC **separately within each class**. This:\n",
            "\n",
            "1. Preserves label proportions by design\n",
            "2. Maintains density–diversity structure **within each class**\n",
            "3. Still provides weighted representatives that approximate the full distribution"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Label-wise DDC: apply DDC separately to each class\n",
            "S_labelwise_list = []\n",
            "w_labelwise_list = []\n",
            "y_labelwise_list = []\n",
            "\n",
            "for class_label in np.unique(y_train):\n",
            "    # Extract data for this class\n",
            "    class_mask = (y_train == class_label)\n",
            "    X_class = X_train[class_mask]\n",
            "    \n",
            "    # Compute class proportion\n",
            "    p_class = np.sum(class_mask) / len(y_train)\n",
            "    \n",
            "    # Allocate representatives proportionally\n",
            "    k_class = max(1, int(round(k_reps * p_class)))\n",
            "    \n",
            "    print(f\"\\nClass {class_label}: {np.sum(class_mask):,} samples ({p_class:.2%})\")\n",
            "    print(f\"  Allocating {k_class} representatives...\")\n",
            "    \n",
            "    # Fit DDC on this class\n",
            "    n0_class = min(20_000, len(X_class))\n",
            "    \n",
            "    S_class, w_class, info_class = fit_ddc_coreset(\n",
            "        X_class,\n",
            "        k=k_class,\n",
            "        n0=n0_class,\n",
            "        alpha=0.3,\n",
            "        m_neighbors=32,\n",
            "        gamma=1.0,\n",
            "        refine_iters=1,\n",
            "        reweight_full=True,\n",
            "        random_state=RANDOM_STATE + class_label,  # Different seed per class\n",
            "    )\n",
            "    \n",
            "    S_labelwise_list.append(S_class)\n",
            "    w_labelwise_list.append(w_class)\n",
            "    y_labelwise_list.append(np.full(len(S_class), class_label))\n",
            "\n",
            "# Concatenate all classes\n",
            "S_labelwise = np.vstack(S_labelwise_list)\n",
            "w_labelwise = np.concatenate(w_labelwise_list)\n",
            "y_labelwise = np.concatenate(y_labelwise_list)\n",
            "\n",
            "# Renormalize weights\n",
            "w_labelwise = w_labelwise / w_labelwise.sum()\n",
            "\n",
            "print(f\"\\n✅ Label-wise DDC coreset created: {S_labelwise.shape}\")\n",
            "print(f\"  Weights sum: {w_labelwise.sum():.6f}\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Verify label proportions are preserved\n",
            "print(\"📊 Label Distribution Comparison:\")\n",
            "print(f\"  Original training set:\")\n",
            "orig_props = np.bincount(y_train) / len(y_train)\n",
            "for label, prop in enumerate(orig_props):\n",
            "    print(f\"    Class {label}: {prop:.4f}\")\n",
            "\n",
            "print(f\"\\n  Label-wise DDC coreset:\")\n",
            "labelwise_props = np.bincount(y_labelwise) / len(y_labelwise)\n",
            "for label, prop in enumerate(labelwise_props):\n",
            "    print(f\"    Class {label}: {prop:.4f}\")\n",
            "\n",
            "print(f\"\\n✅ Class proportion preservation:\")\n",
            "for label in range(len(orig_props)):\n",
            "    diff = abs(labelwise_props[label] - orig_props[label])\n",
            "    print(f\"    Class {label}: {diff:.6f} difference\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Key observation**: Label-wise DDC preserves class proportions by design, while still using density–diversity selection **within each class**. This gives us the best of both worlds: distribution preservation and label balance."
        ]
    })
    
    # ========== DISTRIBUTION COMPARISON ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Distribution Comparison\n",
            "\n",
            "Let's compare how well each subset/coreset preserves the **marginal distributions** of the original training data. We'll use:\n",
            "\n",
            "- **Wasserstein-1 distance**: Measures how much we need to \"move\" probability mass to match distributions\n",
            "- **Kolmogorov-Smirnov statistic**: Measures the maximum difference between cumulative distribution functions\n",
            "\n",
            "For DDC coresets, we'll use the **weights** to compute weighted distributions."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Select a few features for comparison (e.g., first 5 or high-variance features)\n",
            "feature_indices = list(range(min(5, X_train.shape[1])))\n",
            "feature_names = [f\"Feature {i}\" for i in feature_indices]\n",
            "\n",
            "print(f\"Comparing distributions for {len(feature_indices)} features:\")\n",
            "print(feature_names)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def compute_wasserstein_weighted(source_data, target_data, source_weights=None, target_weights=None):\n",
            "    \"\"\"Compute Wasserstein-1 distance between weighted distributions.\"\"\"\n",
            "    if source_weights is None:\n",
            "        source_weights = np.ones(len(source_data)) / len(source_data)\n",
            "    if target_weights is None:\n",
            "        target_weights = np.ones(len(target_data)) / len(target_data)\n",
            "    \n",
            "    # Sort by value for computing Wasserstein distance\n",
            "    source_sorted_idx = np.argsort(source_data)\n",
            "    target_sorted_idx = np.argsort(target_data)\n",
            "    \n",
            "    source_sorted = source_data[source_sorted_idx]\n",
            "    target_sorted = target_data[target_sorted_idx]\n",
            "    \n",
            "    source_weights_sorted = source_weights[source_sorted_idx]\n",
            "    target_weights_sorted = target_weights[target_sorted_idx]\n",
            "    \n",
            "    # Compute cumulative distributions\n",
            "    source_cdf = np.cumsum(source_weights_sorted)\n",
            "    target_cdf = np.cumsum(target_weights_sorted)\n",
            "    \n",
            "    # Interpolate to common grid\n",
            "    all_values = np.unique(np.concatenate([source_sorted, target_sorted]))\n",
            "    all_values = np.sort(all_values)\n",
            "    \n",
            "    source_cdf_interp = np.interp(all_values, source_sorted, source_cdf, left=0, right=1)\n",
            "    target_cdf_interp = np.interp(all_values, target_sorted, target_cdf, left=0, right=1)\n",
            "    \n",
            "    # Wasserstein-1 is the integral of |CDF_diff|\n",
            "    w1 = np.trapz(np.abs(source_cdf_interp - target_cdf_interp), all_values)\n",
            "    \n",
            "    return w1\n",
            "\n",
            "def compute_ks_weighted(source_data, target_data, source_weights=None, target_weights=None, n_samples=10000):\n",
            "    \"\"\"Approximate KS statistic for weighted distributions by sampling.\"\"\"\n",
            "    if source_weights is None:\n",
            "        source_weights = np.ones(len(source_data)) / len(source_data)\n",
            "    if target_weights is None:\n",
            "        target_weights = np.ones(len(target_data)) / len(target_data)\n",
            "    \n",
            "    # Sample from weighted distributions\n",
            "    source_samples = np.random.choice(\n",
            "        source_data, size=n_samples, p=source_weights / source_weights.sum(), replace=True\n",
            "    )\n",
            "    target_samples = np.random.choice(\n",
            "        target_data, size=n_samples, p=target_weights / target_weights.sum(), replace=True\n",
            "    )\n",
            "    \n",
            "    # Compute KS statistic\n",
            "    ks_stat, _ = ks_2samp(source_samples, target_samples)\n",
            "    \n",
            "    return ks_stat"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare distributions for each feature\n",
            "results = []\n",
            "\n",
            "for feat_idx, feat_name in zip(feature_indices, feature_names):\n",
            "    # Full training data (reference)\n",
            "    X_train_feat = X_train[:, feat_idx]\n",
            "    \n",
            "    # Random subset\n",
            "    X_random_feat = X_random[:, feat_idx]\n",
            "    w1_random = compute_wasserstein_weighted(X_train_feat, X_random_feat)\n",
            "    ks_random = compute_ks_weighted(X_train_feat, X_random_feat)\n",
            "    \n",
            "    # Stratified subset\n",
            "    X_strat_feat = X_strat[:, feat_idx]\n",
            "    w1_strat = compute_wasserstein_weighted(X_train_feat, X_strat_feat)\n",
            "    ks_strat = compute_ks_weighted(X_train_feat, X_strat_feat)\n",
            "    \n",
            "    # Global DDC\n",
            "    S_global_feat = S_global[:, feat_idx]\n",
            "    w1_global = compute_wasserstein_weighted(X_train_feat, S_global_feat, target_weights=w_global)\n",
            "    ks_global = compute_ks_weighted(X_train_feat, S_global_feat, target_weights=w_global)\n",
            "    \n",
            "    # Label-wise DDC\n",
            "    S_labelwise_feat = S_labelwise[:, feat_idx]\n",
            "    w1_labelwise = compute_wasserstein_weighted(X_train_feat, S_labelwise_feat, target_weights=w_labelwise)\n",
            "    ks_labelwise = compute_ks_weighted(X_train_feat, S_labelwise_feat, target_weights=w_labelwise)\n",
            "    \n",
            "    results.append({\n",
            "        'feature': feat_name,\n",
            "        'W1_random': w1_random,\n",
            "        'W1_strat': w1_strat,\n",
            "        'W1_global_ddc': w1_global,\n",
            "        'W1_labelwise_ddc': w1_labelwise,\n",
            "        'KS_random': ks_random,\n",
            "        'KS_strat': ks_strat,\n",
            "        'KS_global_ddc': ks_global,\n",
            "        'KS_labelwise_ddc': ks_labelwise,\n",
            "    })\n",
            "\n",
            "# Create results DataFrame\n",
            "dist_results_df = pd.DataFrame(results)\n",
            "print(\"📊 Distribution Preservation Metrics:\")\n",
            "print(\"\\nWasserstein-1 Distance (lower is better):\")\n",
            "print(dist_results_df[['feature', 'W1_random', 'W1_strat', 'W1_global_ddc', 'W1_labelwise_ddc']].to_string(index=False))\n",
            "print(\"\\nKolmogorov-Smirnov Statistic (lower is better):\")\n",
            "print(dist_results_df[['feature', 'KS_random', 'KS_strat', 'KS_global_ddc', 'KS_labelwise_ddc']].to_string(index=False))"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compute average metrics across features\n",
            "avg_metrics = {\n",
            "    'Method': ['Random', 'Stratified', 'Global DDC', 'Label-wise DDC'],\n",
            "    'Avg W1': [\n",
            "        dist_results_df['W1_random'].mean(),\n",
            "        dist_results_df['W1_strat'].mean(),\n",
            "        dist_results_df['W1_global_ddc'].mean(),\n",
            "        dist_results_df['W1_labelwise_ddc'].mean(),\n",
            "    ],\n",
            "    'Avg KS': [\n",
            "        dist_results_df['KS_random'].mean(),\n",
            "        dist_results_df['KS_strat'].mean(),\n",
            "        dist_results_df['KS_global_ddc'].mean(),\n",
            "        dist_results_df['KS_labelwise_ddc'].mean(),\n",
            "    ]\n",
            "}\n",
            "\n",
            "avg_df = pd.DataFrame(avg_metrics)\n",
            "print(\"📊 Average Distribution Preservation (across features):\")\n",
            "print(avg_df.to_string(index=False))"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Key observations**:\n",
            "- DDC coresets (especially label-wise) typically preserve distributions better than random sampling\n",
            "- Label-wise DDC often matches or exceeds global DDC in distribution preservation while also preserving label proportions"
        ]
    })
    
    # ========== DOWNSTREAM MODEL COMPARISON ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Downstream Model Comparison\n",
            "\n",
            "Now let's train logistic regression models on each subset/coreset and evaluate on the **same test set**. This shows the **practical impact** of distribution preservation on model performance."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train models on each subset/coreset\n",
            "models = {}\n",
            "predictions = {}\n",
            "\n",
            "# 1. Random subset\n",
            "lr_random = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=None)\n",
            "lr_random.fit(X_random, y_random)\n",
            "models['Random'] = lr_random\n",
            "predictions['Random'] = lr_random.predict_proba(X_test)[:, 1]\n",
            "\n",
            "# 2. Stratified subset\n",
            "lr_strat = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=None)\n",
            "lr_strat.fit(X_strat, y_strat)\n",
            "models['Stratified'] = lr_strat\n",
            "predictions['Stratified'] = lr_strat.predict_proba(X_test)[:, 1]\n",
            "\n",
            "# 3. Global DDC coreset (use weights)\n",
            "lr_global = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=None)\n",
            "sample_weights_global = w_global * len(X_train)  # Scale weights to approximate sample counts\n",
            "lr_global.fit(S_global, y_global, sample_weight=sample_weights_global)\n",
            "models['Global DDC'] = lr_global\n",
            "predictions['Global DDC'] = lr_global.predict_proba(X_test)[:, 1]\n",
            "\n",
            "# 4. Label-wise DDC coreset (use weights)\n",
            "lr_labelwise = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=None)\n",
            "sample_weights_labelwise = w_labelwise * len(X_train)\n",
            "lr_labelwise.fit(S_labelwise, y_labelwise, sample_weight=sample_weights_labelwise)\n",
            "models['Label-wise DDC'] = lr_labelwise\n",
            "predictions['Label-wise DDC'] = lr_labelwise.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print(\"✅ All models trained\")"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Evaluate all models\n",
            "model_results = []\n",
            "\n",
            "# Full data baseline (already computed)\n",
            "model_results.append(baseline_metrics)\n",
            "\n",
            "# Evaluate other methods\n",
            "for method_name, y_pred_proba in predictions.items():\n",
            "    auc = roc_auc_score(y_test, y_pred_proba)\n",
            "    brier = brier_score_loss(y_test, y_pred_proba)\n",
            "    y_pred = (y_pred_proba >= 0.5).astype(int)\n",
            "    accuracy = accuracy_score(y_test, y_pred)\n",
            "    \n",
            "    model_results.append({\n",
            "        'method': method_name,\n",
            "        'auc': auc,\n",
            "        'brier': brier,\n",
            "        'accuracy': accuracy\n",
            "    })\n",
            "\n",
            "# Create comparison table\n",
            "comparison_df = pd.DataFrame(model_results)\n",
            "comparison_df['auc_diff'] = comparison_df['auc'] - baseline_auc\n",
            "comparison_df['brier_diff'] = comparison_df['brier'] - baseline_brier\n",
            "\n",
            "print(\"📊 Model Performance Comparison:\")\n",
            "print(\"\\n\" + comparison_df.to_string(index=False))\n",
            "\n",
            "print(\"\\n📉 Deviation from Full-Data Baseline:\")\n",
            "for method in ['Random', 'Stratified', 'Global DDC', 'Label-wise DDC']:\n",
            "    row = comparison_df[comparison_df['method'] == method].iloc[0]\n",
            "    print(f\"  {method:15s}: AUC {row['auc_diff']:+.4f}, Brier {row['brier_diff']:+.4f}\")"
        ]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Key observations**:\n",
            "- Label-wise DDC typically performs closest to the full-data baseline\n",
            "- Global DDC may underperform if class proportions are important for the task\n",
            "- DDC coresets generally outperform random sampling"
        ]
    })
    
    # ========== VISUALIZATIONS ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Visualizations\n",
            "\n",
            "Let's visualize the distribution preservation and spatial coverage of each method."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Plot marginal distributions for a couple of features\n",
            "n_features_to_plot = min(2, len(feature_indices))\n",
            "\n",
            "fig, axes = plt.subplots(1, n_features_to_plot, figsize=(6 * n_features_to_plot, 5))\n",
            "if n_features_to_plot == 1:\n",
            "    axes = [axes]\n",
            "\n",
            "for plot_idx, feat_idx in enumerate(feature_indices[:n_features_to_plot]):\n",
            "    ax = axes[plot_idx]\n",
            "    \n",
            "    # Full training data (reference)\n",
            "    ax.hist(X_train[:, feat_idx], bins=50, density=True, alpha=0.3, \n",
            "            label='Full Data', color='gray', edgecolor='black')\n",
            "    \n",
            "    # Random subset\n",
            "    ax.hist(X_random[:, feat_idx], bins=30, density=True, alpha=0.5, \n",
            "            label='Random', color='blue', histtype='step', linewidth=2)\n",
            "    \n",
            "    # Stratified subset\n",
            "    ax.hist(X_strat[:, feat_idx], bins=30, density=True, alpha=0.5, \n",
            "            label='Stratified', color='green', histtype='step', linewidth=2)\n",
            "    \n",
            "    # Global DDC (weighted)\n",
            "    ax.hist(S_global[:, feat_idx], bins=30, weights=w_global, density=True, \n",
            "            label='Global DDC', color='red', histtype='step', linewidth=2)\n",
            "    \n",
            "    # Label-wise DDC (weighted)\n",
            "    ax.hist(S_labelwise[:, feat_idx], bins=30, weights=w_labelwise, density=True, \n",
            "            label='Label-wise DDC', color='orange', histtype='step', linewidth=2)\n",
            "    \n",
            "    ax.set_xlabel(f'Feature {feat_idx}')\n",
            "    ax.set_ylabel('Density')\n",
            "    ax.set_title(f'Marginal Distribution: Feature {feat_idx}')\n",
            "    ax.legend()\n",
            "    ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2D PCA projection to visualize spatial coverage\n",
            "pca = PCA(n_components=2, random_state=RANDOM_STATE)\n",
            "X_train_2d = pca.fit_transform(X_train)\n",
            "\n",
            "# Project subsets\n",
            "X_random_2d = pca.transform(X_random)\n",
            "X_strat_2d = pca.transform(X_strat)\n",
            "S_global_2d = pca.transform(S_global)\n",
            "S_labelwise_2d = pca.transform(S_labelwise)\n",
            "\n",
            "# Plot\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 12))\n",
            "axes = axes.flatten()\n",
            "\n",
            "methods_2d = [\n",
            "    ('Random', X_random_2d, y_random, None, 'blue'),\n",
            "    ('Stratified', X_strat_2d, y_strat, None, 'green'),\n",
            "    ('Global DDC', S_global_2d, y_global, w_global, 'red'),\n",
            "    ('Label-wise DDC', S_labelwise_2d, y_labelwise, w_labelwise, 'orange'),\n",
            "]\n",
            "\n",
            "for ax, (method_name, subset_2d, subset_y, subset_w, color) in zip(axes, methods_2d):\n",
            "    # Background: full data (low alpha)\n",
            "    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], \n",
            "              c=y_train, cmap='RdYlBu', alpha=0.1, s=1, label='Full Data')\n",
            "    \n",
            "    # Overlay: representatives\n",
            "    if subset_w is not None:\n",
            "        # Size proportional to weight\n",
            "        sizes = 200 * (subset_w / subset_w.max())\n",
            "    else:\n",
            "        sizes = 50\n",
            "    \n",
            "    ax.scatter(subset_2d[:, 0], subset_2d[:, 1], \n",
            "              c=subset_y, cmap='RdYlBu', s=sizes, \n",
            "              edgecolors='black', linewidth=0.5, alpha=0.8, label=method_name)\n",
            "    \n",
            "    ax.set_xlabel('PC1')\n",
            "    ax.set_ylabel('PC2')\n",
            "    ax.set_title(f'{method_name} (n={len(subset_2d)})')\n",
            "    ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ROC curves comparison\n",
            "fig, ax = plt.subplots(1, 1, figsize=(8, 6))\n",
            "\n",
            "# Full data baseline\n",
            "fpr_full, tpr_full, _ = roc_curve(y_test, y_pred_proba_full)\n",
            "ax.plot(fpr_full, tpr_full, label=f'Full Data (AUC={baseline_auc:.4f})', \n",
            "        linewidth=2, color='black', linestyle='--')\n",
            "\n",
            "# Other methods\n",
            "for method_name, y_pred_proba in predictions.items():\n",
            "    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)\n",
            "    auc = roc_auc_score(y_test, y_pred_proba)\n",
            "    ax.plot(fpr, tpr, label=f'{method_name} (AUC={auc:.4f})', linewidth=2)\n",
            "\n",
            "ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')\n",
            "ax.set_xlabel('False Positive Rate')\n",
            "ax.set_ylabel('True Positive Rate')\n",
            "ax.set_title('ROC Curves Comparison')\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })
    
    # ========== DISCUSSION ==========
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Discussion and Takeaways\n",
            "\n",
            "### Key Observations\n",
            "\n",
            "1. **DDC is unsupervised**: The global DDC coreset optimizes for feature distribution preservation but **ignores labels**. This can lead to distorted class proportions, which is problematic for supervised learning tasks.\n",
            "\n",
            "2. **Label-wise DDC preserves class balance**: By applying DDC separately within each class, we maintain label proportions while still benefiting from density–diversity selection **within each class**. This approach:\n",
            "   - Preserves class proportions by design\n",
            "   - Maintains distributional fidelity within each class\n",
            "   - Typically performs closer to the full-data baseline than naive random sampling\n",
            "\n",
            "3. **Distribution preservation matters**: Methods that better preserve marginal distributions (measured by Wasserstein-1 and KS statistics) tend to produce models that perform closer to the full-data baseline.\n",
            "\n",
            "4. **Weights are essential**: DDC coresets are **weighted sets**, not just point sets. The weights allow us to approximate the full distribution from a small number of representatives.\n",
            "\n",
            "### When to Use What?\n",
            "\n",
            "**Use Global DDC when:**\n",
            "- You're doing **purely unsupervised** analysis (EDA, clustering, anomaly detection)\n",
            "- Label proportions don't matter\n",
            "- You want to preserve the overall feature distribution\n",
            "\n",
            "**Use Label-wise DDC when:**\n",
            "- You're working on a **supervised learning** problem\n",
            "- Label proportions matter (e.g., imbalanced classification)\n",
            "- You want both distribution preservation AND label balance\n",
            "- You need a small, interpretable subset for model prototyping\n",
            "\n",
            "**Use Random/Stratified sampling when:**\n",
            "- You need a simple baseline for comparison\n",
            "- You don't need distribution preservation\n",
            "- Computational resources are extremely limited\n",
            "\n",
            "### Conclusion\n",
            "\n",
            "DDC coresets provide a principled way to compress large datasets while preserving distributional properties. For supervised learning tasks, **label-wise DDC** is the recommended approach as it combines the benefits of distribution preservation with label balance.\n",
            "\n",
            "---\n",
            "\n",
            "**Resources:**\n",
            "- GitHub: https://github.com/crbazevedo/dd-coresets\n",
            "- PyPI: https://pypi.org/project/dd-coresets/\n",
            "- Documentation: See the main README for API details and more examples"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

if __name__ == "__main__":
    nb = create_notebook()
    with open('binary_classification_ddc.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook created with {len(nb['cells'])} cells")

