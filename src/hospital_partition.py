"""
Hospital partitioning and heterogeneity calculation.
Simulates K non-IID hospitals using Dirichlet(alpha) distribution per class,
computes Jensen-Shannon divergence heterogeneity scores (H),
and partitions each hospital into training and held-out test sets.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Any, Optional


def compute_label_distribution(labels: List[int], num_classes: int = 5) -> np.ndarray:
    """Computes normalized empirical probability distribution over classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = np.sum(counts)
    if total == 0:
        return np.ones(num_classes) / num_classes
    return counts / total


def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes Jensen-Shannon divergence between two distributions.
    scipy.spatial.distance.jensenshannon returns the JS distance (sqrt of JSD with base 2).
    Squaring it yields the divergence JSD in [0, 1].
    """
    # Smooth zero probabilities slightly for numerical stability
    eps = 1e-12
    p_smooth = (p + eps) / np.sum(p + eps)
    q_smooth = (q + eps) / np.sum(q + eps)
    js_dist = jensenshannon(p_smooth, q_smooth, base=2)
    return float(js_dist ** 2)


def partition_dataset_dirichlet(
    df: pd.DataFrame,
    num_hospitals: int = 5,
    alpha: float = 0.5,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Partitions the dataset across K hospitals using Dirichlet(alpha) per class.
    Each hospital receives a non-IID subset, further split into train and held-out test sets.
    Computes the heterogeneity score H_k for each hospital using Jensen-Shannon Divergence.
    """
    np.random.seed(seed)
    num_classes = len(df["diagnosis"].unique())
    all_labels = df["diagnosis"].values
    global_dist = compute_label_distribution(all_labels, num_classes=num_classes)

    client_indices = [[] for _ in range(num_hospitals)]

    for c in range(num_classes):
        c_indices = np.where(all_labels == c)[0]
        np.random.shuffle(c_indices)
        
        # Sample proportions from Dirichlet(alpha)
        proportions = np.random.dirichlet(np.repeat(alpha, num_hospitals))
        
        # Convert proportions to counts
        counts = (proportions * len(c_indices)).astype(int)
        # Fix rounding disparity
        diff = len(c_indices) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_hospitals] += 1 if diff > 0 else -1
            
        # Distribute indices
        curr = 0
        for k in range(num_hospitals):
            alloc = c_indices[curr: curr + counts[k]]
            client_indices[k].extend(alloc)
            curr += counts[k]

    splits_data = {
        "metadata": {
            "num_hospitals": num_hospitals,
            "alpha": alpha,
            "train_ratio": train_ratio,
            "seed": seed,
            "global_distribution": global_dist.tolist(),
            "total_samples": len(df)
        },
        "hospitals": {}
    }

    summary_rows = []

    for k in range(num_hospitals):
        h_id = f"hospital_{k+1}"
        indices = np.array(client_indices[k])
        np.random.shuffle(indices)

        # Held-out train / test split
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        train_idx = indices[:n_train].tolist()
        test_idx = indices[n_train:].tolist()

        train_labels = all_labels[train_idx].tolist()
        test_labels = all_labels[test_idx].tolist()
        hospital_all_labels = all_labels[indices].tolist()

        local_dist = compute_label_distribution(hospital_all_labels, num_classes=num_classes)
        h_score = compute_jensen_shannon_divergence(local_dist, global_dist)

        splits_data["hospitals"][h_id] = {
            "hospital_id": h_id,
            "hospital_name": f"Hospital {k+1}",
            "train_indices": train_idx,
            "test_indices": test_idx,
            "train_count": len(train_idx),
            "test_count": len(test_idx),
            "total_count": n_total,
            "local_distribution": local_dist.tolist(),
            "heterogeneity_H": round(h_score, 4),
            "class_counts": {
                c: int((all_labels[indices] == c).sum()) for c in range(num_classes)
            }
        }

        summary_rows.append({
            "Hospital": f"Hospital {k+1}",
            "Total Samples": n_total,
            "Train Samples": len(train_idx),
            "Test Samples": len(test_idx),
            "Heterogeneity (H)": round(h_score, 4),
            "No DR (0)": int((all_labels[indices] == 0).sum()),
            "Mild (1)": int((all_labels[indices] == 1).sum()),
            "Moderate (2)": int((all_labels[indices] == 2).sum()),
            "Severe (3)": int((all_labels[indices] == 3).sum()),
            "Proliferative (4)": int((all_labels[indices] == 4).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    return splits_data, summary_df


def save_splits(splits_data: Dict[str, Any], filepath: str = "artifacts/splits.json"):
    """Saves hospital split specifications to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(splits_data, f, indent=2)


def load_splits(filepath: str = "artifacts/splits.json") -> Dict[str, Any]:
    """Loads hospital split specifications from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_hospital_dfs(
    df: pd.DataFrame,
    splits_data: Dict[str, Any],
    hospital_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieves train and test dataframes for a specific hospital."""
    h_info = splits_data["hospitals"][hospital_id]
    train_df = df.iloc[h_info["train_indices"]].copy().reset_index(drop=True)
    test_df = df.iloc[h_info["test_indices"]].copy().reset_index(drop=True)
    return train_df, test_df
