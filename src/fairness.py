"""
Cross-hospital fairness evaluation and clinical metrics.
Implements the 5 Cross-Hospital Client Fairness Metrics:
1. Performance Disparity (Accuracy / AUC spread)
2. Explanation Disparity (Faithfulness spread)
3. Worst-Client Faithfulness (min faithfulness across all hospitals)
4. Small-Client Penalty (faithfulness gap: large vs small hospitals)
5. Heterogeneity Penalty (correlation between H and faithfulness)
Also computes Macro-F1 and Quadratic Weighted Kappa (QWK) for ordinal DR grading.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from typing import Dict, List, Tuple, Any, Optional


def compute_clinical_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int = 5
) -> Dict[str, float]:
    """
    Computes diagnostic metrics for Diabetic Retinopathy:
    - Accuracy
    - Macro-averaged F1 (gives equal weight to each DR grade)
    - Quadratic Weighted Kappa (QWK, penalizes distant ordinal grading errors)
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    
    # Quadratic Weighted Kappa
    try:
        qwk = float(cohen_kappa_score(y_t, y_p, weights="quadratic", labels=list(range(num_classes))))
        if np.isnan(qwk):
            qwk = 0.0
    except Exception:
        qwk = 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "qwk": round(qwk, 4)
    }


def compute_five_fairness_metrics(
    hospital_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates the 5 Client-Level Fairness Criteria defined in the FaithFed-XAI specification:
    
    1. Performance Disparity: max(Acc_k) - min(Acc_k)
    2. Explanation Disparity: max(Faith_k) - min(Faith_k)
    3. Worst-Client Faithfulness: min(Faith_k)
    4. Small-Client Penalty: Faith_large - Faith_small
    5. Heterogeneity Penalty: Pearson correlation r(H_k, Faith_k)
    """
    h_ids = list(hospital_data.keys())
    accuracies = [hospital_data[h]["accuracy"] for h in h_ids]
    faithfulness_scores = [hospital_data[h]["faithfulness"] for h in h_ids]
    heterogeneity_scores = [hospital_data[h]["heterogeneity_H"] for h in h_ids]
    sample_counts = [hospital_data[h]["sample_count"] for h in h_ids]

    # 1. Performance Disparity (spread across hospitals)
    perf_disp = float(np.max(accuracies) - np.min(accuracies))

    # 2. Explanation Disparity (spread of explanation faithfulness)
    expl_disp = float(np.max(faithfulness_scores) - np.min(faithfulness_scores))

    # 3. Worst-Client Faithfulness
    worst_faith = float(np.min(faithfulness_scores))

    # 4. Small-Client Penalty
    # Partition into smaller-data vs larger-data hospitals (e.g. median split or bottom 40%)
    sorted_by_size = sorted(zip(sample_counts, faithfulness_scores), key=lambda x: x[0])
    k_split = max(1, len(sorted_by_size) // 2)
    small_clients_faith = [f for _, f in sorted_by_size[:k_split]]
    large_clients_faith = [f for _, f in sorted_by_size[k_split:]]
    small_client_penalty = float(np.mean(large_clients_faith) - np.mean(small_clients_faith))

    # 5. Heterogeneity Penalty: correlation between H and Faithfulness
    # If high H clients suffer lower faithfulness, correlation is negative (penalty is positive)
    if len(set(heterogeneity_scores)) > 1 and len(set(faithfulness_scores)) > 1:
        corr, _ = pearsonr(heterogeneity_scores, faithfulness_scores)
        het_penalty = float(-corr) if not np.isnan(corr) else 0.0
    else:
        het_penalty = 0.0

    return {
        "performance_disparity": round(perf_disp, 4),
        "explanation_disparity": round(expl_disp, 4),
        "worst_client_faithfulness": round(worst_faith, 4),
        "small_client_penalty": round(small_client_penalty, 4),
        "heterogeneity_penalty": round(het_penalty, 4),
        "per_hospital_summary": {
            h: {
                "accuracy": round(hospital_data[h]["accuracy"], 4),
                "faithfulness": round(hospital_data[h]["faithfulness"], 4),
                "H": round(hospital_data[h]["heterogeneity_H"], 4),
                "samples": hospital_data[h]["sample_count"],
                "epsilon": round(hospital_data[h].get("epsilon", 4.0), 3)
            }
            for h in h_ids
        }
    }
