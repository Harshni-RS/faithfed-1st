"""
FaithFed-XAI: An Explanation-Aware, Privacy-Adaptive Federated Learning Framework
with Client Fairness Evaluation for Medical Imaging Diagnosis
"""

from src.model import SmallCNN, get_model
from src.data_preprocessing import ben_graham_normalize, get_dataloader, APTOSDataset, DR_CLASSES, DR_DESCRIPTIONS
from src.hospital_partition import partition_dataset_dirichlet, load_splits, save_splits, get_hospital_dfs
from src.fedbn import extract_shared_parameters, extract_local_parameters, apply_shared_parameters, apply_local_parameters
from src.differential_privacy import AdaptivePrivacyController, apply_dp_sgd_step
from src.secure_aggregation import create_pairwise_masks_for_clients, mask_client_update, aggregate_masked_updates, verify_mask_cancellation
from src.gradcam import GradCAM, overlay_heatmap
from src.faithfulness import compute_insertion_deletion_auc, compute_ssim_explanation_drift
from src.fairness import compute_clinical_metrics, compute_five_fairness_metrics
from src.federated_trainer import FederatedTrainer
from src.inference import RetinalDiagnosticEngine

__all__ = [
    "SmallCNN",
    "get_model",
    "ben_graham_normalize",
    "get_dataloader",
    "APTOSDataset",
    "DR_CLASSES",
    "DR_DESCRIPTIONS",
    "partition_dataset_dirichlet",
    "load_splits",
    "save_splits",
    "get_hospital_dfs",
    "extract_shared_parameters",
    "extract_local_parameters",
    "apply_shared_parameters",
    "apply_local_parameters",
    "AdaptivePrivacyController",
    "apply_dp_sgd_step",
    "create_pairwise_masks_for_clients",
    "mask_client_update",
    "aggregate_masked_updates",
    "verify_mask_cancellation",
    "GradCAM",
    "overlay_heatmap",
    "compute_insertion_deletion_auc",
    "compute_ssim_explanation_drift",
    "compute_clinical_metrics",
    "compute_five_fairness_metrics",
    "FederatedTrainer",
    "RetinalDiagnosticEngine"
]
