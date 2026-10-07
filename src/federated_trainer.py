"""
Federated Learning Orchestration Engine for FaithFed-XAI and Comparative Baselines.
Coordinates:
- Non-IID Hospital Data Loaders
- FedBN Personalization Layer
- DP-SGD with Explanation-Aware Adaptive Privacy Controller (H + A + D)
- Pairwise Additive Masking Secure Aggregation
- Cross-Hospital Faithfulness and Client Fairness Auditing
"""

import os
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple

from src.model import SmallCNN, get_model
from src.data_preprocessing import get_dataloader
from src.hospital_partition import load_splits, get_hospital_dfs
from src.fedbn import (
    extract_shared_parameters,
    extract_local_parameters,
    apply_shared_parameters,
    apply_local_parameters,
    federated_average_shared
)
from src.differential_privacy import (
    AdaptivePrivacyController,
    apply_dp_sgd_step,
    compute_noise_multiplier
)
from src.secure_aggregation import (
    create_pairwise_masks_for_clients,
    mask_client_update,
    aggregate_masked_updates,
    verify_mask_cancellation
)
from src.gradcam import GradCAM
from src.faithfulness import (
    compute_ssim_explanation_drift,
    compute_insertion_deletion_auc
)
from src.fairness import (
    compute_clinical_metrics,
    compute_five_fairness_metrics
)


class FederatedTrainer:
    """
    Orchestrates the entire FaithFed-XAI lifecycle and benchmark baselines.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Setup paths
        self.artifacts_dir = config.get("output", {}).get("artifacts_dir", "artifacts")
        self.results_dir = config.get("output", {}).get("results_dir", "results")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_dir, "hospitals"), exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Load dataset and splits
        self.csv_path = config["data"]["csv_path"]
        self.img_dir = config["data"]["img_dir"]
        self.img_size = config["data"].get("img_size", 128)
        self.num_classes = config["data"].get("num_classes", 5)
        self.num_hospitals = config["data"].get("num_hospitals", 5)

        self.df = pd.read_csv(self.csv_path)
        splits_path = os.path.join(self.artifacts_dir, "splits.json")
        if not os.path.exists(splits_path):
            from src.hospital_partition import partition_dataset_dirichlet, save_splits
            splits_data, _ = partition_dataset_dirichlet(
                self.df,
                num_hospitals=self.num_hospitals,
                alpha=config["data"].get("dirichlet_alpha", 0.5),
                seed=config["data"].get("seed", 42)
            )
            save_splits(splits_data, splits_path)
            self.splits_data = splits_data
        else:
            self.splits_data = load_splits(splits_path)

        self.hospital_ids = [f"hospital_{i+1}" for i in range(self.num_hospitals)]
        self.heterogeneity_scores = {
            hid: self.splits_data["hospitals"][hid]["heterogeneity_H"]
            for hid in self.hospital_ids
        }

        # Training parameters
        self.algorithm = config.get("federated", {}).get("algorithm", "FaithFed-XAI")
        self.num_rounds = config.get("federated", {}).get("num_rounds", 5)
        self.local_epochs = config.get("federated", {}).get("local_epochs", 1)
        self.batch_size = config.get("federated", {}).get("batch_size", 16)
        self.lr = config.get("federated", {}).get("lr", 0.001)

        # Differential privacy settings
        dp_cfg = config.get("differential_privacy", {})
        self.dp_enabled = dp_cfg.get("enabled", True)
        self.target_epsilon = dp_cfg.get("target_epsilon", 4.0)
        self.target_delta = dp_cfg.get("target_delta", 1e-5)
        self.max_grad_norm = dp_cfg.get("max_grad_norm", 1.0)
        # "opacus" = per-sample clipping + accountant (recommended); "manual" = original hand-written step
        self.dp_engine = dp_cfg.get("engine", "opacus")
        self.dp_reports: Dict[str, Dict[str, float]] = {}
        self.adaptive_dp = dp_cfg.get("adaptive", True) and (self.algorithm == "FaithFed-XAI")

        weights = dp_cfg.get("weights", {"w_h": 0.35, "w_a": 0.35, "w_e": 0.30})
        self.privacy_controller = AdaptivePrivacyController(
            num_hospitals=self.num_hospitals,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
            w_h=weights.get("w_h", 0.35),
            w_a=weights.get("w_a", 0.35),
            w_e=weights.get("w_e", 0.30)
        )

        # Initialize Global Model
        self.global_model = get_model(num_classes=self.num_classes).to(self.device)

        # Initialize Hospital Local Models
        self.local_models: Dict[str, SmallCNN] = {
            hid: get_model(num_classes=self.num_classes).to(self.device)
            for hid in self.hospital_ids
        }

        # Synchronize initial weights
        init_state = self.global_model.state_dict()
        for lm in self.local_models.values():
            lm.load_state_dict(init_state)

        # Reference samples for explanation drift tracking across rounds
        self.reference_samples = self._prepare_reference_samples()
        self.baseline_heatmaps: Dict[str, List[np.ndarray]] = {}

        # Round logs
        self.round_history: List[Dict[str, Any]] = []

    def _prepare_reference_samples(self, samples_per_class: int = 1) -> List[Tuple[torch.Tensor, int, str]]:
        """Prepares a fixed set of reference fundus images to track SSIM explanation drift."""
        ref_samples = []
        for c in range(self.num_classes):
            c_df = self.df[self.df["diagnosis"] == c]
            if len(c_df) > 0:
                loader = get_dataloader(
                    c_df.iloc[:samples_per_class],
                    self.img_dir,
                    batch_size=1,
                    img_size=self.img_size,
                    is_training=False
                )
                for img, lbl, img_id in loader:
                    ref_samples.append((img.to(self.device), int(lbl.item()), img_id[0]))
                    break
        return ref_samples

    def train_client_local(
        self,
        hospital_id: str,
        dataloader,
        epochs: int,
        noise_multiplier: float,
        epsilon: float = 0.0
    ) -> nn.Module:
        """Runs local client training for specified epochs with optional DP-SGD."""
        model = self.local_models[hospital_id]
        model.train()

        # If using RetinalNet with pretrained weights, freeze lower blocks
        if hasattr(model, "layer4"):
            for p in model.parameters():
                p.requires_grad = False
            for p in model.layer4.parameters():
                p.requires_grad = True
            for p in model.fc1.parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True

        # Class weights to counteract severe DR class imbalance
        class_weights = torch.tensor([1.0, 1.3, 1.0, 1.6, 1.6], dtype=torch.float32, device=self.device)
        # --- Opacus path: per-sample clipping + accountant-calibrated noise ---
        if self.dp_enabled and noise_multiplier > 0.0 and self.dp_engine == "opacus":
            from src.opacus_dp import train_client_with_opacus
            self.dp_reports[hospital_id] = train_client_with_opacus(
                model=model,
                dataloader=dataloader,
                epochs=epochs,
                lr=self.lr,
                target_epsilon=epsilon if epsilon > 0 else self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
                class_weights=class_weights,
                device=self.device,
            )
            return model

        # --- Manual path (original hand-written DP-SGD step) ---
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=1e-4)

        for ep in range(epochs):
            for batch_x, batch_y, _ in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                if self.dp_enabled and noise_multiplier > 0.0:
                    apply_dp_sgd_step(
                        model=model,
                        optimizer=optimizer,
                        loss=loss,
                        max_grad_norm=self.max_grad_norm,
                        noise_multiplier=noise_multiplier,
                        batch_size=batch_x.size(0),
                        device=self.device
                    )
                else:
                    loss.backward()
                    optimizer.step()

        return model

    def evaluate_client(
        self,
        hospital_id: str,
        test_loader
    ) -> Tuple[Dict[str, float], float, List[np.ndarray]]:
        """
        Evaluates local model on held-out test split.
        Returns clinical metrics, faithfulness score, and reference heatmaps.
        """
        model = self.local_models[hospital_id]
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for bx, by, _ in test_loader:
                bx = bx.to(self.device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(by.tolist())

        metrics = compute_clinical_metrics(all_targets, all_preds, num_classes=self.num_classes)

        # Compute Grad-CAM heatmaps on reference images for faithfulness & drift
        t_layer = getattr(model, "layer4", getattr(model, "block5", None))
        if hasattr(t_layer, "conv"):
            t_layer = t_layer.conv
        gradcam = GradCAM(model, target_layer=t_layer)
        ref_heatmaps = []
        faithfulness_list = []

        for ref_img, ref_lbl, _ in self.reference_samples:
            hmap, pred_c, _ = gradcam.generate_heatmap(ref_img, target_class=ref_lbl)
            ref_heatmaps.append(hmap)
            
            # Compute insertion/deletion faithfulness on reference images
            f_res = compute_insertion_deletion_auc(
                model=model,
                input_tensor=ref_img,
                heatmap=hmap,
                target_class=ref_lbl,
                num_steps=5,
                device=self.device
            )
            faithfulness_list.append(f_res["faithfulness_score"])

        avg_faithfulness = float(np.mean(faithfulness_list)) if faithfulness_list else 0.5
        metrics["faithfulness"] = round(avg_faithfulness, 4)

        return metrics, avg_faithfulness, ref_heatmaps

    def run_federated_round(
        self,
        round_idx: int,
        privacy_allocations: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Executes one full federated training and evaluation round.
        """
        is_fedbn = ("FedBN" in self.algorithm) or (self.algorithm == "FaithFed-XAI")
        client_sample_counts = []
        client_shared_updates = []
        round_eval_results = {}
        round_drifts = {}

        # 1. Distribute Global Parameters to Clients
        global_shared = extract_shared_parameters(self.global_model)
        for hid in self.hospital_ids:
            local_model = self.local_models[hid]
            if is_fedbn:
                apply_shared_parameters(local_model, global_shared)
            else:
                local_model.load_state_dict(self.global_model.state_dict())

        # 2. Local Training & Evaluation for each Hospital
        for hid in self.hospital_ids:
            train_df, test_df = get_hospital_dfs(self.df, self.splits_data, hid)
            max_s = self.config.get("data", {}).get("max_samples_per_client")
            if max_s:
                train_df = train_df.iloc[:max_s].copy()
                test_df = test_df.iloc[:max(5, max_s // 4)].copy()

            client_sample_counts.append(len(train_df))

            train_loader = get_dataloader(
                train_df, self.img_dir, batch_size=self.batch_size, img_size=self.img_size, is_training=True
            )
            test_loader = get_dataloader(
                test_df, self.img_dir, batch_size=self.batch_size, img_size=self.img_size, is_training=False
            )

            # Determine noise multiplier for this client
            noise_mult = privacy_allocations[hid]["noise_multiplier"] if self.dp_enabled else 0.0

            # Train locally
            self.train_client_local(
                hid, train_loader, epochs=self.local_epochs,
                noise_multiplier=noise_mult,
                epsilon=privacy_allocations[hid]["epsilon"] if self.dp_enabled else 0.0
            )

            # Evaluate on held-out test partition
            eval_metrics, faith_score, ref_heatmaps = self.evaluate_client(hid, test_loader)
            round_eval_results[hid] = {
                "accuracy": eval_metrics["accuracy"],
                "macro_f1": eval_metrics["macro_f1"],
                "qwk": eval_metrics["qwk"],
                "faithfulness": faith_score,
                "heterogeneity_H": self.heterogeneity_scores[hid],
                "sample_count": len(train_df),
                "epsilon": privacy_allocations[hid]["epsilon"],
                "epsilon_spent_opacus": self.dp_reports.get(hid, {}).get("epsilon_spent"),
                "noise_multiplier_opacus": self.dp_reports.get(hid, {}).get("noise_multiplier")
            }

            # 3. Track SSIM Explanation Drift (D) relative to Round 1 baseline
            if round_idx == 1:
                self.baseline_heatmaps[hid] = ref_heatmaps
                drift = 0.0
            else:
                base_maps = self.baseline_heatmaps[hid]
                drifts = [
                    compute_ssim_explanation_drift(b, c)
                    for b, c in zip(base_maps, ref_heatmaps)
                ]
                drift = float(np.mean(drifts))

            round_drifts[hid] = drift

            # 4. Extract shared parameter updates for aggregation
            shared_weights = extract_shared_parameters(self.local_models[hid])
            client_shared_updates.append(shared_weights)

        # 5. Secure Aggregation with Pairwise Additive Masking
        total_samples = sum(client_sample_counts)
        client_weights = [c / total_samples for c in client_sample_counts]

        sec_agg_enabled = self.config.get("secure_aggregation", {}).get("enabled", True)
        if sec_agg_enabled:
            # Generate pairwise masks
            client_masks = create_pairwise_masks_for_clients(
                self.hospital_ids,
                global_shared,
                round_idx=round_idx,
                scale=self.config.get("secure_aggregation", {}).get("mask_scale", 1.0)
            )
            # Verify exact mask cancellation (zero residual)
            is_valid, max_res = verify_mask_cancellation(client_masks)

            # Mask each client update
            masked_updates = [
                mask_client_update(client_shared_updates[i], client_masks[self.hospital_ids[i]], client_weights[i])
                for i in range(len(self.hospital_ids))
            ]

            # Server aggregates masked updates
            aggregated_shared, audit_info = aggregate_masked_updates(masked_updates, global_shared)
        else:
            # Plain unmasked weighted FedAvg
            aggregated_shared = federated_average_shared(client_shared_updates, client_sample_counts)
            audit_info = {"protocol": "Plain FedAvg (Unmasked)"}

        # 6. Update Global Model
        if is_fedbn:
            apply_shared_parameters(self.global_model, aggregated_shared)
        else:
            self.global_model.load_state_dict(aggregated_shared)

        # 7. Compute Cross-Hospital Fairness Metrics
        fairness_metrics = compute_five_fairness_metrics(round_eval_results)

        # 8. Adaptive DP Controller: compute next round budgets
        next_allocations = {}
        if self.adaptive_dp:
            next_allocations = self.privacy_controller.compute_next_round_budgets(
                round_idx=round_idx + 1,
                hospital_ids=self.hospital_ids,
                heterogeneity_scores=self.heterogeneity_scores,
                current_accuracies={hid: round_eval_results[hid]["accuracy"] for hid in self.hospital_ids},
                explanation_drifts=round_drifts
            )
        else:
            # Fixed DP allocation
            for hid in self.hospital_ids:
                sigma = compute_noise_multiplier(self.target_epsilon, self.target_delta)
                next_allocations[hid] = {
                    "epsilon": self.target_epsilon,
                    "delta": self.target_delta,
                    "noise_multiplier": round(sigma, 4),
                    "composite_score": 0.0,
                    "H": self.heterogeneity_scores[hid],
                    "A": 0.0,
                    "D": round_drifts[hid]
                }

        # Round Summary Log
        mean_acc = float(np.mean([r["accuracy"] for r in round_eval_results.values()]))
        mean_f1 = float(np.mean([r["macro_f1"] for r in round_eval_results.values()]))
        mean_qwk = float(np.mean([r["qwk"] for r in round_eval_results.values()]))
        mean_faith = float(np.mean([r["faithfulness"] for r in round_eval_results.values()]))

        round_record = {
            "round": round_idx,
            "algorithm": self.algorithm,
            "mean_accuracy": round(mean_acc, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "mean_qwk": round(mean_qwk, 4),
            "mean_faithfulness": round(mean_faith, 4),
            "performance_disparity": fairness_metrics["performance_disparity"],
            "explanation_disparity": fairness_metrics["explanation_disparity"],
            "worst_client_faithfulness": fairness_metrics["worst_client_faithfulness"],
            "small_client_penalty": fairness_metrics["small_client_penalty"],
            "heterogeneity_penalty": fairness_metrics["heterogeneity_penalty"],
            "per_hospital": round_eval_results,
            "drifts": round_drifts,
            "heterogeneity": {h: float(v) for h, v in self.heterogeneity_scores.items()},
            "best_accuracies": {h: float(v) for h, v in self.privacy_controller.best_accuracies.items()},
            "next_privacy_allocations": next_allocations,
            "audit_info": audit_info
        }

        self.round_history.append(round_record)
        return round_record, next_allocations

    def train_federated(self, num_rounds: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Executes complete multi-round federated training run.
        """
        rounds = num_rounds or self.num_rounds
        allocations = self.privacy_controller.initialize_round_1(self.hospital_ids)

        for r in range(1, rounds + 1):
            print(f"\n--- [Round {r}/{rounds}] Algorithm: {self.algorithm} ---")
            record, next_allocations = self.run_federated_round(r, allocations)
            allocations = next_allocations

            print(f"Mean Acc: {record['mean_accuracy']:.4f} | Macro-F1: {record['mean_macro_f1']:.4f} | QWK: {record['mean_qwk']:.4f}")
            print(f"Mean Faithfulness: {record['mean_faithfulness']:.4f} | Perf Disparity: {record['performance_disparity']:.4f} | Expl Disparity: {record['explanation_disparity']:.4f}")

        # Save artifacts
        self.save_artifacts()
        return self.round_history

    def save_artifacts(self):
        """Saves models, round logs, and summary configurations."""
        # 1. Global model
        global_path = os.path.join(self.artifacts_dir, "global_model.pth")
        torch.save(self.global_model.state_dict(), global_path)
        print(f"Saved global model checkpoint to {global_path}")

        # 2. Local hospital personalized models
        for hid, lm in self.local_models.items():
            h_path = os.path.join(self.artifacts_dir, "hospitals", f"{hid}.pth")
            torch.save(lm.state_dict(), h_path)

        # 3. Round logs CSV
        csv_rows = []
        for rec in self.round_history:
            row = {
                "round": rec["round"],
                "algorithm": rec["algorithm"],
                "mean_accuracy": rec["mean_accuracy"],
                "mean_macro_f1": rec["mean_macro_f1"],
                "mean_qwk": rec["mean_qwk"],
                "mean_faithfulness": rec["mean_faithfulness"],
                "performance_disparity": rec["performance_disparity"],
                "explanation_disparity": rec["explanation_disparity"],
                "worst_client_faithfulness": rec["worst_client_faithfulness"],
                "small_client_penalty": rec["small_client_penalty"],
                "heterogeneity_penalty": rec["heterogeneity_penalty"]
            }
            # Per-hospital columns
            for hid, h_data in rec["per_hospital"].items():
                row[f"{hid}_acc"] = h_data["accuracy"]
                row[f"{hid}_faith"] = h_data["faithfulness"]
                row[f"{hid}_eps"] = h_data["epsilon"]
            csv_rows.append(row)

        round_log_path = os.path.join(self.artifacts_dir, "round_log.csv")
        pd.DataFrame(csv_rows).to_csv(round_log_path, index=False)
        print(f"Saved federated training log to {round_log_path}")

        # 3b. Differential-privacy telemetry for the dashboard (real values, not typed ones)
        dp_rounds = []
        for rec in self.round_history:
            hospitals = {}
            for hid, h in rec["per_hospital"].items():
                nxt = rec.get("next_privacy_allocations", {}).get(hid, {})
                spent = h.get("epsilon_spent_opacus")
                sigma = h.get("noise_multiplier_opacus")
                hospitals[hid] = {
                    "current_accuracy": float(h["accuracy"]),
                    "best_accuracy": float(rec.get("best_accuracies", {}).get(hid, h["accuracy"])),
                    "H_raw": float(rec.get("heterogeneity", {}).get(hid, 0.0)),
                    "drift_D": float(rec["drifts"].get(hid, 0.0)),
                    "A": nxt.get("A"),
                    "composite_S": nxt.get("composite_score"),
                    "allocated_epsilon": float(h["epsilon"]),
                    "epsilon_spent": float(spent) if spent is not None else None,
                    "noise_multiplier": float(sigma) if sigma is not None else None,
                    "next_epsilon": nxt.get("epsilon"),
                    "next_noise_multiplier_formula": nxt.get("noise_multiplier"),
                }
            dp_rounds.append({"round": rec["round"], "hospitals": hospitals})

        dp_report = {
            "engine": getattr(self, "dp_engine", "manual") if self.dp_enabled else "off",
            "algorithm": self.algorithm,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "max_grad_norm": self.max_grad_norm,
            "weights": {
                "w_h": getattr(self.privacy_controller, "w_h", None),
                "w_a": getattr(self.privacy_controller, "w_a", None),
                "w_e": getattr(self.privacy_controller, "w_e", None),
            },
            "rounds": dp_rounds,
        }
        dp_path = os.path.join(self.artifacts_dir, "dp_report.json")
        with open(dp_path, "w", encoding="utf-8") as f:
            json.dump(dp_report, f, indent=2, default=float)
        print(f"Saved differential-privacy report to {dp_path}")

        # 4. JSON Config & summary
        summary_path = os.path.join(self.artifacts_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "config": self.config,
                "total_rounds": len(self.round_history),
                "final_round_metrics": self.round_history[-1] if self.round_history else {}
            }, f, indent=2)