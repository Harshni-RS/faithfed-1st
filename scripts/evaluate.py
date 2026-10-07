"""
Evaluation and held-out benchmarking script for FaithFed-XAI.
Computes test-time diagnostic utility (Accuracy, Macro-F1, QWK),
explanation faithfulness (Insertion/Deletion AUC), and the 5 Cross-Hospital Fairness Metrics.
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_model
from src.data_preprocessing import get_dataloader, DR_CLASSES
from src.hospital_partition import load_splits, get_hospital_dfs
from src.gradcam import GradCAM
from src.faithfulness import compute_insertion_deletion_auc
from src.fairness import compute_clinical_metrics, compute_five_fairness_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="FaithFed-XAI Held-Out Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--model_path", type=str, default="artifacts/global_model.pth", help="Path to model weights")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--eval_faithfulness", action="store_true", default=True, help="Compute Insertion/Deletion AUC")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    artifacts_dir = config.get("output", {}).get("artifacts_dir", "artifacts")
    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(config["data"]["csv_path"])
    splits_path = os.path.join(artifacts_dir, "splits.json")
    splits_data = load_splits(splits_path)

    # Load model
    model = get_model(num_classes=config["data"].get("num_classes", 5)).to(device)
    if os.path.exists(args.model_path):
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded model checkpoint from {args.model_path}")
    else:
        print(f"Warning: Checkpoint {args.model_path} not found. Running with initialized weights.")

    model.eval()
    t_layer = getattr(model, "layer4", getattr(model, "block5", None))
    if hasattr(t_layer, "conv"):
        t_layer = t_layer.conv
    gradcam = GradCAM(model, target_layer=t_layer)

    hospital_ids = list(splits_data["hospitals"].keys())
    hospital_results = {}
    table_rows = []

    print("\n" + "=" * 75)
    print("Held-Out Test Set Evaluation Across 5 Non-IID Hospitals")
    print("=" * 75)

    for hid in hospital_ids:
        h_info = splits_data["hospitals"][hid]
        _, test_df = get_hospital_dfs(df, splits_data, hid)

        test_loader = get_dataloader(
            test_df,
            config["data"]["img_dir"],
            batch_size=config["federated"].get("batch_size", 16),
            img_size=config["data"].get("img_size", 128),
            is_training=False
        )

        all_preds = []
        all_targets = []
        faithfulness_scores = []

        with torch.no_grad():
            for bx, by, _ in test_loader:
                bx = bx.to(device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(by.tolist())

        metrics = compute_clinical_metrics(all_targets, all_preds, num_classes=5)

        # Compute faithfulness on a subset of test samples
        if args.eval_faithfulness:
            sample_count = min(10, len(test_df))
            eval_sample_loader = get_dataloader(
                test_df.iloc[:sample_count],
                config["data"]["img_dir"],
                batch_size=1,
                img_size=config["data"].get("img_size", 128),
                is_training=False
            )
            for s_img, s_lbl, _ in eval_sample_loader:
                hmap, _, _ = gradcam.generate_heatmap(s_img.to(device), target_class=int(s_lbl.item()))
                f_res = compute_insertion_deletion_auc(
                    model=model,
                    input_tensor=s_img,
                    heatmap=hmap,
                    target_class=int(s_lbl.item()),
                    num_steps=5,
                    device=device
                )
                faithfulness_scores.append(f_res["faithfulness_score"])

        avg_faith = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.5

        hospital_results[hid] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "qwk": metrics["qwk"],
            "faithfulness": round(avg_faith, 4),
            "heterogeneity_H": h_info["heterogeneity_H"],
            "sample_count": len(test_df),
            "epsilon": 4.0
        }

        table_rows.append({
            "Hospital": h_info["hospital_name"],
            "Test Samples": len(test_df),
            "Heterogeneity (H)": h_info["heterogeneity_H"],
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Macro-F1": f"{metrics['macro_f1']:.4f}",
            "QWK": f"{metrics['qwk']:.4f}",
            "Faithfulness": f"{avg_faith:.4f}"
        })

    summary_df = pd.DataFrame(table_rows)
    print(summary_df.to_string(index=False))

    # Compute 5 Cross-Hospital Fairness Metrics
    fairness = compute_five_fairness_metrics(hospital_results)

    print("\n" + "=" * 75)
    print("5 Cross-Hospital Client Fairness Metrics")
    print("=" * 75)
    print(f"1. Performance Disparity (Accuracy Spread): {fairness['performance_disparity']:.4f}")
    print(f"2. Explanation Disparity (Faithfulness Spread): {fairness['explanation_disparity']:.4f}")
    print(f"3. Worst-Client Faithfulness: {fairness['worst_client_faithfulness']:.4f}")
    print(f"4. Small-Client Penalty (Large vs Small): {fairness['small_client_penalty']:.4f}")
    print(f"5. Heterogeneity Penalty (Corr with H): {fairness['heterogeneity_penalty']:.4f}")
    print("=" * 75)

    # Save to results/evaluation_report.json
    report = {
        "hospital_results": hospital_results,
        "fairness_metrics": fairness,
        "overall_mean_accuracy": float(np.mean([r["accuracy"] for r in hospital_results.values()])),
        "overall_mean_macro_f1": float(np.mean([r["macro_f1"] for r in hospital_results.values()])),
        "overall_mean_qwk": float(np.mean([r["qwk"] for r in hospital_results.values()])),
        "overall_mean_faithfulness": float(np.mean([r["faithfulness"] for r in hospital_results.values()]))
    }

    out_file = os.path.join(results_dir, "evaluation_report.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved held-out evaluation report to: {out_file}")


if __name__ == "__main__":
    main()
