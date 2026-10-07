"""
Faithfulness evaluation and explanation drift monitoring.
Implements Insertion AUC, Deletion AUC, and SSIM-based Explanation Drift (D)
to quantify the clinical reliability and truthfulness of Grad-CAM heatmaps.

Insertion / Deletion (corrected version)
----------------------------------------
Pixels are ranked by the Grad-CAM heatmap (most salient first).
  * Deletion : starting from the real image, replace the top-k pixels with a
               "neutral" baseline; the evidence for the class should FALL fast.
  * Insertion: starting from the baseline, copy back the top-k real pixels;
               the evidence should RISE fast.

Three fixes compared with the first version:
  1. Strong baseline. The old code used an 11x11 average-pool "blur", which is
     almost the original image, so nothing was really removed. The baseline is
     now a heavy Gaussian blur (sigma = image_size / 8 by default).
  2. Unsaturated evidence. Softmax probability sits at ~1.0 for a confident
     model, so both curves go flat. The default metric is now the target-class
     log-odds (logit margin = z_target - logsumexp(other logits)), which keeps
     changing after the probability saturates.
  3. Normalised curves. Evidence is rescaled so 0 = baseline image and
     1 = original image. Then insertion runs 0 -> 1, deletion runs 1 -> 0, and a
     random pixel order scores about 0.5. A score clearly above 0.5 means the
     heatmap is more faithful than chance. An optional random-order control
     measures this directly.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List, Tuple, Optional, Any
from src.gradcam import GradCAM


# ----------------------------------------------------------------------------
# Small numpy helpers (kept torch-free so they are easy to test)
# ----------------------------------------------------------------------------
def _auc(y, x) -> float:
    """Trapezoidal area under y(x) (NumPy 1.x / 2.x compatible)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(y, x))


def _normalize_evidence(raw, m_base: float, m_full: float, min_gap: float):
    """
    Rescales evidence so baseline image -> 0 and original image -> 1, clipped to [0, 1].
    Returns (normalised_array, degenerate_flag). If the model is not more
    confident on the real image than on the baseline (gap <= min_gap) the curves
    cannot be interpreted, so a neutral 0.5 curve is returned with degenerate=True.
    """
    raw = np.asarray(raw, dtype=np.float64)
    gap = m_full - m_base
    if gap <= min_gap:
        return np.full_like(raw, 0.5), True
    return np.clip((raw - m_base) / gap, 0.0, 1.0), False


def _faithfulness_from_curves(ins_curve, del_curve, fractions) -> Tuple[float, float, float]:
    ins_auc = _auc(ins_curve, fractions)
    del_auc = _auc(del_curve, fractions)
    score = float(np.clip((ins_auc + (1.0 - del_auc)) / 2.0, 0.0, 1.0))
    return ins_auc, del_auc, score


# ----------------------------------------------------------------------------
# Torch helpers
# ----------------------------------------------------------------------------
def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for a [1, C, H, W] tensor (reflect padding)."""
    _, c, h, w = x.shape
    radius = int(min(np.ceil(3.0 * sigma), min(h, w) - 1))
    k = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-(k ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = g.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kv, groups=c)
    return x


def _make_baseline(x: torch.Tensor, substrate: str, sigma_frac: float) -> torch.Tensor:
    h, w = x.shape[2], x.shape[3]
    if substrate == "blur":
        return _gaussian_blur(x, sigma=max(1.0, sigma_frac * max(h, w)))
    if substrate == "mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x).clone()
    if substrate == "zero":
        return torch.zeros_like(x)
    if substrate == "legacy":  # the original 11x11 average-pool substrate (for comparison only)
        return F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    raise ValueError(f"Unknown substrate '{substrate}' (use blur | mean | zero | legacy)")


@torch.no_grad()
def _logits_for_images(model: torch.nn.Module, imgs: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    outs = []
    for i in range(0, imgs.shape[0], chunk):
        outs.append(model(imgs[i:i + chunk]).float().cpu())
    return torch.cat(outs, dim=0)


def _evidence(logits: torch.Tensor, target: int, metric: str) -> np.ndarray:
    """Per-image evidence for the target class."""
    if metric == "prob":
        return F.softmax(logits, dim=1)[:, target].numpy().astype(np.float64)
    if metric == "logit":
        z_t = logits[:, target]
        others = torch.cat([logits[:, :target], logits[:, target + 1:]], dim=1)
        return (z_t - torch.logsumexp(others, dim=1)).numpy().astype(np.float64)
    raise ValueError(f"Unknown metric '{metric}' (use logit | prob)")


def _build_images(x, baseline, order, fractions, h, w):
    """Returns (deletion images, insertion images), one per fraction, as stacked tensors."""
    total = h * w
    rank = np.empty(total, dtype=np.int64)
    rank[order] = np.arange(total)  # rank 0 = most salient pixel
    dels, inss = [], []
    for frac in fractions:
        k = int(round(frac * total))
        mask = torch.from_numpy((rank < k).reshape(1, 1, h, w)).to(x.device)
        dels.append(torch.where(mask, baseline, x))   # remove the top-k pixels
        inss.append(torch.where(mask, x, baseline))   # reveal only the top-k pixels
    return torch.cat(dels, dim=0), torch.cat(inss, dim=0)


def compute_insertion_deletion_auc(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: Optional[int] = None,
    num_steps: int = 10,
    device: Optional[torch.device] = None,
    substrate: str = "blur",
    metric: str = "logit",
    normalize: bool = True,
    blur_sigma_frac: float = 0.125,
    n_random: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Computes Insertion and Deletion curves and AUC for an explanation heatmap.

    Args:
        model, input_tensor ([1,C,H,W] normalised), heatmap ([H,W] in [0,1]),
        target_class (None -> predicted class), num_steps, device: as before.
        substrate: "blur" (default) | "mean" | "zero" | "legacy" (old 11x11 avg-pool).
        metric: "logit" (default, target log-odds) | "prob" (softmax probability).
        normalize: rescale evidence to 0 (baseline) .. 1 (original image).
        n_random: number of random pixel orders to score as a control (0 = skip).

    Returns (all previous keys are kept):
        insertion_auc, deletion_auc, faithfulness_score, auc_difference,
        step_fractions, insertion_curve, deletion_curve, target_class,
    plus:
        insertion_curve_prob / deletion_curve_prob (raw softmax probabilities),
        metric, substrate, degenerate, baseline_prob, original_prob,
        random_faithfulness, faithfulness_vs_random (only if n_random > 0).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    x = input_tensor.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    h, w = x.shape[2], x.shape[3]

    with torch.no_grad():
        pred_class = int(torch.argmax(model(x), dim=1).item())
    if target_class is None:
        target_class = pred_class

    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.shape != (h, w):
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    order = np.argsort(-heat.flatten(), kind="stable")  # most salient first

    baseline = _make_baseline(x, substrate, blur_sigma_frac)
    fractions = np.linspace(0.0, 1.0, num_steps + 1)
    n = len(fractions)

    def run(order_):
        del_imgs, ins_imgs = _build_images(x, baseline, order_, fractions, h, w)
        logits = _logits_for_images(model, torch.cat([del_imgs, ins_imgs], dim=0))
        return logits[:n], logits[n:]

    del_logits, ins_logits = run(order)
    del_raw = _evidence(del_logits, target_class, metric)
    ins_raw = _evidence(ins_logits, target_class, metric)
    del_prob = _evidence(del_logits, target_class, "prob")
    ins_prob = _evidence(ins_logits, target_class, "prob")

    # Endpoints: deletion[0] and insertion[-1] are the original image;
    # deletion[-1] and insertion[0] are the baseline image.
    m_full = float(del_raw[0])
    m_base = float(ins_raw[0])
    min_gap = 1e-3 if metric == "logit" else 1e-4

    degenerate = False
    if normalize:
        ins_curve, d1 = _normalize_evidence(ins_raw, m_base, m_full, min_gap)
        del_curve, d2 = _normalize_evidence(del_raw, m_base, m_full, min_gap)
        degenerate = d1 or d2
    else:
        ins_curve, del_curve = ins_raw, del_raw

    ins_auc, del_auc, score = _faithfulness_from_curves(ins_curve, del_curve, fractions)

    result: Dict[str, Any] = {
        "target_class": target_class,
        "insertion_auc": round(ins_auc, 4),
        "deletion_auc": round(del_auc, 4),
        "faithfulness_score": round(score, 4),
        "auc_difference": round(ins_auc - del_auc, 4),
        "step_fractions": fractions.tolist(),
        "insertion_curve": [round(float(v), 4) for v in ins_curve],
        "deletion_curve": [round(float(v), 4) for v in del_curve],
        "insertion_curve_prob": [round(float(v), 4) for v in ins_prob],
        "deletion_curve_prob": [round(float(v), 4) for v in del_prob],
        "metric": metric,
        "substrate": substrate,
        "degenerate": bool(degenerate),
        "original_prob": round(float(del_prob[0]), 4),
        "baseline_prob": round(float(ins_prob[0]), 4),
    }

    # Optional control: how does a RANDOM pixel order score on the same image?
    if n_random > 0 and normalize and not degenerate:
        rng = np.random.default_rng(seed)
        rand_scores = []
        for _ in range(int(n_random)):
            d_l, i_l = run(rng.permutation(h * w))
            d_c, _ = _normalize_evidence(_evidence(d_l, target_class, metric), m_base, m_full, min_gap)
            i_c, _ = _normalize_evidence(_evidence(i_l, target_class, metric), m_base, m_full, min_gap)
            rand_scores.append(_faithfulness_from_curves(i_c, d_c, fractions)[2])
        rand_mean = float(np.mean(rand_scores))
        result["random_faithfulness"] = round(rand_mean, 4)
        result["faithfulness_vs_random"] = round(score - rand_mean, 4)

    return result


def compute_ssim_explanation_drift(
    baseline_heatmap: np.ndarray,
    current_heatmap: np.ndarray
) -> float:
    """
    Computes Explanation Drift (D) between the round-1 baseline heatmap
    and the current round heatmap on a reference retinal image.
    
    Drift D = 1.0 - max(0.0, SSIM(current, baseline))
    D = 0 -> No drift (identical explanation)
    D = 1 -> Maximum drift (explanation severely degraded or distorted)
    """
    # Ensure float32 in [0, 1]
    base = np.clip(baseline_heatmap.astype(np.float32), 0.0, 1.0)
    curr = np.clip(current_heatmap.astype(np.float32), 0.0, 1.0)

    # SSIM on 2D arrays with data_range=1.0
    sim_score = ssim(base, curr, data_range=1.0)
    drift = 1.0 - max(0.0, float(sim_score))
    return float(np.clip(drift, 0.0, 1.0))


def compute_average_drift(
    baseline_heatmaps: List[np.ndarray],
    current_heatmaps: List[np.ndarray]
) -> float:
    """Computes mean SSIM drift across a set of reference retinal images."""
    if not baseline_heatmaps or not current_heatmaps:
        return 0.0
    drifts = [
        compute_ssim_explanation_drift(b, c)
        for b, c in zip(baseline_heatmaps, current_heatmaps)
    ]
    return float(np.mean(drifts))