"""
Page 2: Explanation Quality & Faithfulness Auditing.
Examines the impact of differential privacy noise on visual explanations,
evaluates Insertion/Deletion AUC, and visualizes SSIM Explanation Drift (D).
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from src.inference import RetinalDiagnosticEngine
from src.gradcam import overlay_heatmap
from src.faithfulness import compute_ssim_explanation_drift, compute_insertion_deletion_auc
from src.data_preprocessing import ben_graham_normalize, DR_CLASSES

st.set_page_config(page_title="Explanation Quality | FaithFed-XAI", page_icon="📈", layout="wide")

st.title("📈 Explanation Quality & Faithfulness Auditing")
st.markdown("Quantifying the clinical fidelity of Grad-CAM heatmaps under differential privacy noise using **Insertion/Deletion AUC** and **SSIM Explanation Drift ($D$)**.")

# Architectural Note
with st.expander("ℹ️ The Privacy-Explainability Trade-off in Medical AI", expanded=False):
    st.markdown("""
    In standard deep learning, Differential Privacy (DP-SGD) injects calibrated Gaussian noise into per-sample gradients:
    $$\\mathbf{g} \\leftarrow \\text{clip}(\\mathbf{g}, C) + \\mathcal{N}(0, \\sigma^2 C^2 \\mathbf{I})$$
    While this provides mathematical guarantees against gradient inversion attacks, prior literature (Saifullah et al., 2024) 
    demonstrates that DP noise can severely distort model attributions. In medical imaging, this may cause Grad-CAM to highlight uninformative background pixels rather than true retinal lesions.
    
    **FaithFed-XAI Solution:**
    Our framework measures Explanation Drift $D = 1 - \\text{SSIM}(\\text{Heatmap}^{(t)}, \\text{Heatmap}^{(1)})$ after each round.
    Hospitals experiencing significant explanation distortion are automatically allocated higher privacy slack (larger $\\epsilon$, less noise) in subsequent rounds!
    """)

# Interactive Faithfulness Simulator
st.subheader("1. Interactive Insertion & Deletion Faithfulness Audit")
col_sel1, col_sel2 = st.columns([2, 1])

@st.cache_resource
def get_engine():
    model_path = "artifacts/global_model.pth" if os.path.exists("artifacts/global_model.pth") else None
    return RetinalDiagnosticEngine(model_path=model_path, img_size=128)

engine = get_engine()

# Select a sample fundus image
sample_dir = "data/aptos/train_images"
all_samples = [f for f in os.listdir(sample_dir) if f.endswith(".png")][:20]

with col_sel1:
    selected_sample = st.selectbox("Select Patient Retinal Image for Verification", all_samples)

with col_sel2:
    audit_steps = st.slider("Evaluation Steps", min_value=5, max_value=20, value=10, step=5)

sample_path = os.path.join(sample_dir, selected_sample)
report = engine.diagnose(sample_path, compute_faithfulness=True)

col_vis1, col_vis2, col_vis3 = st.columns(3)

with col_vis1:
    st.image(report["normalized_image_rgb"], caption="Preprocessed Fundus Image", width="stretch")

with col_vis2:
    st.image(report["heatmap"], caption="Grad-CAM Lesion Heatmap", width="stretch")

with col_vis3:
    st.image(report["overlay_normalized"], caption="Grad-CAM Overlay", width="stretch")

# Curves Plot
f_data = report["faithfulness"]
c1, c2, c3 = st.columns(3)
c1.metric("Faithfulness Score (0-1)", f"{f_data['faithfulness_score']:.4f}", help="Composite measure of truthfulness")
c2.metric("Insertion AUC", f"{f_data['insertion_auc']:.4f}", help="Higher is better: confidence rises quickly")
c3.metric("Deletion AUC", f"{f_data['deletion_auc']:.4f}", help="Lower is better: confidence collapses quickly")

fig, ax = plt.subplots(figsize=(8, 3.8))
steps = f_data["step_fractions"]
ax.plot(steps, f_data["insertion_curve"], "g-o", linewidth=2, label=f"Insertion Curve (AUC = {f_data['insertion_auc']:.3f})")
ax.plot(steps, f_data["deletion_curve"], "r--s", linewidth=2, label=f"Deletion Curve (AUC = {f_data['deletion_auc']:.3f})")
ax.fill_between(steps, f_data["insertion_curve"], alpha=0.15, color="green")
ax.fill_between(steps, f_data["deletion_curve"], alpha=0.15, color="red")
ax.set_xlabel("Proportion of Most-Salient Pixels Modified", fontsize=10)
ax.set_ylabel("Predicted Class Probability", fontsize=10)
ax.set_title("Clinical Faithfulness Profile (Truthfulness Verification)", fontsize=11, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, linestyle=":", alpha=0.6)
st.pyplot(fig)

st.markdown("---")

# Section 2: SSIM Explanation Drift Across Rounds
st.subheader("2. Explanation Drift ($D$) Across Federated Rounds")

round_log_file = "artifacts/round_log.csv"
if os.path.exists(round_log_file):
    round_df = pd.read_csv(round_log_file)
    st.markdown("Live telemetry recorded from federated training:")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Mean Faithfulness Score Across Rounds**")
        st.line_chart(round_df.set_index("round")["mean_faithfulness"])
    
    with col_chart2:
        st.markdown("**Cross-Hospital Explanation Disparity**")
        st.line_chart(round_df.set_index("round")["explanation_disparity"])
else:
    st.info("Federated training logs (`artifacts/round_log.csv`) will be displayed here once training is initiated.")

# Visual Explanation Drift Demonstration
st.subheader("3. Visualizing Explanation Drift: Baseline vs Perturbed Heatmaps")

col_drift1, col_drift2, col_drift3 = st.columns(3)

base_heatmap = report["heatmap"]
# Simulate slight DP gradient perturbation
np.random.seed(42)
noise = np.random.normal(0, 0.15, base_heatmap.shape)
perturbed_heatmap = np.clip(cv2.GaussianBlur(base_heatmap + noise, (5, 5), 1.0), 0.0, 1.0)
drift_val = compute_ssim_explanation_drift(base_heatmap, perturbed_heatmap)
diff_map = np.abs(base_heatmap - perturbed_heatmap)

with col_drift1:
    st.image(base_heatmap, caption="Round 1 Baseline Heatmap", width="stretch")

with col_drift2:
    st.image(perturbed_heatmap, caption="Current Round (With DP Noise)", width="stretch")

with col_drift3:
    st.image(diff_map, caption=f"Explanation Difference Map (Drift D = {drift_val:.4f})", width="stretch")

st.caption(f"**Structural Similarity (SSIM):** {1.0 - drift_val:.4f} | **Calculated Explanation Drift Signal ($D$):** {drift_val:.4f}")
