"""
Page 1: Retinal Fundus Image Diagnosis & Grad-CAM Visual Explainability.
Provides clinical decision support for ophthalmologists with lesion localization
and pre-diagnostic inspection of per-hospital differential privacy noise and budget.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from src.inference import RetinalDiagnosticEngine, CLINICAL_RISK
from src.data_preprocessing import DR_CLASSES, DR_DESCRIPTIONS, ben_graham_normalize

st.set_page_config(page_title="Retinal Diagnosis | FaithFed-XAI", page_icon="🔬", layout="wide")

st.title("🔬 Retinal Diagnosis & Visual Explanation (Grad-CAM)")
st.markdown("Automated Diabetic Retinopathy grading with Ben Graham preprocessing, class activation heatmaps, and per-hospital differential privacy noise verification.")

# Predefined hospital privacy specifications
HOSPITAL_SPECS = {
    "Global Aggregated Model": {
        "file": "artifacts/global_model.pth",
        "description": "Federated Aggregation (Weighted FedAvg + FedBN)",
        "epsilon": 4.00,
        "noise_multiplier": 0.80,
        "heterogeneity_H": 0.0000,
        "accuracy_loss_A": 0.00,
        "drift_D": 0.00,
        "composite_S": 0.00,
        "type": "Global"
    },
    "Hospital 1 (Rural General Center)": {
        "file": "artifacts/hospitals/hospital_1.pth",
        "description": "Large rural hospital (1,648 patients), predominantly low-stage fundus scans.",
        "epsilon": 3.75,
        "noise_multiplier": 0.88,
        "heterogeneity_H": 0.0494,
        "accuracy_loss_A": 0.04,
        "drift_D": 0.06,
        "composite_S": 0.051,
        "type": "Local Client"
    },
    "Hospital 2 (Community Clinic)": {
        "file": "artifacts/hospitals/hospital_2.pth",
        "description": "Medium community eye clinic (553 patients) with mixed demographics.",
        "epsilon": 3.90,
        "noise_multiplier": 0.85,
        "heterogeneity_H": 0.0608,
        "accuracy_loss_A": 0.03,
        "drift_D": 0.08,
        "composite_S": 0.056,
        "type": "Local Client"
    },
    "Hospital 3 (Specialized Retina Center)": {
        "file": "artifacts/hospitals/hospital_3.pth",
        "description": "Small specialized surgical center (271 patients) with high demographic skew.",
        "epsilon": 4.65,
        "noise_multiplier": 0.71,
        "heterogeneity_H": 0.2536,
        "accuracy_loss_A": 0.08,
        "drift_D": 0.18,
        "composite_S": 0.171,
        "type": "Local Client (High Relief)"
    },
    "Hospital 4 (Urban Eye Hospital)": {
        "file": "artifacts/hospitals/hospital_4.pth",
        "description": "Urban public eye facility (538 patients) with balanced screening.",
        "epsilon": 3.70,
        "noise_multiplier": 0.89,
        "heterogeneity_H": 0.0399,
        "accuracy_loss_A": 0.02,
        "drift_D": 0.05,
        "composite_S": 0.036,
        "type": "Local Client"
    },
    "Hospital 5 (Tertiary Diabetes Center)": {
        "file": "artifacts/hospitals/hospital_5.pth",
        "description": "Tertiary referral hospital (652 patients) rich in proliferative DR cases.",
        "epsilon": 4.00,
        "noise_multiplier": 0.82,
        "heterogeneity_H": 0.0589,
        "accuracy_loss_A": 0.05,
        "drift_D": 0.11,
        "composite_S": 0.071,
        "type": "Local Client"
    }
}
from src.dp_report import load_dp_report, latest_hospital_telemetry
_tele = latest_hospital_telemetry(load_dp_report())
if _tele:
    for _name, _spec in HOSPITAL_SPECS.items():
        if not _name.startswith("Hospital"):
            continue
        _t = _tele.get(f"hospital_{_name.split()[1]}")
        if _t:
            _spec["epsilon"] = _t["epsilon_spent"] if _t["epsilon_spent"] is not None else _t["allocated_epsilon"]
            _spec["noise_multiplier"] = _t["noise_multiplier"] if _t["noise_multiplier"] is not None else (_t.get("next_noise_multiplier_formula") or 0.0)
            _spec["heterogeneity_H"] = _t["H_raw"]
            _spec["accuracy_loss_A"] = _t["A"] or 0.0
            _spec["drift_D"] = _t["drift_D"]
            _spec["composite_S"] = _t["composite_S"] or 0.0
    _locals = [s for n, s in HOSPITAL_SPECS.items() if n.startswith("Hospital")]
    for _k in ("epsilon", "noise_multiplier", "heterogeneity_H", "accuracy_loss_A", "drift_D", "composite_S"):
        HOSPITAL_SPECS["Global Aggregated Model"][_k] = sum(s[_k] for s in _locals) / len(_locals)
# Sidebar: Hospital Selection & Privacy Noise
st.sidebar.header("1. Active Model & DP Noise")
selected_hospital_name = st.sidebar.selectbox(
    "Choose Clinical Model Source",
    list(HOSPITAL_SPECS.keys()),
    index=0
)
active_spec = HOSPITAL_SPECS[selected_hospital_name]

# Check if hospital model file exists, fallback to global model
model_file_to_use = active_spec["file"]
if not os.path.exists(model_file_to_use):
    model_file_to_use = "artifacts/global_model.pth"

@st.cache_resource
def get_engine_for_model(model_path: str):
    if not os.path.exists(model_path):
        model_path = None
    return RetinalDiagnosticEngine(model_path=model_path, img_size=128)

engine = get_engine_for_model(model_file_to_use)

# Hospital Differential Privacy Noise Inspection Banner
with st.expander(f"🛡️ Hospital Privacy & Noise Inspection: {selected_hospital_name}", expanded=True):
    col_n1, col_n2, col_n3, col_n4, col_n5 = st.columns(5)
    
    with col_n1:
        st.metric(
            label="Privacy Budget (ε)",
            value=f"{active_spec['epsilon']:.2f}",
            help="Higher ε grants more privacy allowance / less perturbation"
        )
    with col_n2:
        st.metric(
            label="Noise Multiplier (σ)",
            value=f"{active_spec['noise_multiplier']:.2f}",
            help="Standard deviation of Gaussian noise added to per-sample clipped gradients in DP-SGD"
        )
    with col_n3:
        st.metric(
            label="Heterogeneity (H)",
            value=f"{active_spec['heterogeneity_H']:.4f}",
            help="Jensen-Shannon divergence between this clinic's label distribution and the global distribution"
        )
    with col_n4:
        st.metric(
            label="Explanation Drift (D)",
            value=f"{active_spec['drift_D']:.2f}",
            help="1 - SSIM relative to round-1 baseline heatmap"
        )
    with col_n5:
        st.metric(
            label="Composite S Score",
            value=f"{active_spec['composite_S']:.3f}",
            help="S = w_h*H + w_a*A + w_e*D (drives adaptive budget adjustment)"
        )
    
    st.caption(f"**Hospital Profile:** {active_spec['description']}")
    if active_spec["epsilon"] > 4.0:
        st.info("💡 **Adaptive Controller Relief Active:** Hospital 3 receives higher privacy budget (ε=4.65) and lower noise (σ=0.71) because of high data heterogeneity and explanation drift.")
    if selected_hospital_name.startswith("Hospital") and active_spec["epsilon"] > 4.0:
        st.info(f"💡 **Adaptive Controller Relief Active:** this hospital receives a higher privacy budget (ε={active_spec['epsilon']:.2f}) and lower noise (σ={active_spec['noise_multiplier']:.2f}) because of its data heterogeneity, accuracy loss and explanation drift.")
# Curated canonical sample images for all 5 grades (verified in train.csv with clinical progression)
CANONICAL_APTOS_SAMPLES = {
    0: "002c21358ce6",  # Grade 0: Normal retina (no lesions)
    1: "0024cdab0c1e",  # Grade 1: Mild NPDR (microaneurysms only)
    2: "0161338f53cc",  # Grade 2: Moderate NPDR (hemorrhages + exudates)
    3: "093a42649c29",  # Grade 3: Severe NPDR (multi-quadrant hemorrhages, venous beading)
    4: "2fe06bedb2c4",  # Grade 4: Proliferative DR (extensive neovascularization and advanced lesions)
}

@st.cache_data
def get_sample_images():
    csv_path = "data/aptos/train.csv"
    img_dir = "data/aptos/train_images"
    samples = {}
    if os.path.exists(csv_path) and os.path.exists(img_dir):
        df = pd.read_csv(csv_path)
        for grade in range(5):
            preferred_id = CANONICAL_APTOS_SAMPLES.get(grade)
            path = os.path.join(img_dir, f"{preferred_id}.png") if preferred_id else ""
            if os.path.exists(path):
                samples[grade] = {"path": path, "id": preferred_id, "grade": grade}
            else:
                grade_df = df[df["diagnosis"] == grade]
                for _, row in grade_df.iterrows():
                    fallback_path = os.path.join(img_dir, f"{row['id_code']}.png")
                    if os.path.exists(fallback_path):
                        samples[grade] = {"path": fallback_path, "id": row["id_code"], "grade": grade}
                        break
    return samples

sample_dict = get_sample_images()

# Sidebar: Image selection
st.sidebar.markdown("---")
st.sidebar.header("2. Retinal Image Selection")
input_mode = st.sidebar.radio("Image Source", ["Select Curated APTOS Sample", "Upload Custom Fundus Image"])

selected_image = None
image_label = ""
ground_truth_grade = None

if input_mode == "Select Curated APTOS Sample":
    sample_options = [f"Grade {g}: {DR_CLASSES[g]}" for g in sorted(sample_dict.keys())]
    selected_option = st.sidebar.selectbox("Choose Sample Case", sample_options, index=0)
    grade_idx = int(selected_option.split(":")[0].replace("Grade ", "").strip())
    selected_image = sample_dict[grade_idx]["path"]
    ground_truth_grade = grade_idx
    image_label = f"Patient ID: {sample_dict[grade_idx]['id']} (Ground Truth: Grade {grade_idx} — {DR_CLASSES[grade_idx]})"
else:
    uploaded_file = st.sidebar.file_uploader("Upload Retinal Image (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        image_label = f"Uploaded File: {uploaded_file.name}"

st.sidebar.markdown("---")
st.sidebar.header("3. Grad-CAM Settings")
overlay_alpha = st.sidebar.slider("Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
colormap_name = st.sidebar.selectbox("Colormap", ["JET", "TURBO", "INFERNO", "VIRIDIS"])
colormap_map = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS
}
selected_colormap = colormap_map[colormap_name]

if selected_image is None:
    st.info("Please select or upload a fundus photograph to initiate diagnosis.")
else:
    st.markdown(f"**Current Subject:** `{image_label}`")

    # Run Diagnosis
    with st.spinner(f"Executing diagnosis using {selected_hospital_name}..."):
        report = engine.diagnose(selected_image, compute_faithfulness=True)

    pred_grade = report["predicted_grade"]
    risk_info = report["clinical_risk"]

    # Top Diagnostic Banner
    col_pred1, col_pred2, col_pred3 = st.columns([1.5, 1, 1])

    with col_pred1:
        st.markdown(f"""
        <div style="background-color: {risk_info['color']}15; border-left: 6px solid {risk_info['color']}; padding: 15px; border-radius: 8px;">
            <h3 style="margin: 0; color: {risk_info['color']};">Diagnosis: Grade {pred_grade} — {report['grade_name']}</h3>
            <p style="margin: 5px 0 0 0; font-size: 0.95rem; color: #374151;"><b>Description:</b> {report['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_pred2:
        is_match = (ground_truth_grade == pred_grade) if ground_truth_grade is not None else None
        delta_str = "Exact Match" if is_match else ("Discrepancy" if ground_truth_grade is not None else None)
        st.metric(
            label="Confidence",
            value=f"{report['confidence'] * 100:.1f}%",
            delta=delta_str
        )

    with col_pred3:
        st.metric(
            label="Clinical Risk Level",
            value=risk_info["level"],
            help=risk_info["action"]
        )

    st.caption(f"**Recommended Clinical Action:** {risk_info['action']}")
    st.markdown("---")

    # Visual Inspection & Explainability
    col_img1, col_img2, col_img3 = st.columns(3)

    with col_img1:
        st.subheader("1. Raw Fundus Photo")
        st.image(report["raw_image_rgb"], caption="Raw Captured Fundus Photography", width="stretch")

    with col_img2:
        st.subheader("2. Ben Graham Normalization")
        st.image(report["normalized_image_rgb"], caption="Local-Average Color Subtraction (σ=10)", width="stretch")

    with col_img3:
        st.subheader("3. Grad-CAM Lesion Map")
        from src.gradcam import overlay_heatmap
        custom_overlay = overlay_heatmap(
            report["normalized_image_rgb"],
            report["heatmap"],
            alpha=overlay_alpha,
            colormap=selected_colormap
        )
        st.image(custom_overlay, caption=f"Grad-CAM on Conv Block 5 / Layer 4 ({colormap_name})", width="stretch")

    st.markdown("---")

    # Detailed Analysis Tabs
    tab_probs, tab_faith, tab_pathology = st.tabs([
        "Class Probabilities & Confidence",
        "Explanation Faithfulness (Insertion/Deletion)",
        "Clinical Pathology Correlation"
    ])

    with tab_probs:
        prob_df = pd.DataFrame([
            {"DR Grade": f"Grade {i}: {k}", "Probability (%)": round(v * 100, 2)}
            for i, (k, v) in enumerate(report["class_probabilities"].items())
        ])
        st.bar_chart(prob_df.set_index("DR Grade"), height=260)
        st.dataframe(prob_df, width="stretch")

    with tab_faith:
        f_info = report["faithfulness"]
        if f_info:
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("Faithfulness Score", f"{f_info['faithfulness_score']:.4f}", help="(Ins_AUC + (1 - Del_AUC)) / 2")
            f_col2.metric("Insertion AUC", f"{f_info['insertion_auc']:.4f}", help="Higher is better: confidence rises quickly as salient pixels are inserted")
            f_col3.metric("Deletion AUC", f"{f_info['deletion_auc']:.4f}", help="Lower is better: confidence collapses quickly as salient pixels are deleted")

            fig, ax = plt.subplots(figsize=(7, 3.5))
            steps = f_info["step_fractions"]
            ax.plot(steps, f_info["insertion_curve"], "g-o", label=f"Insertion (AUC={f_info['insertion_auc']:.3f})")
            ax.plot(steps, f_info["deletion_curve"], "r--s", label=f"Deletion (AUC={f_info['deletion_auc']:.3f})")
            ax.set_xlabel("Fraction of Salient Pixels Modified")
            ax.set_ylabel("Target Class Probability")
            ax.set_title("Grad-CAM Faithfulness Verification Curves")
            ax.legend(loc="best")
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)
        else:
            st.info("Faithfulness evaluation was disabled for this run.")

    with tab_pathology:
        st.markdown("""
        ### Pathological Hallmark Reference
        The Grad-CAM heatmap identifies key microvascular changes:
        - **Microaneurysms:** Small round red dots; earliest clinically observable sign of DR.
        - **Dot and Blot Hemorrhages:** Bleeding into middle retinal layers (Mild / Moderate DR).
        - **Hard Exudates:** Waxy yellow lipid deposits with sharp margins from vascular leakage.
        - **Cotton Wool Spots:** Fluffy white patches resulting from arteriolar occlusion and retinal nerve ischemia.
        - **Neovascularization:** Fragile new vessels on optic disc (NVD) or elsewhere (NVE) characteristic of **Proliferative DR (Grade 4)**.
        """)
