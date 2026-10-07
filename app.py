"""
FaithFed-XAI: Streamlit Clinical Decision Support & Research Dashboard.
Main Landing Page: Framework Overview, Architecture, and Live Status.
"""

import os
import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="FaithFed-XAI | Retinal Diagnostic Framework",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: #E0E7FF;
        color: #3730A3;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">FaithFed-XAI: Clinical Retinal Diagnostic Framework</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An Explanation-Aware, Privacy-Adaptive Federated Learning Framework with Client Fairness Evaluation for Medical Imaging Diagnosis</div>', unsafe_allow_html=True)

# Overview Badges
# st.markdown("""
# <div style="display: flex; gap: 10px; margin-bottom: 20px;">
#     <span class="badge">APTOS Retinopathy</span>
#     <span class="badge">FedBN Personalization</span>
#     <span class="badge">Adaptive DP (H+A+D)</span>
#     <span class="badge">Pairwise Secure Aggregation</span>
#     <span class="badge">Grad-CAM XAI</span>
#     <span class="badge">5 Client Fairness Metrics</span>
# </div>
# """, unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/visible.png", width=70)
    st.title("FaithFed-XAI")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Dataset:** APTOS 2019 Blindness Detection")
    st.markdown("**Model:** RetinalNet (ResNet18-GN · FedBN + DP-SGD)")
    st.markdown("---")
    # st.markdown("### Navigation")
    # st.markdown("- **Home:** Framework Architecture")
    # st.markdown("- **1. Diagnose:** Real-time Retinal Inference & Grad-CAM")
    # st.markdown("- **2. Explanation Quality:** Insertion/Deletion AUC & Drift")
    # st.markdown("- **3. Hospital Fairness:** 5-Hospital Non-IID Fairness Audit")
    # st.markdown("- **4. Privacy Budget:** Adaptive DP Controller & Secure Agg")

# Key Framework Pillars
col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.markdown("""
#     <div class="metric-card">
#         <h4 style="color:#1E3A8A; margin-top:0;">5 Non-IID Hospitals</h4>
#         <p style="font-size: 0.9rem; color:#4B5563;">Simulated clinical sites partitioned via Dirichlet(α=0.5) with Jensen-Shannon Divergence heterogeneity scoring.</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="metric-card">
#         <h4 style="color:#1E3A8A; margin-top:0;">Personalized FedBN</h4>
#         <p style="font-size: 0.9rem; color:#4B5563;">GroupNorm affine parameters remain strictly local to counteract scanner/demographic variance while sharing feature extractors.</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="metric-card">
#         <h4 style="color:#1E3A8A; margin-top:0;">Adaptive DP (H+A+D)</h4>
#         <p style="font-size: 0.9rem; color:#4B5563;">Dynamic budget allocation balancing Data Heterogeneity (H), Accuracy Loss (A), and Grad-CAM Drift (D) under fixed average budget.</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col4:
#     st.markdown("""
#     <div class="metric-card">
#         <h4 style="color:#1E3A8A; margin-top:0;">Client Fairness Audit</h4>
#         <p style="font-size: 0.9rem; color:#4B5563;">Auditing Performance & Explanation Disparity, Worst-Client Faithfulness, Small-Client and Heterogeneity Penalties.</p>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("---")

# Clinical Data & Hospital Splits Status
st.subheader("Clinical Data & Non-IID Hospital Splits Status")

splits_file = "artifacts/splits.json"
if os.path.exists(splits_file):
    with open(splits_file, "r") as f:
        splits_data = json.load(f)
        
    st.success(f"Hospital splits verified: {len(splits_data['hospitals'])} non-IID clinical sites generated from {splits_data['metadata']['total_samples']} APTOS images.")
    
    table_data = []
    for hid, info in splits_data["hospitals"].items():
        table_data.append({
            "Hospital": info["hospital_name"],
            "Total Patients": info["total_count"],
            "Train Samples": info["train_count"],
            "Held-Out Test": info["test_count"],
            "Heterogeneity (H)": info["heterogeneity_H"],
            "Class 0 (No DR)": info["class_counts"].get("0", 0),
            "Class 1 (Mild)": info["class_counts"].get("1", 0),
            "Class 2 (Moderate)": info["class_counts"].get("2", 0),
            "Class 3 (Severe)": info["class_counts"].get("3", 0),
            "Class 4 (Proliferative)": info["class_counts"].get("4", 0)
        })
    st.dataframe(pd.DataFrame(table_data), width="stretch")
else:
    st.warning("Splits not yet generated. Run `python scripts/train.py` or partition data.")

st.markdown("---")
# st.markdown("### Quick Diagnostic Demo")
# st.info("👈 Use the sidebar to navigate to **'1_Diagnose'** to test fundus image evaluation, Grad-CAM lesion mapping, and faithfulness scoring!")
