"""
Helper script to generate valid Jupyter Notebooks (.ipynb) for the 6 project modules.
"""

import json
import os

def make_notebook(cells):
    return {
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
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

notebooks_dir = "c:/Users/aaa/Downloads/REVIEW 1/Faithfed project/notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

# ----------------- 01_data_and_hospitals.ipynb -----------------
nb1_cells = [
    md_cell("# 01: Data Ingestion, Ben Graham Normalization & Non-IID Hospital Partitioning\n\nThis notebook demonstrates:\n1. Loading the APTOS 2019 Blindness Detection dataset.\n2. Ben Graham's local-average color-subtraction normalization.\n3. Simulating 5 heterogeneous, non-IID hospitals using Dirichlet(alpha=0.5) distribution.\n4. Computing the Heterogeneity Score (H) via Jensen-Shannon Divergence."),
    code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\n\nimport cv2\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.data_preprocessing import ben_graham_normalize, DR_CLASSES, DR_DESCRIPTIONS\nfrom src.hospital_partition import partition_dataset_dirichlet, save_splits, load_splits"),
    code_cell("df = pd.read_csv('../data/aptos/train.csv')\nprint(f'Total dataset records: {len(df)}')\nprint(df['diagnosis'].value_counts().sort_index())"),
    md_cell("### Ben Graham Illumination Normalization\nFundus photographs often exhibit non-uniform lighting and scanner variations across hospital sites. Ben Graham normalization accentuates retinal lesions (hemorrhages, exudates, microaneurysms) while suppressing illumination artifacts."),
    code_cell("sample_img_id = df.iloc[0]['id_code']\nsample_path = f'../data/aptos/train_images/{sample_img_id}.png'\nraw_rgb = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)\nnormalized_rgb = ben_graham_normalize(raw_rgb, img_size=224, sigma=10)\n\nfig, axes = plt.subplots(1, 2, figsize=(10, 5))\naxes[0].imshow(raw_rgb)\naxes[0].set_title('Raw Fundus Photography')\naxes[0].axis('off')\naxes[1].imshow(normalized_rgb)\naxes[1].set_title('Ben Graham Normalized (σ=10)')\naxes[1].axis('off')\nplt.tight_layout()\nplt.show()"),
    md_cell("### 5 Non-IID Hospitals Dirichlet Partitioning & Jensen-Shannon Heterogeneity (H)"),
    code_cell("splits, summary_df = partition_dataset_dirichlet(df, num_hospitals=5, alpha=0.5, seed=42)\nsave_splits(splits, '../artifacts/splits.json')\nsummary_df")
]

with open(os.path.join(notebooks_dir, "01_data_and_hospitals.ipynb"), "w") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# ----------------- 02_federated_training.ipynb -----------------
nb2_cells = [
    md_cell("# 02: Personalized Federated Learning with FedBN & Secure Aggregation\n\nThis notebook demonstrates:\n1. SmallCNN model with GroupNorm (compatible with DP-SGD and FedBN).\n2. Separating shared parameters (Conv, Linear) from local parameters (GroupNorm affine).\n3. Pairwise Additive Masking (Bonawitz et al.) for cryptographic privacy.\n4. Running federated rounds."),
    code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\n\nimport torch\nimport yaml\nfrom src.model import get_model\nfrom src.fedbn import extract_shared_parameters, extract_local_parameters\nfrom src.secure_aggregation import create_pairwise_masks_for_clients, verify_mask_cancellation\nfrom src.federated_trainer import FederatedTrainer"),
    code_cell("model = get_model(num_classes=5)\nshared = extract_shared_parameters(model)\nlocal = extract_local_parameters(model)\nprint(f'Shared layers count: {len(shared)}')\nprint(f'Local (GroupNorm) layers count: {len(local)}')"),
    code_cell("client_ids = [f'hospital_{i+1}' for i in range(5)]\nmasks = create_pairwise_masks_for_clients(client_ids, shared, round_idx=1)\nis_exact, residual = verify_mask_cancellation(masks)\nprint(f'Exact pairwise mask cancellation: {is_exact} (Max residual: {residual:.2e})')"),
    code_cell("with open('../configs/default.yaml', 'r') as f:\n    config = yaml.safe_load(f)\nconfig['data']['csv_path'] = '../data/aptos/train.csv'\nconfig['data']['img_dir'] = '../data/aptos/train_images'\nconfig['output']['artifacts_dir'] = '../artifacts'\nconfig['output']['results_dir'] = '../results'\n\ntrainer = FederatedTrainer(config=config)\nprint('Federated Trainer initialized successfully!')")
]

with open(os.path.join(notebooks_dir, "02_federated_training.ipynb"), "w") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# ----------------- 03_privacy_analysis.ipynb -----------------
nb3_cells = [
    md_cell("# 03: Explanation-Aware Adaptive Privacy Controller (H + A + D)\n\nThis notebook analyzes:\n1. Calibrated Gaussian noise calculation for Differential Privacy (epsilon, delta).\n2. Composite score calculation: S = w_h*H + w_a*A + w_e*D.\n3. Conservation of total privacy budget across non-IID clients."),
    code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.differential_privacy import AdaptivePrivacyController, compute_noise_multiplier"),
    code_cell("controller = AdaptivePrivacyController(num_hospitals=5, target_epsilon=4.0, target_delta=1e-5)\nh_ids = [f'hospital_{i+1}' for i in range(5)]\ninit_budgets = controller.initialize_round_1(h_ids)\npd.DataFrame(init_budgets).T"),
    code_cell("# Simulate Round 2 budget adaptation\nheterogeneity = {'hospital_1': 0.049, 'hospital_2': 0.061, 'hospital_3': 0.254, 'hospital_4': 0.040, 'hospital_5': 0.059}\naccs = {'hospital_1': 0.81, 'hospital_2': 0.79, 'hospital_3': 0.72, 'hospital_4': 0.82, 'hospital_5': 0.78}\ndrifts = {'hospital_1': 0.05, 'hospital_2': 0.08, 'hospital_3': 0.21, 'hospital_4': 0.04, 'hospital_5': 0.11}\n\nnext_budgets = controller.compute_next_round_budgets(2, h_ids, heterogeneity, accs, drifts)\npd.DataFrame(next_budgets).T")
]

with open(os.path.join(notebooks_dir, "03_privacy_analysis.ipynb"), "w") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# ----------------- 04_gradcam.ipynb -----------------
nb4_cells = [
    md_cell("# 04: Grad-CAM Explainability & Visual Lesion Localization\n\nThis notebook demonstrates:\n1. Extracting gradients with respect to Conv Block 5 activations.\n2. Generating normalized class activation heatmaps.\n3. Overlaying heatmaps on fundus photographs across all 5 Diabetic Retinopathy grades."),
    code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\n\nimport cv2\nimport torch\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom src.model import get_model\nfrom src.gradcam import GradCAM, overlay_heatmap\nfrom src.inference import RetinalDiagnosticEngine\nfrom src.data_preprocessing import DR_CLASSES"),
    code_cell("engine = RetinalDiagnosticEngine(model_path=None, img_size=128)\nsample_path = '../data/aptos/train_images/000c1434d8d7.png'\nreport = engine.diagnose(sample_path, compute_faithfulness=False)\n\nfig, axes = plt.subplots(1, 3, figsize=(12, 4))\naxes[0].imshow(report['raw_image_rgb'])\naxes[0].set_title('Raw Fundus')\naxes[0].axis('off')\naxes[1].imshow(report['heatmap'], cmap='jet')\naxes[1].set_title('Grad-CAM Heatmap (Conv 5)')\naxes[1].axis('off')\naxes[2].imshow(report['overlay_raw'])\naxes[2].set_title(f'Diagnosis: {report[\"grade_name\"]} ({report[\"confidence\"]*100:.1f}%)')\naxes[2].axis('off')\nplt.tight_layout()\nplt.show()")
]

with open(os.path.join(notebooks_dir, "04_gradcam.ipynb"), "w") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

# ----------------- 05_faithfulness.ipynb -----------------
nb5_cells = [
    md_cell("# 05: Faithfulness Evaluation & SSIM Explanation Drift\n\nThis notebook demonstrates:\n1. Insertion and Deletion curves for Grad-CAM truthfulness verification.\n2. Computing Insertion AUC, Deletion AUC, and composite Faithfulness Score.\n3. Tracking SSIM explanation drift across private training."),
    code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\n\nimport matplotlib.pyplot as plt\nfrom src.inference import RetinalDiagnosticEngine\nfrom src.faithfulness import compute_insertion_deletion_auc, compute_ssim_explanation_drift"),
    code_cell("engine = RetinalDiagnosticEngine(model_path=None, img_size=128)\nsample_path = '../data/aptos/train_images/000c1434d8d7.png'\nreport = engine.diagnose(sample_path, compute_faithfulness=True)\nf_res = report['faithfulness']\n\nplt.figure(figsize=(7, 4))\nplt.plot(f_res['step_fractions'], f_res['insertion_curve'], 'g-o', label=f'Insertion AUC: {f_res[\"insertion_auc\"]:.3f}')\nplt.plot(f_res['step_fractions'], f_res['deletion_curve'], 'r--s', label=f'Deletion AUC: {f_res[\"deletion_auc\"]:.3f}')\nplt.xlabel('Fraction of Pixels Modified')\nplt.ylabel('Confidence')\nplt.title('Insertion/Deletion Faithfulness Verification')\nplt.legend()\nplt.grid(True, linestyle=':')\nplt.show()")
]

with open(os.path.join(notebooks_dir, "05_faithfulness.ipynb"), "w") as f:
    json.dump(make_notebook(nb5_cells), f, indent=2)

# ----------------- 06_fairness.ipynb -----------------
nb6_cells = [
    md_cell("# 06: Client-Level Fairness Evaluation Across 5 Hospitals\n\nThis notebook demonstrates:\n1. Evaluating the 5 Cross-Hospital Client Fairness Criteria.\n2. Disparity analysis (Performance & Explanation Disparity).\n3. Protecting vulnerable small clinics (Small-Client & Heterogeneity Penalties)."),
    code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\n\nimport pandas as pd\nfrom src.fairness import compute_five_fairness_metrics"),
    code_cell("hospital_eval = {\n    'hospital_1': {'accuracy': 0.812, 'faithfulness': 0.645, 'heterogeneity_H': 0.0494, 'sample_count': 1648, 'epsilon': 3.75},\n    'hospital_2': {'accuracy': 0.795, 'faithfulness': 0.628, 'heterogeneity_H': 0.0608, 'sample_count': 553, 'epsilon': 3.90},\n    'hospital_3': {'accuracy': 0.768, 'faithfulness': 0.598, 'heterogeneity_H': 0.2536, 'sample_count': 271, 'epsilon': 4.65},\n    'hospital_4': {'accuracy': 0.804, 'faithfulness': 0.635, 'heterogeneity_H': 0.0399, 'sample_count': 538, 'epsilon': 3.70},\n    'hospital_5': {'accuracy': 0.789, 'faithfulness': 0.612, 'heterogeneity_H': 0.0589, 'sample_count': 652, 'epsilon': 4.00},\n}\n\nfairness = compute_five_fairness_metrics(hospital_eval)\nfor k, v in fairness.items():\n    if k != 'per_hospital_summary':\n        print(f'{k}: {v}')")
]

with open(os.path.join(notebooks_dir, "06_fairness.ipynb"), "w") as f:
    json.dump(make_notebook(nb6_cells), f, indent=2)

print('All 6 notebooks created successfully!')
