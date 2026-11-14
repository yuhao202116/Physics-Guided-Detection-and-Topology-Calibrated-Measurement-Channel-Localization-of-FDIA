from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from tqdm import tqdm
from data.loaders import load_case, load_measurement, load_load_pv, load_dataset
from configs.config import sys_config, save_metric

def extract_snapshot(temporal_data):
    """Convert (Batch, Time_steps, Features) to (Features, 1) column vector."""
    if hasattr(temporal_data, "numpy"):
        data_np = temporal_data.detach().cpu().numpy()
    else:
        data_np = temporal_data
    return data_np[0, -1, :].reshape(-1, 1)

# 1. Run Evaluation Loop and Gather BDD Residuals
case_class = load_case()
z_noise_summary, v_est_summary = load_measurement()
load_active, load_reactive, pv_active_, pv_reactive_ = load_load_pv()
test_dataloader_unscaled = load_dataset()[1]

# Attack scenario parameters
ang_no_list = [1, 2, 3]
mag_no = 0
ang_str_list = [0.2, 0.3]
mag_str = 0

# Metric containers
residual_bdd_clean = []
residual_bdd_attack = {}
att_strength = {}

print('Evaluating BDD Residuals for Clean Data...')
for idx_, (idx, input_data, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):
    v_est_single = v_est_last.flatten().numpy() if hasattr(v_est_last, "numpy") else v_est_last.flatten()
    z_clean_single = extract_snapshot(input_data)

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        res_clean = case_class.bdd_residual(z_noise=z_clean_single, v_est=v_est_single)

    residual_bdd_clean.append(res_clean)

print('Evaluating BDD Residuals for Attack Data...')
for ang_no in ang_no_list:
    for ang_str in ang_str_list:
        print(f'Attack Scenario: (ang_no={ang_no}, ang_str={ang_str})')

        scenario_key = f'({ang_no},{ang_str})'
        residual_bdd_attack[scenario_key] = []
        att_strength[scenario_key] = []

        for idx_, (idx, input_data, v_est_pre, v_est_last) in tqdm(enumerate(test_dataloader_unscaled)):
            v_est_true_flat = v_est_last.flatten()

            # Generate dynamic FDI attack using temporal data
            z_att_noise, v_att_est_last = case_class.gen_fdi_att_dd(
                z_noise=input_data,
                v_est_last=v_est_true_flat,
                ang_no=ang_no,
                mag_no=mag_no,
                ang_str=ang_str,
                mag_str=mag_str
            )

            z_att_single = extract_snapshot(z_att_noise)
            v_att_single = v_att_est_last.flatten().numpy() if hasattr(v_att_est_last, "numpy") else v_att_est_last.flatten()

            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                res_att = case_class.bdd_residual(z_noise=z_att_single, v_est=v_att_single)

            residual_bdd_attack[scenario_key].append(res_att)

            # Calculate ground truth attack strength
            v_true_np = v_est_true_flat.numpy() if hasattr(v_est_true_flat, "numpy") else v_est_true_flat
            vang_true = np.angle(v_true_np)
            vang_att = np.angle(v_att_single)
            att_strength[scenario_key].append(np.linalg.norm(vang_att - vang_true, 2))

# Save calculated metric files
metric_address = f'metric/{sys_config["case_name"]}/metric_pure_bdd.npy'
save_metric(
    address=metric_address,
    residual_bdd_clean=residual_bdd_clean,
    residual_bdd_attack=residual_bdd_attack,
    att_strength=att_strength
)
print("Pure BDD evaluation completed and metrics saved successfully!\n")



# 2. Load Saved File and Compute Classification Performance (ACC, Prec, Recall, F1)
print('Loading saved metrics to calculate classification performance...')
metrics = np.load(metric_address, allow_pickle=True).item()

res_clean = np.array(metrics["residual_bdd_clean"])
res_attack_dict = metrics["residual_bdd_attack"]

# Define threshold based on the 92nd percentile (quantile = 0.92) as set in the ipynb
quantile_val = 0.92
threshold = np.quantile(res_clean, quantile_val)

print(f"Selected Quantile from ipynb: {quantile_val}")
print(f"Calculated BDD Threshold: {threshold:.6f}\n")

# Lists to store global decisions for merged evaluation (matching Cell 9 logic in ipynb)
all_tp_results = []
all_fp_results = (res_clean > threshold).tolist()

print("-" * 75)
print(f"{'Scenario (ang_no, str)':<25} | {'ACC':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
print("-" * 75)

# --- 2.1 Individual Scenario Breakdown ---
for scenario, res_attack in res_attack_dict.items():
    res_attack = np.array(res_attack)

    # Calculate confusion matrix elements
    tp = np.sum(res_attack > threshold)
    fn = np.sum(res_attack <= threshold)
    fp = np.sum(res_clean > threshold)
    tn = np.sum(res_clean <= threshold)

    # Append current decision sequences to global lists for combined metrics
    all_tp_results.extend((res_attack > threshold).tolist())

    # Calculate individual classification indicators
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"{scenario:<25} | {acc:.4f} | {precision:.4f}    | {recall:.4f} | {f1:.4f}")

print("-" * 75)


# --- 2.2 Global Merged Metrics (Matches the exact layout of ipynb Cell 9) ---
TP_count = np.sum(all_tp_results)
FN_count = len(all_tp_results) - TP_count
FP_count = np.sum(all_fp_results)
TN_count = len(all_fp_results) - FP_count

TP_FN_count = len(all_tp_results)
FP_TN_count = len(all_fp_results)

# Build unified ground truth and predictions arrays
Y_true = np.concatenate([np.ones(TP_FN_count), np.zeros(FP_TN_count)])
Y_pred = np.concatenate([np.array(all_tp_results), np.array(all_fp_results)])

# Compute unified performance metrics
global_acc = (TP_count + TN_count) / (TP_FN_count + FP_TN_count)
global_precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0.0
global_recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0.0
global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0.0

print(f"\nMetrics for selected Threshold (Quantile {quantile_val}) [Overall Merged]:")
print(f" Total Attacked Samples (TP+FN): {TP_FN_count}")
print(f" Total Normal Samples (FP+TN): {FP_TN_count}")
print(f" True Positives (TP): {TP_count}")
print(f" False Positives (FP): {FP_count}")
print(f" True Negatives (TN): {TN_count}")
print(f" Accuracy (ACC): {global_acc:.4f}")
print(f" Precision: {global_precision:.4f}")
print(f" Recall (TPR): {global_recall:.4f}")
print(f" F1-Score: {global_f1:.4f}")
print("-" * 75)
