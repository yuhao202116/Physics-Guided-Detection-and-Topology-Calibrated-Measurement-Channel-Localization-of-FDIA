"""Expanded inference-control ablations for TopoLoc-ConvAE."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import torch

from models.localization.topoloc_convae import TopoLocConvAE
from configs.config import sys_config
from configs.config_mea_idx import define_mea_idx_noise
from configs.nn_setting import nn_setting
from evaluation.localization import (
    _aggregate_metrics,
    _attack_plan,
    _build_attack_scenario,
    _metrics_for_sample,
    _test_records,
)
from data.dataset import rnn_dataset_evl, scaler
from data.loaders import load_case
from configs.paths import OUTPUT_ROOT, checkpoint_path


OUT = OUTPUT_ROOT / "topoloc" / sys_config["case_name"]
CASE_NAME = sys_config["case_name"]
SEED = 42
N_SAMPLES = 200
BATCH_SIZE = 64
K_VALUES = (5, 10, 15, 20, 30)
MAIN_SCENARIOS = tuple((n, s) for n in (1, 2, 3) for s in (0.20, 0.30))
STRENGTH_SCENARIOS = tuple(
    (3, s) for s in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    if (3, s) not in MAIN_SCENARIOS
)
WEIGHT = None


def infer(model, attacked_scaled, device, **kwargs):
    residuals, priors = [], []
    for start in range(0, len(attacked_scaled), BATCH_SIZE):
        batch = torch.from_numpy(attacked_scaled[start : start + BATCH_SIZE]).to(device)
        with torch.no_grad():
            output = model(batch, **kwargs)
        residuals.append(output[1].cpu().numpy())
        priors.append(output[4].cpu().numpy())
    return np.concatenate(residuals), np.concatenate(priors)


def normalize_rows(values):
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / (high - low + 1e-8)


def scenario_scores(residual, prior, variant, rng):
    if variant == "ConvAE residual only":
        return residual
    if variant == "TopoLoc prior only":
        return prior
    if variant == "No graph aggregation":
        return residual * ((1 - WEIGHT) + WEIGHT * prior)
    if variant == "No temporal context":
        return residual * ((1 - WEIGHT) + WEIGHT * prior)
    if variant == "No residual context":
        return residual * ((1 - WEIGHT) + WEIGHT * prior)
    if variant == "Shuffled measurement mapping":
        shuffled = prior[:, rng.permutation(prior.shape[1])]
        return residual * ((1 - WEIGHT) + WEIGHT * shuffled)
    if variant == "Shuffled sample prior":
        shuffled = prior[rng.permutation(prior.shape[0])]
        return residual * ((1 - WEIGHT) + WEIGHT * shuffled)
    if variant == "Additive normalized fusion":
        return (1 - WEIGHT) * normalize_rows(residual) + WEIGHT * prior
    if variant == "Multiplicative fusion":
        return residual * ((1 - WEIGHT) + WEIGHT * prior)
    raise ValueError(variant)


def main():
    global WEIGHT
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED + 300)
    device = torch.device(nn_setting["device"])
    case_class = load_case()
    _, _, noise_sigma = define_mea_idx_noise(case_class.case, sys_config["measure_type"])
    scale_values = scaler()
    scale_min = np.asarray(scale_values.min, dtype=np.float64)
    scale_diff = np.asarray(scale_values.max - scale_values.min + 1e-8, dtype=np.float64)
    records = _test_records(rnn_dataset_evl(mode="test", istransform=False), N_SAMPLES)
    model = TopoLocConvAE().to(device)
    model.localizer.load_state_dict(
        torch.load(checkpoint_path("TopoLoc_ConvAE.pt"), map_location=device, weights_only=True)
    )
    model.eval()
    localization = json.loads(
        (OUT / f"topoloc_convae_localization_{CASE_NAME}.json").read_text(encoding="utf-8")
    )
    WEIGHT = float(localization["metadata"]["selected_fusion_weight"])

    variants = (
        "ConvAE residual only",
        "TopoLoc prior only",
        "Multiplicative fusion",
        "No graph aggregation",
        "No temporal context",
        "No residual context",
        "Shuffled measurement mapping",
        "Shuffled sample prior",
        "Additive normalized fusion",
    )
    rows = []
    for ang_no, ang_str in MAIN_SCENARIOS + STRENGTH_SCENARIOS:
        scenario = _build_attack_scenario(
            records,
            _attack_plan(case_class, len(records), ang_no, SEED),
            case_class,
            scale_min,
            scale_diff,
            noise_sigma,
            ang_no,
            ang_str,
        )
        control_kwargs = {
            "No graph aggregation": {"use_graph": False},
            "No temporal context": {"use_temporal": False},
            "No residual context": {"use_residual": False},
        }
        cache = {}
        for variant in variants:
            key = tuple(sorted(control_kwargs.get(variant, {}).items()))
            if key not in cache:
                cache[key] = infer(
                    model, scenario["attacked_scaled"], device, **dict(key)
                )
            residual, prior = cache[key]
            scores = scenario_scores(residual, prior, variant, rng)
            aggregate = _aggregate_metrics(
                [
                    _metrics_for_sample(truth, score, label, K_VALUES)
                    for truth, score, label in zip(
                        scenario["truth"], scores, scenario["labels"]
                    )
                ]
            )
            rows.append(
                {
                    "variant": variant,
                    "ang_no": ang_no,
                    "ang_str": ang_str,
                    "protocol": "main" if (ang_no, ang_str) in MAIN_SCENARIOS else "strength_ang3",
                    **aggregate,
                }
            )

    summary = []
    for variant in variants:
        selected = [
            row for row in rows
            if row["variant"] == variant and row["protocol"] == "main"
        ]
        summary.append(
            {
                "variant": variant,
                "ndcg_15_mean": float(np.mean([row["ndcg_15_mean"] for row in selected])),
                "auprc_mean": float(np.mean([row["auprc_mean"] for row in selected])),
                "overlap_15_mean": float(np.mean([row["overlap_15_mean"] for row in selected])),
                "protocol": "main six scenarios",
            }
        )
    payload = {
        "note": "Fixed-checkpoint inference controls compare topology, temporal, residual, mapping, and fusion settings.",
        "fusion_weight": WEIGHT,
        "summary": summary,
        "scenario_metrics": rows,
    }
    (OUT / f"topoloc_ablation_extended_{CASE_NAME}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (OUT / f"topoloc_ablation_extended_{CASE_NAME}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / f"topoloc_ablation_summary_{CASE_NAME}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(
            f"{row['variant']:30s} NDCG@15={row['ndcg_15_mean']:.6f} "
            f"AUPRC={row['auprc_mean']:.6f} Overlap@15={row['overlap_15_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
