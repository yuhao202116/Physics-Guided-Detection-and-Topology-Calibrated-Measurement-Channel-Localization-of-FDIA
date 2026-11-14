"""Validation and measurement-channel evaluation for TopoLoc-ConvAE."""

from __future__ import annotations

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
    _measurement_labels,
    _metrics_for_sample,
    _test_records,
)
from data.dataset import rnn_dataset_evl, scaler
from data.loaders import load_case
from configs.paths import OUTPUT_ROOT, checkpoint_path


OUT = OUTPUT_ROOT / "topoloc" / sys_config["case_name"]
CASE_NAME = sys_config["case_name"]
MODEL_NAME = "TopoLoc-ConvAE"
SEED = 42
N_SAMPLES = 200
BATCH_SIZE = 64
K_VALUES = (5, 10, 15, 20, 30)
STRENGTHS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MAIN_SCENARIOS = tuple((n, s) for n in (1, 2, 3) for s in (0.20, 0.30))
WEIGHTS = (0.0, 0.2, 1 / 3, 0.5, 2 / 3, 0.8, 0.9, 0.95, 1.0)


def infer(model, attacked_scaled, device):
    residuals, priors, node_priors = [], [], []
    for start in range(0, len(attacked_scaled), BATCH_SIZE):
        batch = torch.from_numpy(attacked_scaled[start : start + BATCH_SIZE]).to(device)
        with torch.no_grad():
            output = model(batch)
        residuals.append(output[1].cpu().numpy())
        priors.append(output[4].cpu().numpy())
        node_priors.append(output[3].cpu().numpy())
    return np.concatenate(residuals), np.concatenate(priors), np.concatenate(node_priors)


def score_metrics(scenario, scores):
    return [
        _metrics_for_sample(truth, score, binary, K_VALUES)
        for truth, score, binary in zip(scenario["truth"], scores, scenario["labels"])
    ]


def build_scenarios(records, pairs, case_class, scale_min, scale_diff, noise_sigma):
    plans = {
        ang_no: _attack_plan(case_class, len(records), ang_no, SEED)
        for ang_no in {pair[0] for pair in pairs}
    }
    return {
        pair: _build_attack_scenario(
            records,
            plans[pair[0]],
            case_class,
            scale_min,
            scale_diff,
            noise_sigma,
            pair[0],
            pair[1],
        )
        for pair in pairs
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(nn_setting["device"])
    case_class = load_case()
    _, _, noise_sigma = define_mea_idx_noise(case_class.case, sys_config["measure_type"])
    scale_values = scaler()
    scale_min = np.asarray(scale_values.min, dtype=np.float64)
    scale_diff = np.asarray(scale_values.max - scale_values.min + 1e-8, dtype=np.float64)
    model = TopoLocConvAE().to(device)
    model.localizer.load_state_dict(
        torch.load(checkpoint_path("TopoLoc_ConvAE.pt"), map_location=device, weights_only=True)
    )
    model.eval()

    validation_records = _test_records(
        rnn_dataset_evl(mode="valid", istransform=False), N_SAMPLES
    )
    validation_scenarios = build_scenarios(
        validation_records,
        MAIN_SCENARIOS,
        case_class,
        scale_min,
        scale_diff,
        noise_sigma,
    )
    validation_cache = {}
    for pair, scenario in validation_scenarios.items():
        validation_cache[pair] = infer(model, scenario["attacked_scaled"], device)[:2]
    validation_rows = []
    for weight in WEIGHTS:
        values = []
        for pair, scenario in validation_scenarios.items():
            residual, prior = validation_cache[pair]
            fused = residual * ((1.0 - weight) + weight * prior)
            aggregate = _aggregate_metrics(score_metrics(scenario, fused))
            values.append(aggregate)
        validation_rows.append(
            {
                "fusion_weight": weight,
                "ndcg_15_mean": float(np.mean([row["ndcg_15_mean"] for row in values])),
                "auprc_mean": float(np.mean([row["auprc_mean"] for row in values])),
            }
        )
        print(
            f"validation weight={weight:.3f} "
            f"NDCG@15={validation_rows[-1]['ndcg_15_mean']:.6f}"
        )
    selected_weight = max(validation_rows, key=lambda row: row["ndcg_15_mean"])[
        "fusion_weight"
    ]

    test_records = _test_records(rnn_dataset_evl(mode="test", istransform=False), N_SAMPLES)
    all_pairs = tuple((n, s) for n in (1, 2, 3) for s in STRENGTHS)
    test_scenarios = build_scenarios(
        test_records, all_pairs, case_class, scale_min, scale_diff, noise_sigma
    )
    metric_rows, sample_rows, gate_rows = [], [], []
    ablation_scenarios = []
    single_sample = None
    labels = _measurement_labels(case_class)
    for pair, scenario in test_scenarios.items():
        residual, prior, node_prior = infer(model, scenario["attacked_scaled"], device)
        variants = {
            "ConvAE residual only": residual,
            "TopoLoc prior only": prior,
            MODEL_NAME: residual * ((1.0 - selected_weight) + selected_weight * prior),
        }
        variant_metrics = {}
        for variant, scores in variants.items():
            per_sample = score_metrics(scenario, scores)
            aggregate = _aggregate_metrics(per_sample)
            variant_metrics[variant] = (per_sample, aggregate)
            ablation_scenarios.append(
                {"variant": variant, "ang_no": pair[0], "ang_str": pair[1], **aggregate}
            )
        per_sample, aggregate = variant_metrics[MODEL_NAME]
        metric_rows.append(
            {
                "case": sys_config["case_name"],
                "model": MODEL_NAME,
                "ang_no": pair[0],
                "ang_str": pair[1],
                "n_samples": len(test_records),
                **aggregate,
            }
        )
        attacked_hit = []
        for index, attacked_buses in enumerate(scenario["bus_positions"]):
            attacked = {bus - 1 for bus in attacked_buses}
            predicted = set(np.argsort(-node_prior[index])[: pair[0]])
            attacked_hit.append(bool(attacked & predicted))
        gate_rows.append(
            {
                "ang_no": pair[0],
                "ang_str": pair[1],
                "attacked_bus_hit": float(np.mean(attacked_hit)),
            }
        )
        if pair == (3, 0.30):
            sample_rows = [
                {"model": MODEL_NAME, "sample_index": index, **row}
                for index, row in enumerate(per_sample)
            ]
            scores = variants[MODEL_NAME]
            single_sample = [
                {
                    "measurement_index": index,
                    "measurement_label": label,
                    "true_perturbation": float(scenario["truth"][0, index]),
                    "is_significant": int(scenario["labels"][0, index]),
                    "attacked_buses": ",".join(
                        str(value) for value in scenario["bus_positions"][0]
                    ),
                    MODEL_NAME: float(scores[0, index]),
                }
                for index, label in enumerate(labels)
            ]
        print(
            f"{MODEL_NAME}: ({pair[0]}, {pair[1]:.2f}) "
            f"NDCG@15={aggregate['ndcg_15_mean']:.6f} "
            f"AUPRC={aggregate['auprc_mean']:.6f} "
            f"bus_hit={gate_rows[-1]['attacked_bus_hit']:.3f}"
        )

    main_ablation = []
    for variant in ("ConvAE residual only", "TopoLoc prior only", MODEL_NAME):
        rows = [
            row
            for row in ablation_scenarios
            if row["variant"] == variant and (row["ang_no"], row["ang_str"]) in MAIN_SCENARIOS
        ]
        main_ablation.append(
            {
                "variant": variant,
                "ndcg_15_mean": float(np.mean([row["ndcg_15_mean"] for row in rows])),
                "auprc_mean": float(np.mean([row["auprc_mean"] for row in rows])),
                "overlap_15_mean": float(np.mean([row["overlap_15_mean"] for row in rows])),
            }
        )
    payload = {
        "metadata": {
            "model": MODEL_NAME,
            "conv_checkpoint": str(checkpoint_path("ConvAE.pt")),
            "localizer_checkpoint": str(checkpoint_path("TopoLoc_ConvAE.pt")),
            "selected_fusion_weight": selected_weight,
            "training": "normal-only self-supervised bus-aware corruption",
            "score_definition": "ConvAE absolute residual calibrated by node topology prior",
        },
        "model_status": {
            "model": MODEL_NAME,
            "class": "models.localization.topoloc_convae.TopoLocConvAE",
            "checkpoint": str(checkpoint_path("TopoLoc_ConvAE.pt")),
            "score_type": "convae_residual_x_node_topology_prior",
            "status": "LOADED",
            "reason": "",
        },
        "validation_fusion": validation_rows,
        "main_ablation": main_ablation,
        "ablation_scenarios": ablation_scenarios,
        "metrics": metric_rows,
        "sample_metrics": sample_rows,
        "single_sample": single_sample,
        "gate_diagnostics": gate_rows,
    }
    output = OUT / f"topoloc_convae_localization_{CASE_NAME}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selected fusion weight={selected_weight}")
    print(json.dumps(main_ablation, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
