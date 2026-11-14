"""Shared measurement-channel localization benchmark."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# PYPOWER still imports numpy.in1d, removed in newer NumPy releases.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from configs.config import sys_config
from configs.paths import CHECKPOINT_ROOT
from configs.config_mea_idx import define_mea_idx_noise
from configs.nn_setting import nn_setting
from models.baselines.anomaly_transformer import AnomTrans
from models.baselines.at_gvae import ATGVAE
from models.baselines.convae import Conv_AE
from models.baselines.lstmae import Encoder as LSTMEncoder, LSTM_AE
from models.detection.pgae_tau import PGAE_tau
from models.baselines.piconvae import PIConvAE
from models.baselines.revgae import ReVGAE
from data.dataset import rnn_dataset_evl, scaler
from physics.attacks import FDI
from data.loaders import load_case


DEFAULT_MODEL_REGISTRY = {
    "ConvAE": {
        "class": "models.baselines.convae.Conv_AE",
        "checkpoint": "saved_model/{case}/ConvAE.pt",
        "score_type": "measurement_reconstruction",
    },
    "LSTMAE": {
        "class": "models.baselines.lstmae.LSTM_AE (checkpoint-compatible adapter)",
        "checkpoint": "saved_model/{case}/LSTMAE.pt",
        "score_type": "measurement_reconstruction",
    },
    "AT": {
        "class": "models.baselines.anomaly_transformer.AnomTrans",
        "checkpoint": "saved_model/{case}/AnomTrans.pt",
        "score_type": "measurement_reconstruction",
    },
    "PIConvAE": {
        "class": "models.baselines.piconvae.PIConvAE",
        "checkpoint": "saved_model/{case}/PIConvAE.pt",
        "score_type": "measurement_reconstruction",
    },
    "ReVGAE": {
        "class": "models.baselines.revgae.ReVGAE",
        "checkpoint": "saved_model/{case}/ReVGAE.pt",
        "score_type": "measurement_node_reconstruction",
    },
    "AT-GVAE": {
        "class": "models.baselines.at_gvae.ATGVAE",
        "checkpoint": "saved_model/{case}/AT_GVAE.pt",
        "score_type": "measurement_node_reconstruction",
    },
    "PGAE-tau": {
        "class": "models.detection.pgae_tau.PGAE_tau",
        "checkpoint": "saved_model/{case}/PGAE_tau.pt",
        "score_type": "physics_residual",
    },
}


class CompatibleLSTMDecoder(nn.Module):
    """Decoder layout used by the existing case57 LSTMAE checkpoint."""

    def __init__(self, latent_dim: int, feature_size: int, sample_length: int, no_layer: int):
        super().__init__()
        layers = []
        input_size = latent_dim
        hidden_size = 64
        for _ in range(no_layer):
            layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
            input_size = hidden_size
        self.LSTM_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_size, feature_size)
        self.sample_length = sample_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, self.sample_length, 1)
        for layer in self.LSTM_layers:
            x, _ = layer(x)
        return self.output_layer(x)


class CompatibleLSTMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder()
        self.decoder = CompatibleLSTMDecoder(
            latent_dim=nn_setting["lattent_dim"],
            feature_size=nn_setting["feature_size"],
            sample_length=nn_setting["sample_length"],
            no_layer=nn_setting["no_layer"],
        )

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)


def _checkpoint_compatible_piconvae():
    """Load the case14 PIConv checkpoint layout with its latent width."""
    if sys_config["case_name"] != "case14":
        return PIConvAE()
    original = nn_setting["lattent_dim"]
    try:
        nn_setting["lattent_dim"] = 48
        return PIConvAE()
    finally:
        nn_setting["lattent_dim"] = original


def _checkpoint_compatible_convae():
    """Load the case14 ConvAE checkpoint layout with its latent width."""
    if sys_config["case_name"] != "case14":
        return Conv_AE()
    original = nn_setting["lattent_dim"]
    try:
        nn_setting["lattent_dim"] = 10
        return Conv_AE()
    finally:
        nn_setting["lattent_dim"] = original


@dataclass
class ModelAdapter:
    name: str
    model: nn.Module
    output_kind: str

    def score(
        self,
        attacked_scaled: torch.Tensor,
        attacked_raw_last: torch.Tensor,
        scale_min: torch.Tensor,
        scale_diff: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(attacked_scaled)
        if self.output_kind == "sequence_scaled":
            recon_scaled = output[1][:, -1, :]
        elif self.output_kind == "at_scaled":
            recon_scaled = output[0][:, -1, :]
        elif self.output_kind == "sequence_raw":
            recon_scaled = (output[0][:, -1, :] - scale_min) / scale_diff
        elif self.output_kind == "physics_raw":
            recon_scaled = (output[0] - scale_min) / scale_diff
        elif self.output_kind == "node_sequence_scaled":
            attacked_scaled_last = (attacked_raw_last - scale_min) / scale_diff
            return (attacked_scaled_last - output[0][:, -1, :]).square()
        elif self.output_kind == "atgvae_scaled":
            attacked_scaled_last = (attacked_raw_last - scale_min) / scale_diff
            generated = output[0][:, -1, :]
            recognized = output[1][:, -1, :]
            return (
                self.model.generator_score_weight
                * (attacked_scaled_last - generated).square()
                + self.model.discriminator_score_weight
                * (attacked_scaled_last - recognized).square()
            )
        else:
            raise ValueError(f"Unsupported output kind: {self.output_kind}")
        attacked_scaled_last = (attacked_raw_last - scale_min) / scale_diff
        return torch.abs(attacked_scaled_last - recon_scaled)


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    try:
        value = fn(x, y).statistic
    except (ValueError, FloatingPointError):
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def _ndcg_at_k(relevance: np.ndarray, score: np.ndarray, k: int) -> float:
    k = min(k, relevance.size)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    predicted = np.argsort(-score, kind="stable")[:k]
    ideal = np.argsort(-relevance, kind="stable")[:k]
    dcg = float(np.sum(relevance[predicted] * discounts))
    idcg = float(np.sum(relevance[ideal] * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def _metrics_for_sample(
    truth: np.ndarray,
    score: np.ndarray,
    labels: np.ndarray,
    k_values: Sequence[int],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for k in k_values:
        kk = min(k, truth.size)
        pred_top = set(np.argsort(-score, kind="stable")[:kk])
        true_top = set(np.argsort(-truth, kind="stable")[:kk])
        result[f"overlap_{k}"] = len(pred_top & true_top) / kk
        result[f"ndcg_{k}"] = _ndcg_at_k(truth, score, kk)
    result["pearson"] = _safe_corr(pearsonr, truth, score)
    result["spearman"] = _safe_corr(spearmanr, truth, score)
    if labels.min() != labels.max():
        result["auprc"] = float(average_precision_score(labels, score))
        result["auroc"] = float(roc_auc_score(labels, score))
    else:
        result["auprc"] = float("nan")
        result["auroc"] = float("nan")
    return result


def _aggregate_metrics(per_sample: List[Dict[str, float]]) -> Dict[str, float]:
    keys = per_sample[0].keys()
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in per_sample], dtype=float)
        finite = values[np.isfinite(values)]
        result[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
        result[f"{key}_std"] = float(finite.std()) if finite.size else float("nan")
    return result


def _measurement_labels(case_class) -> List[str]:
    type_order = ["pf", "pt", "pi", "vang", "qf", "qt", "qi", "vmag"]
    labels: List[str] = []
    for measure_type in type_order:
        indices = list(case_class.idx.get(measure_type, []))
        for local_index, physical_index in enumerate(indices):
            if measure_type in ("pi", "qi", "vang", "vmag"):
                labels.append(f"{measure_type}_bus_{int(physical_index) + 1}")
            elif measure_type in ("pf", "qf", "pt", "qt"):
                branch = case_class.case["branch"][int(physical_index)]
                labels.append(
                    f"{measure_type}_{int(branch[0])}_{int(branch[1])}"
                )
            else:
                labels.append(f"{measure_type}_{local_index}")
    return labels


def _load_models(
    project_root: Path,
    registry: Mapping[str, Mapping[str, object]],
    device: torch.device,
) -> Tuple[Dict[str, ModelAdapter], List[Dict[str, object]]]:
    lstm_constructor = LSTM_AE if sys_config["case_name"] in {"case14", "case30"} else CompatibleLSTMAE
    constructors = {
        "ConvAE": (_checkpoint_compatible_convae, "sequence_scaled"),
        "LSTMAE": (lstm_constructor, "sequence_scaled"),
        "AT": (AnomTrans, "at_scaled"),
        "PIConvAE": (_checkpoint_compatible_piconvae, "sequence_raw"),
        "ReVGAE": (ReVGAE, "node_sequence_scaled"),
        "AT-GVAE": (ATGVAE, "atgvae_scaled"),
        "PGAE-tau": (PGAE_tau, "physics_raw"),
    }
    models: Dict[str, ModelAdapter] = {}
    status: List[Dict[str, object]] = []
    nn_setting["anomtrans_model_path"] = str(
        CHECKPOINT_ROOT / sys_config['case_name'] / "AnomTrans.pt"
    )
    for name, spec in registry.items():
        checkpoint_rel = str(spec["checkpoint"]).format(case=sys_config["case_name"])
        relative = Path(checkpoint_rel)
        checkpoint = (
            CHECKPOINT_ROOT.joinpath(*relative.parts[1:])
            if relative.parts and relative.parts[0] == "saved_model"
            else project_root / relative
        )
        row = {
            "model": name,
            "class": spec["class"],
            "checkpoint": str(checkpoint),
            "score_type": spec["score_type"],
        }
        if spec.get("available") is False:
            row.update(status="SKIPPED", reason=spec.get("reason", "Unavailable"))
            status.append(row)
            continue
        if name not in constructors:
            row.update(status="SKIPPED", reason="No localization adapter is registered.")
            status.append(row)
            continue
        if not checkpoint.exists():
            row.update(status="SKIPPED", reason="Checkpoint does not exist.")
            status.append(row)
            continue
        constructor, output_kind = constructors[name]
        try:
            model = constructor().to(device)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            models[name] = ModelAdapter(name, model, output_kind)
            row.update(status="LOADED", reason="")
        except Exception as exc:  # preserve the exact reason in the workbook
            if name == "AT" and "size mismatch" in str(exc):
                reason = (
                    "Checkpoint uses 142 measurement features; case57 requires "
                    f"{nn_setting['feature_size']}. A compatible checkpoint is required."
                )
            else:
                reason = f"{type(exc).__name__}: {exc}"
            row.update(status="SKIPPED", reason=reason)
        status.append(row)
    return models, status


def _test_records(dataset, n_samples: int):
    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    records = []
    for index in range(min(n_samples, len(dataset))):
        _, window, _, current_state = dataset[index]
        records.append((as_numpy(window), as_numpy(current_state).reshape(-1)))
    return records


def _attack_plan(case_class, n_samples: int, ang_no: int, seed: int):
    py_rng = random.Random(seed + 1009 * ang_no)
    np_rng = np.random.default_rng(seed + 7919 * ang_no)
    plans = []
    for _ in range(n_samples):
        positions = py_rng.sample(case_class.non_ref_index, ang_no)
        signs = -1 + 2 * np_rng.integers(0, 2, size=ang_no)
        unit = np_rng.random(ang_no)
        plans.append((positions, signs, unit))
    return plans


def _build_attack_scenario(
    records,
    plans,
    case_class,
    scale_min: np.ndarray,
    scale_diff: np.ndarray,
    noise_sigma: np.ndarray,
    ang_no: int,
    ang_str: float,
):
    attacked_windows = []
    attacked_last = []
    truth = []
    labels = []
    bus_positions = []
    for (window, state), (positions, signs, unit) in zip(records, plans):
        clean_last = window[-1].astype(float)
        voltage_angle = np.angle(state)
        voltage_magnitude = np.abs(state)
        attacked_angle = voltage_angle.copy()
        ratio = (ang_str - 0.1 + 0.1 * unit) * signs
        attacked_angle[positions] = voltage_angle[positions] * (1.0 + ratio)
        attacked_state = voltage_magnitude * np.exp(1j * attacked_angle)
        attack_vector = (
            case_class.h_x_pypower(attacked_state)
            - case_class.h_x_pypower(state)
        ).reshape(-1)
        z_attacked = clean_last + attack_vector
        attacked_window = window.astype(float).copy()
        attacked_window[-1] = z_attacked
        attacked_windows.append((attacked_window - scale_min) / scale_diff)
        attacked_last.append(z_attacked)
        truth.append(np.abs(attack_vector) / scale_diff)
        labels.append((np.abs(attack_vector) / (noise_sigma + 1e-12) >= 1.0).astype(int))
        bus_positions.append([int(position) + 1 for position in positions])
    return {
        "attacked_scaled": np.asarray(attacked_windows, dtype=np.float32),
        "attacked_raw_last": np.asarray(attacked_last, dtype=np.float32),
        "truth": np.asarray(truth, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int8),
        "bus_positions": bus_positions,
        "ang_no": ang_no,
        "ang_str": ang_str,
    }


def _infer_scores(
    adapter: ModelAdapter,
    scenario,
    device: torch.device,
    scale_min: torch.Tensor,
    scale_diff: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    scores = []
    attacked_scaled = scenario["attacked_scaled"]
    attacked_raw_last = scenario["attacked_raw_last"]
    for start in range(0, len(attacked_scaled), batch_size):
        stop = min(start + batch_size, len(attacked_scaled))
        scaled_batch = torch.from_numpy(attacked_scaled[start:stop]).to(device)
        raw_last_batch = torch.from_numpy(attacked_raw_last[start:stop]).to(device)
        batch_score = adapter.score(
            scaled_batch, raw_last_batch, scale_min, scale_diff
        )
        scores.append(batch_score.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def run_localization_benchmark(
    project_root: str,
    output_json: str,
    model_registry: Optional[Mapping[str, Mapping[str, object]]] = None,
    n_samples: int = 200,
    seed: int = 42,
    batch_size: int = 32,
    k_values: Sequence[int] = (5, 10, 15, 20, 30),
    main_ang_no: Sequence[int] = (1, 2, 3),
    main_ang_str: Sequence[float] = (0.2, 0.3),
    strength_values: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
):
    project_path = Path(project_root).resolve()
    registry = dict(model_registry or DEFAULT_MODEL_REGISTRY)
    device = torch.device(nn_setting["device"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    case_class = load_case()
    idx, _, noise_sigma = define_mea_idx_noise(
        case_class.case, sys_config["measure_type"]
    )
    _ = FDI(case_class.case, noise_sigma, idx, sys_config["fpr"])
    scale = scaler()
    scale_min_np = np.asarray(scale.min, dtype=np.float64)
    scale_diff_np = np.asarray(scale.max - scale.min + 1e-8, dtype=np.float64)
    scale_min = torch.tensor(scale_min_np, dtype=torch.float32, device=device)
    scale_diff = torch.tensor(scale_diff_np, dtype=torch.float32, device=device)
    measurement_labels = _measurement_labels(case_class)

    dataset = rnn_dataset_evl(mode="test", istransform=False)
    records = _test_records(dataset, n_samples)
    models, model_status = _load_models(project_path, registry, device)
    if not models:
        raise RuntimeError("No localization model could be loaded.")

    scenario_keys = sorted(
        {
            (int(ang_no), float(ang_str))
            for ang_no in main_ang_no
            for ang_str in list(main_ang_str) + list(strength_values)
        }
    )
    plans = {
        ang_no: _attack_plan(case_class, len(records), ang_no, seed)
        for ang_no in main_ang_no
    }
    scenarios = {
        (ang_no, ang_str): _build_attack_scenario(
            records,
            plans[ang_no],
            case_class,
            scale_min_np,
            scale_diff_np,
            noise_sigma,
            ang_no,
            ang_str,
        )
        for ang_no, ang_str in scenario_keys
    }

    long_rows = []
    sample_rows = []
    score_cache: Dict[Tuple[str, int, float], np.ndarray] = {}
    total_jobs = len(models) * len(scenarios)
    job = 0
    for model_name, adapter in models.items():
        for (ang_no, ang_str), scenario in scenarios.items():
            job += 1
            print(
                f"[{job:03d}/{total_jobs:03d}] {model_name}: "
                f"ang_no={ang_no}, ang_str={ang_str:.2f}"
            )
            scores = _infer_scores(
                adapter,
                scenario,
                device,
                scale_min,
                scale_diff,
                batch_size,
            )
            score_cache[(model_name, ang_no, ang_str)] = scores
            per_sample = [
                _metrics_for_sample(truth, score, label, k_values)
                for truth, score, label in zip(
                    scenario["truth"], scores, scenario["labels"]
                )
            ]
            metrics = _aggregate_metrics(per_sample)
            long_rows.append(
                {
                    "case": sys_config["case_name"],
                    "model": model_name,
                    "ang_no": ang_no,
                    "ang_str": ang_str,
                    "n_samples": len(records),
                    **metrics,
                }
            )
            if ang_no == 3 and math.isclose(ang_str, 0.30):
                for sample_index, row in enumerate(per_sample):
                    sample_rows.append(
                        {
                            "model": model_name,
                            "sample_index": sample_index,
                            **row,
                        }
                    )

    single_key = (3, 0.30)
    single_scenario = scenarios[single_key]
    single_sample = []
    for measurement_index, label in enumerate(measurement_labels):
        row = {
            "measurement_index": measurement_index,
            "measurement_label": label,
            "true_perturbation": float(single_scenario["truth"][0, measurement_index]),
            "is_significant": int(single_scenario["labels"][0, measurement_index]),
            "attacked_buses": ",".join(
                str(value) for value in single_scenario["bus_positions"][0]
            ),
        }
        for model_name in models:
            row[model_name] = float(
                score_cache[(model_name, single_key[0], single_key[1])][
                    0, measurement_index
                ]
            )
        single_sample.append(row)

    payload = {
        "metadata": {
            "case": sys_config["case_name"],
            "feature_size": nn_setting["feature_size"],
            "sample_length": nn_setting["sample_length"],
            "n_samples": len(records),
            "seed": seed,
            "main_ang_no": list(main_ang_no),
            "main_ang_str": list(main_ang_str),
            "strength_values": list(strength_values),
            "mag_no": 0,
            "mag_str": 0,
            "single_sample_ang_no": 3,
            "single_sample_ang_str": 0.30,
            "score_definition": (
                "last-frame normalized measurement score: absolute residual for "
                "ConvAE/LSTMAE/AT/PIConvAE/PGAE; squared residual for ReVGAE; "
                "0.7 generator + 0.3 recognizer weighted squared residual for AT-GVAE"
            ),
            "binary_label": "abs(attack_vector) / noise_sigma >= 1",
        },
        "model_status": model_status,
        "metrics": long_rows,
        "sample_metrics": sample_rows,
        "single_sample": single_sample,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=True)
    print(f"Saved benchmark JSON: {output_path}")
    return payload
