"""TopoLoc-ConvAE: ConvAE residuals calibrated by a node-wise topology prior."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import torch
import torch.nn as nn

from configs.nn_setting import nn_setting
from configs.config import sys_config
from configs.paths import checkpoint_path
from models.baselines.convae import Conv_AE
from data.loaders import load_case


def bus_measurement_matrix(case_class) -> torch.Tensor:
    matrix = torch.zeros(case_class.no_bus, nn_setting["feature_size"])
    cursor = 0
    for measure_type in ("pf", "pt", "pi", "vang", "qf", "qt", "qi", "vmag"):
        for physical_index in case_class.idx.get(measure_type, []):
            if measure_type in {"pi", "qi", "vang", "vmag"}:
                buses = [int(physical_index)]
            else:
                branch = case_class.case["branch"][int(physical_index)]
                buses = [int(branch[0]) - 1, int(branch[1]) - 1]
            for bus in buses:
                matrix[bus, cursor] = 1.0
            cursor += 1
    if cursor != nn_setting["feature_size"]:
        raise RuntimeError(f"Mapped {cursor} measurements; expected {nn_setting['feature_size']}")
    return matrix


class NodeTopologyPrior(nn.Module):
    def __init__(self, n_bus: int, hidden_dim: int = 32):
        super().__init__()
        case_class = load_case()
        membership = bus_measurement_matrix(case_class)
        adjacency = (np.abs(case_class.Ybus) > 1e-6).astype(np.float32)
        np.fill_diagonal(adjacency, 1.0)
        adjacency /= adjacency.sum(axis=1, keepdims=True)
        self.register_buffer("bus_measurement", membership)
        self.register_buffer("adjacency", torch.tensor(adjacency))

        self.local_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.topology_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.node_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.node_head.bias)

    def _pool(self, values: torch.Tensor):
        mask = self.bus_measurement
        count = mask.sum(dim=1).clamp_min(1.0).unsqueeze(0)
        mean = torch.matmul(values, mask.T) / count
        centered = values.unsqueeze(1) - mean.unsqueeze(-1)
        std = ((centered.square() * mask.unsqueeze(0)).sum(dim=-1) / count).sqrt()
        maximum = (values.unsqueeze(1) * mask.unsqueeze(0)).amax(dim=-1)
        return mean, std, maximum

    def forward(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        use_graph: bool = True,
        use_temporal: bool = True,
        use_residual: bool = True,
    ):
        last = x[:, -1, :]
        delta = (last - x[:, -2, :]).abs() if use_temporal else torch.zeros_like(last)
        residual = (
            (last - reconstruction[:, -1, :]).abs()
            if use_residual
            else torch.zeros_like(last)
        )
        value_mean, value_std, _ = self._pool(last)
        delta_mean, delta_std, delta_max = self._pool(delta)
        residual_mean, residual_std, residual_max = self._pool(residual)
        local = torch.stack(
            [
                value_mean,
                value_std,
                delta_mean,
                delta_std,
                delta_max,
                residual_mean,
                residual_std,
                residual_max,
            ],
            dim=-1,
        )
        hidden = self.local_encoder(local)
        neighbor = (
            torch.einsum("ij,bjd->bid", self.adjacency, hidden)
            if use_graph
            else hidden
        )
        global_context = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
        fused = self.topology_fusion(torch.cat([hidden, neighbor, global_context], dim=-1))
        logits = self.node_head(fused).squeeze(-1)
        prior = torch.sigmoid(logits)
        return logits, prior

    def to_measurements(self, node_prior: torch.Tensor):
        membership = self.bus_measurement
        denominator = membership.sum(dim=0).clamp_min(1.0).unsqueeze(0)
        measurement_prior = torch.matmul(node_prior, membership) / denominator
        return measurement_prior / measurement_prior.amax(dim=1, keepdim=True).clamp_min(1e-8)


class TopoLocConvAE(nn.Module):
    def __init__(self, conv_checkpoint: Path | None = None, *, untrained_backbone: bool = False):
        super().__init__()
        if untrained_backbone and conv_checkpoint is not None:
            raise ValueError("Choose either a trained checkpoint or an untrained smoke-test backbone.")
        original_latent = nn_setting["lattent_dim"]
        try:
            if sys_config["case_name"] == "case14":
                nn_setting["lattent_dim"] = 10
            self.backbone = Conv_AE()
        finally:
            nn_setting["lattent_dim"] = original_latent
        if not untrained_backbone:
            checkpoint = conv_checkpoint or checkpoint_path("ConvAE.pt")
            self.backbone.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.localizer = NodeTopologyPrior(load_case().no_bus)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        use_graph: bool = True,
        use_temporal: bool = True,
        use_residual: bool = True,
    ):
        self.backbone.eval()
        with torch.no_grad():
            _, reconstruction = self.backbone(x)
        logits, node_prior = self.localizer(
            x,
            reconstruction,
            use_graph=use_graph,
            use_temporal=use_temporal,
            use_residual=use_residual,
        )
        measurement_prior = self.localizer.to_measurements(node_prior)
        residual = (x[:, -1, :] - reconstruction[:, -1, :]).abs()
        return reconstruction, residual, logits, node_prior, measurement_prior
