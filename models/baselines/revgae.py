#!/usr/bin/env python3
"""Recursive variational graph autoencoder for unsupervised FDIA detection.

The implementation follows Wang et al., "Locational False Data Injection
Attack Detection in Smart Grid Using Recursive Variational Graph
Autoencoder" (IEEE Internet of Things Journal, 2025). It reuses the project
measurement windows and neural-network settings. Run from the project root:

    PGAE_CASE=case14 python models/baselines/revgae.py
"""

import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pypower.idx_brch import F_BUS, T_BUS
from torch.utils.data import DataLoader


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import sys_config
from configs.nn_setting import nn_setting
from data.dataset import rnn_dataset
from training.early_stopping import EarlyStopping
from data.loaders import load_case


MEASUREMENT_TYPES = ("pf", "pt", "pi", "vang", "qf", "qt", "qi", "vmag")
BUS_MEASUREMENTS = {"pi", "qi", "vang", "vmag"}
BRANCH_MEASUREMENTS = {"pf", "pt", "qf", "qt"}


def build_measurement_adjacency(case_class):
    """Build the paper's bus/branch relationship graph for all measurements.

    Each available measurement channel is a graph node. Channels attached to
    buses are connected when their buses coincide or are physically adjacent.
    Bus and branch channels are connected by incidence, while branch channels
    are connected when their branches coincide or share a bus.
    """
    branches = case_class.case_int["branch"]
    endpoints = branches[:, [F_BUS, T_BUS]].astype(np.int64)

    nodes = []
    for measurement_type in MEASUREMENT_TYPES:
        indices = np.asarray(case_class.idx.get(measurement_type, []), dtype=np.int64)
        if measurement_type in BUS_MEASUREMENTS:
            nodes.extend(("bus", int(index)) for index in indices)
        elif measurement_type in BRANCH_MEASUREMENTS:
            nodes.extend(("branch", int(index)) for index in indices)

    expected_features = nn_setting["feature_size"]
    if len(nodes) != expected_features:
        raise ValueError(
            f"Measurement graph has {len(nodes)} nodes, but nn_setting expects "
            f"{expected_features} features"
        )

    bus_neighbors = set()
    for from_bus, to_bus in endpoints:
        bus_neighbors.add((int(from_bus), int(to_bus)))
        bus_neighbors.add((int(to_bus), int(from_bus)))

    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for i, (left_type, left_index) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            right_type, right_index = nodes[j]

            if left_type == "bus" and right_type == "bus":
                connected = (
                    left_index == right_index
                    or (left_index, right_index) in bus_neighbors
                )
            elif left_type == "branch" and right_type == "branch":
                left_buses = set(endpoints[left_index])
                right_buses = set(endpoints[right_index])
                connected = left_index == right_index or bool(
                    left_buses.intersection(right_buses)
                )
            else:
                if left_type == "bus":
                    bus_index, branch_index = left_index, right_index
                else:
                    bus_index, branch_index = right_index, left_index
                connected = bus_index in endpoints[branch_index]

            if connected:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    return torch.tensor(adjacency, dtype=torch.float32)


def normalize_adjacency(adjacency):
    """Return D^(-1/2) (A + I) D^(-1/2), as used by the paper."""
    adjacency_with_self_loops = adjacency + torch.eye(
        adjacency.shape[0], dtype=adjacency.dtype
    )
    degree = adjacency_with_self_loops.sum(dim=1).clamp_min(1.0)
    degree_inv_sqrt = degree.pow(-0.5)
    return (
        degree_inv_sqrt[:, None]
        * adjacency_with_self_loops
        * degree_inv_sqrt[None, :]
    )


class GraphConvolution(nn.Module):
    """Dense GCN layer matching the normalized propagation in Eq. (2)."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, normalized_adjacency):
        support = self.linear(x)
        return torch.einsum("ij,bjf->bif", normalized_adjacency, support)


class ReVGAE(nn.Module):
    """Two-layer recursive VGAE using measurement channels as graph nodes."""

    def __init__(self):
        super().__init__()
        self.device = nn_setting["device"]
        self.sample_length = nn_setting["sample_length"]
        self.n_feature = nn_setting["feature_size"]
        self.hidden_dim = nn_setting["gnn_hidden"]
        self.latent_dim = nn_setting["lattent_dim"]
        self.lr = nn_setting["lr"]
        self.epochs = nn_setting["epochs"]
        self.patience = nn_setting["patience"]
        self.delta = nn_setting["delta"]
        from configs.paths import checkpoint_path
        self.path = str(checkpoint_path("ReVGAE.pt"))

        adjacency = build_measurement_adjacency(load_case())
        self.register_buffer("adjacency", adjacency)
        self.register_buffer(
            "normalized_adjacency", normalize_adjacency(adjacency)
        )

        # Two GCN layers are the standard depth used to aggregate local graph
        # context without excessive over-smoothing.
        self.encoder_hidden = GraphConvolution(
            self.sample_length, self.hidden_dim
        )
        self.encoder_mu = GraphConvolution(self.hidden_dim, self.latent_dim)
        self.encoder_log_std = GraphConvolution(
            self.hidden_dim, self.latent_dim
        )

        self.decoder_hidden_1 = GraphConvolution(
            self.latent_dim, self.hidden_dim
        )
        self.decoder_hidden_2 = GraphConvolution(
            self.hidden_dim, self.hidden_dim
        )
        self.measurement_head = nn.Linear(
            self.hidden_dim, self.sample_length
        )

        self.alpha = 0.8
        self.beta = 0.1
        self.gamma = 0.1

    def encode(self, measurement):
        hidden = F.relu(
            self.encoder_hidden(measurement, self.normalized_adjacency)
        )
        mu = self.encoder_mu(hidden, self.normalized_adjacency)
        log_std = self.encoder_log_std(hidden, self.normalized_adjacency)
        return mu, log_std.clamp(min=-10.0, max=10.0)

    def reparameterize(self, mu, log_std):
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(log_std)

    def decode_measurement(self, latent):
        hidden = F.relu(
            self.decoder_hidden_1(latent, self.normalized_adjacency)
        )
        hidden = F.relu(
            self.decoder_hidden_2(hidden, self.normalized_adjacency)
        )
        return self.measurement_head(hidden)

    @staticmethod
    def decode_adjacency_logits(latent):
        return torch.matmul(latent, latent.transpose(1, 2))

    def _vgae_pass(self, measurement):
        mu, log_std = self.encode(measurement)
        latent = self.reparameterize(mu, log_std)
        reconstruction = self.decode_measurement(latent)
        adjacency_logits = self.decode_adjacency_logits(latent)
        return reconstruction, mu, log_std, adjacency_logits

    def forward(self, x):
        """Recursively reconstruct a batch of ``[B, T, F]`` windows."""
        if x.ndim != 3:
            raise ValueError("ReVGAE expects input with shape [batch, time, feature]")
        if x.shape[1:] != (self.sample_length, self.n_feature):
            raise ValueError(
                f"Expected [B, {self.sample_length}, {self.n_feature}], "
                f"received {tuple(x.shape)}"
            )

        measurement = x.transpose(1, 2)
        first_reconstruction, _, _, _ = self._vgae_pass(measurement)
        reconstruction, mu, log_std, adjacency_logits = self._vgae_pass(
            first_reconstruction
        )
        return (
            reconstruction.transpose(1, 2),
            mu,
            log_std,
            adjacency_logits,
            first_reconstruction.transpose(1, 2),
        )

    def loss_function(self, reconstruction, target, mu, log_std, adjacency_logits):
        measurement_loss = F.mse_loss(reconstruction, target)
        adjacency_target = self.adjacency.unsqueeze(0).expand_as(
            adjacency_logits
        )
        adjacency_loss = F.binary_cross_entropy_with_logits(
            adjacency_logits, adjacency_target
        )
        kl_loss = 0.5 * torch.mean(
            mu.square() + torch.exp(2.0 * log_std) - 1.0 - 2.0 * log_std
        )
        total_loss = (
            self.alpha * measurement_loss
            + self.beta * adjacency_loss
            + self.gamma * kl_loss
        )
        return total_loss, measurement_loss, adjacency_loss, kl_loss

    @torch.no_grad()
    def anomaly_score(self, x, reduction="node"):
        """Calculate squared reconstruction errors for inference.

        ``node`` returns one score per measurement channel, ``sample`` returns
        one score per window, and ``none`` returns every temporal error.
        """
        self.eval()
        reconstruction, _, _, _, _ = self(x.to(self.device))
        error = (reconstruction - x.to(self.device)).square()
        if reduction == "node":
            return error.mean(dim=1)
        if reduction == "sample":
            return error.mean(dim=(1, 2))
        if reduction == "none":
            return error
        raise ValueError("reduction must be 'node', 'sample', or 'none'")

    def _run_epoch(self, loader, optimizer=None):
        training = optimizer is not None
        self.train(training)
        totals = np.zeros(4, dtype=np.float64)

        for input_data, target in loader:
            input_data = input_data.to(self.device)
            target = target.to(self.device)
            reconstruction, mu, log_std, adjacency_logits, _ = self(input_data)
            losses = self.loss_function(
                reconstruction, target, mu, log_std, adjacency_logits
            )

            if training:
                optimizer.zero_grad()
                losses[0].backward()
                optimizer.step()

            totals += np.asarray([loss.detach().item() for loss in losses])

        if len(loader) == 0:
            raise ValueError("ReVGAE received an empty data loader")
        return totals / len(loader)

    def fit(self, train_loader, valid_loader):
        self.to(self.device)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.apply(init_weights)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            delta=self.delta,
            path=self.path,
        )

        print("Start Training ReVGAE...")
        print("*" * 60)
        for epoch in range(1, self.epochs + 1):
            started = time()
            train_losses = self._run_epoch(train_loader, optimizer=optimizer)
            with torch.no_grad():
                valid_losses = self._run_epoch(valid_loader)

            print(
                f"Epoch {epoch} | Time: {time() - started:.2f}s\n"
                f"Train Loss: {train_losses[0]:.7f} | "
                f"Valid Loss: {valid_losses[0]:.7f}\n"
                f"Valid Measurement: {valid_losses[1]:.7f} | "
                f"Adjacency: {valid_losses[2]:.7f} | "
                f"KL: {valid_losses[3]:.7f}"
            )
            print("-" * 60)

            early_stopping(valid_losses[0], self)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        self.load_state_dict(torch.load(self.path, map_location=self.device))
        print(f"ReVGAE training complete. Best model: {self.path}")
        return self


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = nn_setting["device"]
    print(f"Using {device} device")
    print(f"Case: {sys_config['case_name']}")

    model = ReVGAE().to(device)
    print(model)
    print(f"The total number of parameters is {sum(p.numel() for p in model.parameters())}")
    print(
        "The total trainable number of parameters is "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    train_set = rnn_dataset(mode="train", istransform=True)
    train_loader = DataLoader(
        train_set,
        batch_size=nn_setting["batch_size"],
        shuffle=True,
    )
    valid_set = rnn_dataset(mode="valid", istransform=True)
    valid_loader = DataLoader(
        valid_set,
        batch_size=nn_setting["batch_size"],
        shuffle=False,
    )

    print(f"Train set size: {len(train_set)}")
    print(f"Valid set size: {len(valid_set)}")
    model.fit(train_loader, valid_loader)


if __name__ == "__main__":
    main()
