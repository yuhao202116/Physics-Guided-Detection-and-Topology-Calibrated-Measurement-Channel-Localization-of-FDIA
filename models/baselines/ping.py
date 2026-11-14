#!/usr/bin/env python3
"""PING: physics-informed GAT for node-level FDIA localization.

This module is self-contained at the model layer and reuses the project's
existing cases, measurements, configuration, and saved-model layout. Run it
from the project root with::

    python models/baselines/ping.py

Set ``PGAE_CASE`` to select case14, case30, or case57. ``--quick`` performs a
small two-epoch smoke run.
"""

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pypower.api import makeBdc
from pypower.idx_brch import BR_X, F_BUS, T_BUS
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import sys_config
from configs.nn_setting import nn_setting
from data.loaders import load_case
from configs.paths import data_path


DEFAULT_HOURS = 500
DEFAULT_ATTACKS_PER_HOUR = 20
DEFAULT_MISSING_RATE = 0.4
DEFAULT_SEED = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _measurement_layout(idx):
    """Return the offset and physical indices for each measurement type."""
    layout = {}
    offset = 0
    for name, indices in idx.items():
        indices = np.asarray(indices, dtype=np.int64)
        layout[name] = (offset, indices)
        offset += len(indices)
    return layout, offset


class PINGDataset(Dataset):
    """Generate the paper's DC stealth attacks from existing project data.

    Each operating point produces ``attacks_per_hour`` graph samples. Node
    features contain active injection and its availability mask. Directed edge
    features contain active flow, reactance, and the flow availability mask.
    """

    def __init__(
        self,
        mode,
        case_class,
        hours=DEFAULT_HOURS,
        attacks_per_hour=DEFAULT_ATTACKS_PER_HOUR,
        missing_rate_max=DEFAULT_MISSING_RATE,
        seed=DEFAULT_SEED,
    ):
        super().__init__()
        if mode not in {"train", "valid", "test"}:
            raise ValueError("mode must be 'train', 'valid', or 'test'")
        if hours <= 0 or attacks_per_hour <= 0:
            raise ValueError("hours and attacks_per_hour must be positive")
        if not 0.0 <= missing_rate_max <= 1.0:
            raise ValueError("missing_rate_max must be in [0, 1]")

        self.z_all = np.load(data_path("z_noise_summary.npy"), mmap_mode="r")
        self.v_all = np.load(data_path("v_est_summary.npy"), mmap_mode="r")
        self.case_name = sys_config["case_name"]
        self.case_class = case_class
        self.attacks_per_hour = attacks_per_hour
        self.missing_rate_max = missing_rate_max
        self.seed = seed

        case = case_class.case_int
        self.n_bus = case_class.no_bus
        self.n_branch = case_class.no_brh
        self.from_bus = case["branch"][:, F_BUS].astype(np.int64)
        self.to_bus = case["branch"][:, T_BUS].astype(np.int64)
        self.reactance = case["branch"][:, BR_X].astype(np.float64)
        self.reactance_scale = max(np.max(np.abs(self.reactance)), 1e-8)
        self.reference_bus = int(case_class.ref_index[0])

        bbus, bf, _, _ = makeBdc(case["baseMVA"], case["bus"], case["branch"])
        self.bbus = np.asarray(bbus.toarray(), dtype=np.float64)
        self.bf = np.asarray(bf.toarray(), dtype=np.float64)

        self.layout, measurement_count = _measurement_layout(case_class.idx)
        if measurement_count != self.z_all.shape[1]:
            raise ValueError(
                f"Measurement layout has {measurement_count} entries, "
                f"but stored data has {self.z_all.shape[1]}"
            )
        if "pf" not in self.layout or "pi" not in self.layout:
            raise ValueError("PING requires active branch-flow and bus-injection data")

        total_hours = min(hours, len(self.z_all), len(self.v_all))
        train_end = int(total_hours * nn_setting["train_prop"])
        valid_end = int(
            total_hours
            * (nn_setting["train_prop"] + nn_setting["valid_prop"])
        )
        bounds = {
            "train": (0, train_end),
            "valid": (train_end, valid_end),
            "test": (valid_end, total_hours),
        }
        self.start_hour, self.end_hour = bounds[mode]
        if self.end_hour <= self.start_hour:
            raise ValueError(f"The {mode} split is empty for hours={total_hours}")

        edge_index = np.vstack(
            [
                np.concatenate([self.from_bus, self.to_bus]),
                np.concatenate([self.to_bus, self.from_bus]),
            ]
        )
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

    def __len__(self):
        return (self.end_hour - self.start_hour) * self.attacks_per_hour

    def _rng(self, hour, attack_index):
        case_seed = sum(ord(char) for char in self.case_name) * 37
        sample_seed = self.seed + case_seed + hour * 1009 + attack_index * 9176
        return np.random.default_rng(sample_seed)

    def _active_measurements(self, measurement):
        injections = np.zeros(self.n_bus, dtype=np.float64)
        injection_mask = np.zeros(self.n_bus, dtype=np.float64)
        flows = np.zeros(self.n_branch, dtype=np.float64)
        flow_mask = np.zeros(self.n_branch, dtype=np.float64)

        pf_offset, pf_indices = self.layout["pf"]
        pi_offset, pi_indices = self.layout["pi"]
        flows[pf_indices] = measurement[pf_offset:pf_offset + len(pf_indices)]
        injections[pi_indices] = measurement[pi_offset:pi_offset + len(pi_indices)]
        flow_mask[pf_indices] = 1.0
        injection_mask[pi_indices] = 1.0
        return injections, injection_mask, flows, flow_mask

    def __getitem__(self, index):
        hour = self.start_hour + index // self.attacks_per_hour
        attack_index = index % self.attacks_per_hour
        rng = self._rng(hour, attack_index)

        candidates = np.delete(np.arange(self.n_bus), self.reference_bus)
        attack_count = min(
            len(candidates), max(1, int(round(0.5 * self.n_bus)))
        )
        attacked_buses = rng.choice(candidates, size=attack_count, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=attack_count)
        offsets = np.deg2rad(rng.uniform(15.0, 30.0, size=attack_count))
        state_attack = np.zeros(self.n_bus, dtype=np.float64)
        state_attack[attacked_buses] = signs * offsets

        measurement = np.asarray(self.z_all[hour], dtype=np.float64)
        injections, node_mask, flows, edge_mask = self._active_measurements(
            measurement
        )

        # Canonical stealth attack: z_attack = z + Hc. makeBdc accounts for
        # transformer taps, phase shifts, and the case's internal bus ordering.
        injections += self.bbus @ state_attack
        flows += self.bf @ state_attack

        # Randomly drop up to the configured fraction, as in the paper.
        missing_rate = rng.uniform(0.0, self.missing_rate_max)
        missing_nodes = rng.random(self.n_bus) < missing_rate
        missing_edges = rng.random(self.n_branch) < missing_rate
        injections[missing_nodes] = 0.0
        node_mask[missing_nodes] = 0.0
        flows[missing_edges] = 0.0
        edge_mask[missing_edges] = 0.0

        # A graph-local power base avoids topology-dependent magnitude ranges.
        observed_power = np.concatenate(
            [np.abs(injections[node_mask > 0]), np.abs(flows[edge_mask > 0])]
        )
        power_scale = max(np.max(observed_power) if observed_power.size else 0.0, 1e-8)
        injections /= power_scale
        flows /= power_scale

        node_x = np.column_stack([injections, node_mask])
        directed_flow = np.concatenate([flows, -flows])
        directed_x = np.concatenate(
            [self.reactance / self.reactance_scale] * 2
        )
        directed_mask = np.concatenate([edge_mask, edge_mask])
        edge_x = np.column_stack([directed_flow, directed_x, directed_mask])
        labels = np.zeros(self.n_bus, dtype=np.float32)
        labels[attacked_buses] = 1.0

        return Data(
            x=torch.tensor(node_x, dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=torch.tensor(edge_x, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.float32),
        )


class PING(nn.Module):
    """Three-layer edge-aware GAT from the PING paper."""

    def __init__(self, hidden_dim=128, heads=6, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(
            2, hidden_dim, heads=heads, concat=False, edge_dim=3,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=3,
            dropout=dropout,
        )
        self.gat3 = GATConv(
            hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=3,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat3(x, edge_index, edge_attr))
        return self.classifier(x).squeeze(-1)

    @staticmethod
    def _run_epoch(loader, model, device, optimizer=None, grad_clip=1.0):
        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total_nodes = 0
        tp = tn = fp = fn = 0

        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            count = batch.y.numel()
            total_loss += loss.item() * count
            total_nodes += count
            pred = logits.detach() >= 0
            target = batch.y >= 0.5
            tp += torch.logical_and(pred, target).sum().item()
            tn += torch.logical_and(~pred, ~target).sum().item()
            fp += torch.logical_and(pred, ~target).sum().item()
            fn += torch.logical_and(~pred, target).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return {
            "loss": total_loss / max(total_nodes, 1),
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }

    def fit(
        self,
        train_loader,
        valid_loader,
        epochs=None,
        output_path=None,
        device=None,
    ):
        """Train PING using the optimization procedure reported in the paper."""
        device = torch.device(device or nn_setting["device"])
        epochs = int(epochs or nn_setting["epochs"])
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        output_path = Path(
            output_path
            or PROJECT_ROOT / "saved_model" / sys_config["case_name"] / "PING.pt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.to(device)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=nn_setting["lr"], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        patience = nn_setting["patience"]

        print("Start Training PING...")
        print("*" * 60)
        for epoch in range(1, epochs + 1):
            started = time()
            train_result = self._run_epoch(
                train_loader, self, device, optimizer=optimizer
            )
            with torch.no_grad():
                valid_result = self._run_epoch(valid_loader, self, device)
            scheduler.step()

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_result['loss']:.6f} | "
                f"Valid Loss {valid_result['loss']:.6f} | "
                f"Valid Acc {valid_result['accuracy']:.4f} | "
                f"Valid F1 {valid_result['f1']:.4f} | "
                f"{time() - started:.2f}s"
            )

            if valid_result["loss"] < best_loss - nn_setting["delta"]:
                best_loss = valid_result["loss"]
                best_state = copy.deepcopy(self.state_dict())
                torch.save(best_state, output_path)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping after {epoch} epochs")
                    break

        if best_state is None:
            raise RuntimeError("Training completed without a valid checkpoint")
        self.load_state_dict(best_state)
        print(f"PING training complete. Best model: {output_path}")
        return self


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Two-epoch smoke run")
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument(
        "--attacks-per-hour", type=int, default=DEFAULT_ATTACKS_PER_HOUR
    )
    parser.add_argument("--missing-rate", type=float, default=DEFAULT_MISSING_RATE)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.hours = min(args.hours, 40)
        args.attacks_per_hour = min(args.attacks_per_hour, 2)
        args.epochs = min(args.epochs or 2, 2)

    seed_all(args.seed)
    case_class = load_case()
    train_set = PINGDataset(
        "train", case_class, args.hours, args.attacks_per_hour,
        args.missing_rate, args.seed,
    )
    valid_set = PINGDataset(
        "valid", case_class, args.hours, args.attacks_per_hour,
        args.missing_rate, args.seed,
    )
    batch_size = 8 if args.quick else nn_setting["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    print(f"Case: {sys_config['case_name']}")
    print(f"Train graphs: {len(train_set)} | Valid graphs: {len(valid_set)}")
    model = PING()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    model.fit(
        train_loader,
        valid_loader,
        epochs=args.epochs,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
