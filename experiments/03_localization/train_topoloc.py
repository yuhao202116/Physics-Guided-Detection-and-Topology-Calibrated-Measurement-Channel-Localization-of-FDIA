"""Self-supervised training for the TopoLoc-ConvAE node localizer."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.localization.topoloc_convae import TopoLocConvAE
from models.localization.self_supervision import corrupt
from configs.nn_setting import nn_setting
from data.dataset import rnn_dataset
from configs.config import sys_config
from configs.paths import OUTPUT_ROOT, checkpoint_path


OUT = OUTPUT_ROOT / "topoloc" / sys_config["case_name"]
CASE_NAME = sys_config["case_name"]
CHECKPOINT = checkpoint_path("TopoLoc_ConvAE.pt")
SEED = 42
TRAIN_SIZE = 8000
VALID_SIZE = 1600
BATCH_SIZE = 128
MAX_EPOCHS = 20
PATIENCE = 4


def subset(dataset, size, seed):
    indices = np.random.default_rng(seed).choice(
        len(dataset), size=min(size, len(dataset)), replace=False
    )
    return Subset(dataset, indices.tolist())


def run_epoch(model, loader, generator, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "ce": 0.0, "clean_bce": 0.0, "hit1": 0.0, "hit3": 0.0}
    count = 0
    membership = model.localizer.bus_measurement.cpu()
    device = next(model.localizer.parameters()).device
    for batch, _ in loader:
        batch = batch.to(device)
        corrupted, target = corrupt(batch, membership, generator)
        corrupted = corrupted.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            clean_logits = model(batch)[2]
            attacked_logits = model(corrupted)[2]
            ce = F.cross_entropy(attacked_logits / 0.25, target)
            clean_bce = F.binary_cross_entropy_with_logits(
                clean_logits, torch.zeros_like(clean_logits)
            )
            loss = ce + 0.10 * clean_bce
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.localizer.parameters(), 5.0)
                optimizer.step()
        batch_count = batch.shape[0]
        top3 = attacked_logits.topk(3, dim=1).indices
        metrics = {
            "loss": loss,
            "ce": ce,
            "clean_bce": clean_bce,
            "hit1": (attacked_logits.argmax(dim=1) == target).float().mean(),
            "hit3": (top3 == target[:, None]).any(dim=1).float().mean(),
        }
        count += batch_count
        for key, value in metrics.items():
            totals[key] += float(value.detach().cpu()) * batch_count
    return {key: value / count for key, value in totals.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(nn_setting["device"])
    train_data = subset(rnn_dataset("train", True), TRAIN_SIZE, SEED)
    valid_data = subset(rnn_dataset("valid", True), VALID_SIZE, SEED + 1)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    model = TopoLocConvAE().to(device)
    optimizer = torch.optim.AdamW(model.localizer.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    best_loss = float("inf")
    stale = 0
    started = time.time()
    train_generator = torch.Generator().manual_seed(SEED + 10)
    for epoch in range(1, MAX_EPOCHS + 1):
        train = run_epoch(model, train_loader, train_generator, optimizer)
        valid = run_epoch(
            model,
            valid_loader,
            torch.Generator().manual_seed(SEED + 20),
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"valid_{key}": value for key, value in valid.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} train={train['loss']:.5f} valid={valid['loss']:.5f} "
            f"hit1={valid['hit1']:.3f} hit3={valid['hit3']:.3f}"
        )
        if valid["loss"] < best_loss - 1e-6:
            best_loss = valid["loss"]
            stale = 0
            torch.save(model.localizer.state_dict(), CHECKPOINT)
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping")
                break

    with (OUT / f"topoloc_training_history_{CASE_NAME}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (OUT / f"topoloc_training_metadata_{CASE_NAME}.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "training_data": "normal-only with self-supervised bus corruption",
                "train_size": len(train_data),
                "valid_size": len(valid_data),
                "best_valid_loss": best_loss,
                "epochs": len(history),
                "elapsed_seconds": time.time() - started,
                "conv_checkpoint": str(checkpoint_path("ConvAE.pt")),
                "localizer_checkpoint": str(CHECKPOINT),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {CHECKPOINT}")


if __name__ == "__main__":
    main()
