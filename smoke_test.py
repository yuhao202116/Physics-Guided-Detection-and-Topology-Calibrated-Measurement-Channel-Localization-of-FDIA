"""CPU smoke test with temporary synthetic measurements and checkpoints."""

import argparse
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("case14", "case30", "case57"), default="case14")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pgae-smoke-") as temporary:
        os.environ["PGAE_CASE"] = args.case
        os.environ["PGAE_DEVICE"] = "cpu"
        os.environ["PGAE_DATA_DIR"] = str(Path(temporary) / "data")
        os.environ["PGAE_CHECKPOINT_DIR"] = str(Path(temporary) / "checkpoints")
        os.environ["PGAE_OUTPUT_DIR"] = str(Path(temporary) / "outputs")
        run_check(args.case)
    print(f"PASS: {args.case}; temporary inputs and weights removed.")


def run_check(case_name):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from configs.config import sys_config
    from configs.config_mea_idx import define_mea_idx_noise
    from configs.nn_setting import nn_setting
    from configs.paths import data_path, checkpoint_path
    from data.generation import gen_case
    from physics.attacks import FDI
    from data.dataset import rnn_dataset, rnn_dataset_evl
    from models.detection.pgae_tau import PGAE_tau, init_weights_PI, initialize_flat_start
    from training.early_stopping import EarlyStopping
    from models.localization.topoloc_convae import TopoLocConvAE
    from models.localization.self_supervision import corrupt

    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    case = gen_case(case_name)
    indices, n_features, noise = define_mea_idx_noise(case, sys_config["measure_type"])
    grid = FDI(case, noise, indices, sys_config["fpr"])
    n_bus = grid.no_bus
    voltage = (1 + rng.normal(0, 0.01, (48, n_bus))) * np.exp(
        1j * rng.normal(0, 0.02, (48, n_bus))
    )
    # Independent NumPy AC equations provide a reference for the torch decoder.
    v = voltage.T
    sf = (grid.Cf @ v) * np.conj(grid.Yf @ v)
    st = (grid.Ct @ v) * np.conj(grid.Yt @ v)
    si = v * np.conj(grid.Ybus @ v)
    full = np.concatenate(
        [sf.real, st.real, si.real, np.angle(v), sf.imag, st.imag, si.imag, np.abs(v)]
    )
    reference = (grid.IDX @ full).T
    measurements = reference + rng.normal(size=reference.shape) * noise
    data_path("noise_sigma.npy", required=False).parent.mkdir(parents=True)
    for name, array in (
        ("noise_sigma.npy", noise),
        ("z_noise_summary.npy", measurements),
        ("v_est_summary.npy", voltage),
    ):
        np.save(data_path(name, required=False), array, allow_pickle=False)

    train = rnn_dataset("train", True)
    evaluation = rnn_dataset_evl("test", False)
    assert len(evaluation) == 4
    train_end = int(len(measurements) * nn_setting["train_prop"])
    np.testing.assert_allclose(train.transform.min, measurements[:train_end].min(axis=0))
    np.testing.assert_allclose(train.transform.max, measurements[:train_end].max(axis=0))
    x, _ = next(iter(DataLoader(train, batch_size=2)))
    assert x.shape == (2, 6, n_features)

    detector = PGAE_tau().cpu()
    detector.apply(init_weights_PI)
    initialize_flat_start(detector.encoder, n_bus)
    decoded_reference = detector.decoder.physics_layer(
        torch.tensor(voltage.real, dtype=torch.float32),
        torch.tensor(voltage.imag, dtype=torch.float32),
    )
    np.testing.assert_allclose(
        decoded_reference.detach().numpy(), reference, rtol=1e-4, atol=3e-5
    )
    reconstruction, real, imag = detector(x)
    assert reconstruction.shape == (2, n_features)
    assert real.shape == imag.shape == (2, n_bus)
    lower = torch.tensor(train.transform.min, dtype=torch.float32)
    span = torch.tensor(train.transform.max - train.transform.min + 1e-7, dtype=torch.float32)
    loss = (((reconstruction - lower) / span - x[:, -1]) ** 2).mean()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    tau = detector.encoder.topo_attention.tau
    assert tau.numel() == 1 and tau.grad is not None and torch.isfinite(tau.grad).all()
    assert tau.grad.abs().sum() > 0
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in detector.parameters())
    optimizer.step()
    checkpoint = checkpoint_path("smoke.pt")
    EarlyStopping(path=checkpoint)(float(loss.detach()), detector)
    restored = PGAE_tau().cpu()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    detector.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(detector(x)[0], restored(x)[0])

    localizer = TopoLocConvAE(untrained_backbone=True).cpu()
    corrupted, targets = corrupt(
        x, localizer.localizer.bus_measurement, torch.Generator().manual_seed(11)
    )
    outputs = localizer(corrupted)
    expected = ((2, 6, n_features), (2, n_features), (2, n_bus), (2, n_bus), (2, n_features))
    for output, shape in zip(outputs, expected):
        assert output.shape == shape and torch.isfinite(output).all()
    clean_logits = localizer(x)[2]
    prior_loss = torch.nn.functional.cross_entropy(outputs[2] / 0.25, targets)
    prior_loss += 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(
        clean_logits, torch.zeros_like(clean_logits)
    )
    prior_loss.backward()
    assert all(p.grad is None for p in localizer.backbone.parameters())
    gradients = [p.grad for p in localizer.localizer.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
    print("Checked: data loading, AC decoder, shared tau, checkpoint round trip, frozen-backbone localization.")


if __name__ == "__main__":
    main()
