#!/usr/bin/env python3
"""Adversarially trained generative variational autoencoder for FDIA.

This implementation follows Wang et al., "A locational false data injection
attack detection method in smart grid based on adversarial variational
autoencoders" (Applied Soft Computing, 2024). It uses the project's normal
measurement windows and shared neural-network settings. The adversarial phase
uses reconstruction pretraining, linear warm-up, and a bounded discriminator
objective so the paper's min-max loss remains numerically stable.
"""

import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import sys_config
from configs.nn_setting import nn_setting
from data.dataset import rnn_dataset


class LatentSelfAttention(nn.Module):
    """Single-head scaled dot-product attention from Eq. (10)."""

    def __init__(self, latent_dim):
        super().__init__()
        self.scale = latent_dim**-0.5
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)

    def forward(self, latent):
        query = self.query(latent)
        key = self.key(latent)
        value = self.value(latent)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(1, 2)) * self.scale,
            dim=-1,
        )
        return torch.matmul(weights, value)


class VariationalGRUEncoder(nn.Module):
    """Shared GRU encoder with separate mean and log-standard-deviation GRUs."""

    def __init__(self, feature_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_gru = nn.GRU(
            feature_dim, hidden_dim, batch_first=True
        )
        self.mu_gru = nn.GRU(hidden_dim, latent_dim, batch_first=True)
        self.log_std_gru = nn.GRU(
            hidden_dim, latent_dim, batch_first=True
        )
        self.attention = LatentSelfAttention(latent_dim)

    def forward(self, measurement, sample=True):
        hidden, _ = self.hidden_gru(measurement)
        mu, _ = self.mu_gru(hidden)
        log_std, _ = self.log_std_gru(hidden)
        log_std = log_std.clamp(min=-10.0, max=5.0)

        if sample:
            latent = mu + torch.randn_like(mu) * torch.exp(log_std)
        else:
            latent = mu
        return self.attention(latent), mu, log_std


class GRUDecoder(nn.Module):
    """Two-layer GRU decoder used by both adversarial VAE branches."""

    def __init__(self, latent_dim, hidden_dim, feature_dim):
        super().__init__()
        self.gru = nn.GRU(
            latent_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, latent):
        decoded, _ = self.gru(latent)
        return self.output(decoded)


class ATGVAE(nn.Module):
    """Shared-encoder generative and discriminative variational autoencoder."""

    def __init__(self):
        super().__init__()
        self.device = torch.device(nn_setting["device"])
        self.sample_length = nn_setting["sample_length"]
        self.feature_dim = nn_setting["feature_size"]
        self.hidden_dim = nn_setting["gnn_hidden"]
        self.latent_dim = nn_setting["lattent_dim"]
        self.lr = nn_setting["lr"]
        self.epochs = nn_setting["epochs"]
        self.patience = nn_setting["patience"]
        self.delta = nn_setting["delta"]
        from configs.paths import checkpoint_path
        self.path = str(checkpoint_path("AT_GVAE.pt"))

        self.encoder = VariationalGRUEncoder(
            self.feature_dim, self.hidden_dim, self.latent_dim
        )
        self.generator_decoder = GRUDecoder(
            self.latent_dim, self.hidden_dim, self.feature_dim
        )
        self.discriminator_decoder = GRUDecoder(
            self.latent_dim, self.hidden_dim, self.feature_dim
        )

        self.generator_score_weight = 0.7
        self.discriminator_score_weight = 0.3
        self.gradient_clip = 5.0
        self.pretrain_epochs = 20
        self.adversarial_warmup_epochs = 50
        self.max_adversarial_weight = 0.3
        self.margin_multiplier = 2.0

    def encode(self, measurement):
        return self.encoder(measurement, sample=self.training)

    def forward(self, measurement):
        """Return both reconstructions and the latent variables.

        Args:
            measurement: normalized windows with shape ``[B, T, F]``.

        Returns:
            ``(generated, recognized, discriminated, mu, log_std, z, z_hat,``
            ``mu_hat, log_std_hat)``.
            ``recognized`` is ``VAE_D(VAE_G(X))`` from the paper.
        """
        if measurement.ndim != 3:
            raise ValueError("ATGVAE expects input with shape [batch, time, feature]")
        if measurement.shape[1:] != (
            self.sample_length,
            self.feature_dim,
        ):
            raise ValueError(
                f"Expected [B, {self.sample_length}, {self.feature_dim}], "
                f"received {tuple(measurement.shape)}"
            )

        latent, mu, log_std = self.encode(measurement)
        generated = self.generator_decoder(latent)
        discriminated = self.discriminator_decoder(latent)
        reencoded_latent, reencoded_mu, reencoded_log_std = self.encode(generated)
        recognized = self.discriminator_decoder(reencoded_latent)
        return (
            generated,
            recognized,
            discriminated,
            mu,
            log_std,
            latent,
            reencoded_latent,
            reencoded_mu,
            reencoded_log_std,
        )

    @staticmethod
    def kl_divergence(mu, log_std):
        return 0.5 * torch.mean(
            mu.square() + torch.exp(2.0 * log_std) - 1.0 - 2.0 * log_std
        )

    def vae_loss(self, reconstruction, target, mu, log_std):
        reconstruction_loss = F.mse_loss(reconstruction, target)
        kl_loss = self.kl_divergence(mu, log_std)
        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss

    def _adversarial_weight(self, epoch):
        if epoch <= self.pretrain_epochs:
            return 0.0
        progress = (epoch - self.pretrain_epochs) / float(
            self.adversarial_warmup_epochs
        )
        return self.max_adversarial_weight * min(progress, 1.0)

    def _generator_losses(self, measurement, adversarial_weight):
        (
            generated,
            recognized,
            _,
            mu,
            log_std,
            latent,
            reencoded_latent,
            reencoded_mu,
            reencoded_log_std,
        ) = self(measurement)

        generator_vae, generator_reconstruction, generator_kl = self.vae_loss(
            generated, measurement, mu, log_std
        )
        recognition_vae, recognition_reconstruction, recognition_kl = (
            self.vae_loss(
                recognized,
                measurement,
                reencoded_mu,
                reencoded_log_std,
            )
        )
        latent_loss = F.mse_loss(reencoded_latent, latent)
        adversarial_loss = latent_loss + recognition_vae
        generator_loss = generator_vae + adversarial_weight * adversarial_loss
        return {
            "loss": generator_loss,
            "reconstruction": generator_reconstruction,
            "recognition_reconstruction": recognition_reconstruction,
            "kl": generator_kl,
            "recognition_kl": recognition_kl,
            "latent": latent_loss,
            "adversarial": adversarial_loss,
        }

    def _discriminator_losses(self, measurement, adversarial_weight, margin):
        # The shared encoder and D_G provide fixed real/fake representations
        # during the discriminator update. Only D_D is optimized in this step.
        with torch.no_grad():
            latent, _, _ = self.encode(measurement)
            generated = self.generator_decoder(latent)
            reencoded_latent, _, _ = self.encode(generated)

        discriminated = self.discriminator_decoder(latent.detach())
        recognized = self.discriminator_decoder(reencoded_latent.detach())
        real_reconstruction = F.mse_loss(discriminated, measurement)
        fake_reconstruction = F.mse_loss(recognized, measurement)
        hinge_loss = F.relu(
            fake_reconstruction.new_tensor(margin) - fake_reconstruction
        )
        discriminator_loss = (
            real_reconstruction + adversarial_weight * hinge_loss
        )
        return {
            "loss": discriminator_loss,
            "real_reconstruction": real_reconstruction,
            "fake_reconstruction": fake_reconstruction,
            "hinge": hinge_loss,
        }

    @staticmethod
    def _set_trainable(module, trainable):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def _train_batch(
        self,
        measurement,
        adversarial_weight,
        margin,
        generator_optimizer,
        discriminator_optimizer,
    ):
        self._set_trainable(self.generator_decoder, True)
        self._set_trainable(self.discriminator_decoder, False)
        generator_optimizer.zero_grad()
        generator_losses = self._generator_losses(
            measurement, adversarial_weight
        )
        generator_losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.generator_decoder.parameters()),
            self.gradient_clip,
        )
        generator_optimizer.step()

        self._set_trainable(self.generator_decoder, False)
        self._set_trainable(self.discriminator_decoder, True)
        discriminator_optimizer.zero_grad()
        discriminator_losses = self._discriminator_losses(
            measurement, adversarial_weight, margin
        )
        discriminator_losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.discriminator_decoder.parameters(),
            self.gradient_clip,
        )
        discriminator_optimizer.step()

        self._set_trainable(self.generator_decoder, True)
        return generator_losses, discriminator_losses

    @torch.no_grad()
    def _calibrate_margin(self, loader):
        self.eval()
        fake_errors = []
        for measurement, target in loader:
            measurement = measurement.to(self.device)
            target = target.to(self.device)
            generated, recognized, _, _, _, _, _, _, _ = self(measurement)
            fake_errors.append(
                (recognized - target)
                .square()
                .mean(dim=(1, 2))
                .cpu()
            )
        if not fake_errors:
            raise ValueError("ATGVAE received an empty margin loader")
        median_error = torch.cat(fake_errors).median().item()
        return self.margin_multiplier * median_error

    @torch.no_grad()
    def _validation_loss(self, loader):
        self.eval()
        total = 0.0
        for measurement, target in loader:
            measurement = measurement.to(self.device)
            target = target.to(self.device)
            generated, recognized, _, _, _, _, _, _, _ = self(measurement)
            total += (
                self.generator_score_weight * F.mse_loss(generated, target)
                + self.discriminator_score_weight * F.mse_loss(recognized, target)
            ).item()
        if len(loader) == 0:
            raise ValueError("ATGVAE received an empty validation loader")
        return total / len(loader)

    def fit(self, train_loader, valid_loader):
        self.to(self.device)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.apply(init_weights)

        generator_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.generator_decoder.parameters()),
            lr=self.lr,
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator_decoder.parameters(),
            lr=self.lr,
        )

        print("Start Training AT-GVAE...")
        print("*" * 60)
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        margin = 0.0
        for epoch in range(1, self.epochs + 1):
            started = time()
            self.train()
            totals = np.zeros(8, dtype=np.float64)
            adversarial_weight = self._adversarial_weight(epoch)

            for measurement, _ in train_loader:
                measurement = measurement.to(self.device)
                generator_losses, discriminator_losses = self._train_batch(
                    measurement,
                    adversarial_weight,
                    margin,
                    generator_optimizer,
                    discriminator_optimizer,
                )
                totals += np.asarray(
                    [
                        generator_losses["loss"].detach().item(),
                        discriminator_losses["loss"].detach().item(),
                        generator_losses["reconstruction"].detach().item(),
                        discriminator_losses["real_reconstruction"].detach().item(),
                        discriminator_losses["fake_reconstruction"].detach().item(),
                        generator_losses["recognition_reconstruction"].detach().item(),
                        generator_losses["kl"].detach().item(),
                        generator_losses["latent"].detach().item(),
                    ]
                )

            if len(train_loader) == 0:
                raise ValueError("ATGVAE received an empty training loader")
            totals /= len(train_loader)
            validation_loss = self._validation_loss(valid_loader)

            if epoch == self.pretrain_epochs:
                margin = self._calibrate_margin(valid_loader)
                print(f"Calibrated discriminator margin: {margin:.7f}")

            print(
                f"Epoch {epoch} | Time: {time() - started:.2f}s\n"
                f"Generator: {totals[0]:.7f} | Discriminator: {totals[1]:.7f}\n"
                f"Generator Recon: {totals[2]:.7f} | "
                f"Discriminator Real: {totals[3]:.7f}\n"
                f"Discriminator Fake: {totals[4]:.7f} | "
                f"Recognition Recon: {totals[5]:.7f}\n"
                f"Generator KL: {totals[6]:.7f} | "
                f"Latent Consistency: {totals[7]:.7f}\n"
                f"Adversarial Weight: {adversarial_weight:.4f} | "
                f"Margin: {margin:.7f}\n"
                f"Valid Anomaly Reconstruction: {validation_loss:.7f}"
            )
            print("-" * 60)

            if validation_loss < best_validation_loss - self.delta:
                print(
                    f"Validation loss decreased ({best_validation_loss:.6f} "
                    f"--> {validation_loss:.6f}). Saving model ..."
                )
                torch.save(self.state_dict(), self.path)
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            elif epoch > self.pretrain_epochs:
                epochs_without_improvement += 1
                print(
                    "EarlyStopping counter: "
                    f"{epochs_without_improvement} out of {self.patience}"
                )

            if (
                epoch > self.pretrain_epochs
                and epochs_without_improvement >= self.patience
            ):
                print("Early stopping triggered.")
                break

        self.load_state_dict(
            torch.load(self.path, map_location=self.device, weights_only=True)
        )
        print(f"AT-GVAE training complete. Best model: {self.path}")
        return self

    @torch.no_grad()
    def anomaly_score(self, measurement, reduction="node"):
        """Return Eq. (19) weighted squared reconstruction residuals."""
        self.eval()
        measurement = measurement.to(self.device)
        generated, recognized, _, _, _, _, _, _, _ = self(measurement)
        error = (
            self.generator_score_weight * (measurement - generated).square()
            + self.discriminator_score_weight * (measurement - recognized).square()
        )
        if reduction == "node":
            return error.mean(dim=1)
        if reduction == "sample":
            return error.mean(dim=(1, 2))
        if reduction == "none":
            return error
        raise ValueError("reduction must be 'node', 'sample', or 'none'")


def init_weights(module):
    if isinstance(module, nn.GRU):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = nn_setting["device"]
    print(f"Using {device} device")
    print(f"Case: {sys_config['case_name']}")

    model = ATGVAE().to(device)
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
