"""
Anomaly Transformer for FDIA Detection
=======================================
Adapted from: Xu et al., "Anomaly Transformer: Time Series Anomaly Detection
with Association Discrepancy", ICLR 2022.
Original: https://github.com/thuml/Anomaly-Transformer

[Architecture]
  Input [B, S, F] → DataEmbedding → Encoder(AnomalyAttention layers) → Projection → Output [B, S, F]

  Anomaly Attention: prior-association (Gaussian kernel) vs series-association (softmax)
  Anomaly Score = softmax(-discrepancy) ⊙ reconstruction_error

  Author: Hawkyu | 2026-06-20
"""

import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from configs.nn_setting import nn_setting
from configs.paths import data_path, checkpoint_path
from configs.config import sys_config
from training.early_stopping import EarlyStopping
from time import time

torch.manual_seed(0)

# =========================================================================
# Positional Embedding
# =========================================================================
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# =========================================================================
# Token Embedding (Conv1d)
# =========================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x: (B, S, c_in) → (B, d_model, S) → (B, S, d_model)
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

# =========================================================================
# Data Embedding = Token + Position
# =========================================================================
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

# =========================================================================
# Anomaly Attention (core innovation)
#   - prior-association: Gaussian kernel over temporal distances, σ learned
#   - series-association: standard softmax attention
#   - discrepancy = KL(prior || series) used for anomaly scoring
# =========================================================================
class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=True):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # Pre-computed distance matrix
        self.register_buffer('distances', torch.zeros((win_size, win_size)))
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i, j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        # queries/keys/values: (B, L, H, E), sigma: (B, H, L)
        B, L, H, E = queries.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is None:
            # Triangular causal mask
            mask = torch.triu(torch.ones(L, L, device=queries.device, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, -np.inf)
        elif attn_mask is not None:
            scores.masked_fill_(attn_mask, -np.inf)
        attn = scale * scores  # (B, H, L, L)

        # Prior-association: Gaussian kernel N(P_j | P_i; σ_i)
        sigma = sigma.transpose(1, 2)  # (B, H, L)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1  # scale
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, L)  # (B, H, L, L)
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(B, H, 1, 1)  # (B, H, L, L)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        # Series-association
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return V.contiguous(), series, prior, sigma
        return V.contiguous(), None

# =========================================================================
# Attention Layer (multi-head wrapper)
# =========================================================================
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.n_heads = n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)
        sigma = self.sigma_projection(queries.reshape(B, L, -1)).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(queries, keys, values, sigma, None)
        out = out.view(B, L, -1)
        return self.out_projection(out), series, prior, sigma

# =========================================================================
# Encoder Layer
# =========================================================================
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma

# =========================================================================
# Encoder (stack of EncoderLayers)
# =========================================================================
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list, prior_list, sigma_list = [], [], []
        for layer in self.attn_layers:
            x, series, prior, sigma = layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list

# =========================================================================
# Anomaly Transformer — Full Model
# =========================================================================
class AnomTrans(nn.Module):
    """
    Anomaly Transformer for FDIA detection.

    Input:  (B, S, F)  — measurement window
    Output: (B, S, F)  — reconstruction

    Anomaly Score = softmax(-assoc_discrepancy) ⊙ recon_error (per sample)
    """
    def __init__(self):
        super().__init__()
        self.win_size = nn_setting['sample_length']
        enc_in = nn_setting['feature_size']
        c_out = nn_setting['feature_size']
        d_model = nn_setting['anomtrans_d_model']
        n_heads = nn_setting['anomtrans_n_heads']
        e_layers = nn_setting['anomtrans_e_layers']
        d_ff = nn_setting['anomtrans_d_ff']
        dropout = nn_setting['anomtrans_dropout']
        self.k = nn_setting['anomtrans_k']  # Top-k for discrepancy

        self.output_attention = True
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    AnomalyAttention(self.win_size, False, attention_dropout=dropout, output_attention=True),
                    d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation='gelu'
            ) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        # Training
        self.epochs = nn_setting['epochs']
        self.lr = nn_setting['anomtrans_lr']
        self.patience = nn_setting['patience']
        self.delta = nn_setting['delta']
        self.path = str(checkpoint_path('AnomTrans.pt'))
        self.device = nn_setting['device']
        self.temperature = 0.1  # For anomaly score

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        return enc_out, series, prior, sigmas

    def kl_loss(self, p, q):
        """KL divergence between two distributions."""
        p = p + 1e-8
        q = q + 1e-8
        res = p * (torch.log(p) - torch.log(q))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    def association_discrepancy(self, series_list, prior_list):
        """Compute association discrepancy (KL divergence) across layers."""
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior_list)):
            # Normalize prior
            prior_norm = prior_list[u] / (torch.unsqueeze(torch.sum(prior_list[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size) + 1e-8)
            # detach one side for minimax
            series_loss += torch.mean(self.kl_loss(series_list[u], prior_norm.detach()))
            prior_loss += torch.mean(self.kl_loss(prior_norm, series_list[u].detach()))
        return series_loss, prior_loss

    def anomaly_score(self, x):
        """
        Compute per-sample anomaly score.
        Uses the original Anomaly Transformer formula:
          score = softmax(-association_discrepancy) * reconstruction_error (per timestep)
        Returns: (B,) anomaly scores
        """
        self.eval()
        with torch.no_grad():
            output, series, prior, _ = self(x)
            # Reconstruction error per timestep: (B, L)
            recon_err = torch.mean((output - x) ** 2, dim=-1)  # (B, L)

            # Association discrepancy per timestep
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                prior_norm = prior[u] / (torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size) + 1e-8)
                series_loss += self.kl_loss(series[u], prior_norm.detach())
                prior_loss += self.kl_loss(prior_norm, series[u].detach())
            # (B, L) discrepancy
            discrepancy = series_loss + prior_loss

            # Softmax over timesteps: softmax(-discrepancy)
            metric = torch.softmax(-discrepancy * self.temperature, dim=-1)  # (B, L)

            # Per-sample anomaly score: sum of metric * recon_err
            score = torch.sum(metric * recon_err, dim=-1)  # (B,)
        return score  # (B,)

    def fit(self, train_loader, valid_loader):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction='mean')
        es = EarlyStopping(patience=self.patience, verbose=True, delta=self.delta, path=self.path)

        n_train, n_valid = len(train_loader), len(valid_loader)
        print('Start Training Anomaly Transformer...')
        print('*' * 60)

        for epoch in range(1, self.epochs + 1):
            t0 = time()
            self.train()
            train_loss = 0.0
            train_recon = 0.0
            train_disc = 0.0

            for inp, out in train_loader:
                inp = inp.to(self.device)
                output, series, prior, _ = self(inp)

                # Reconstruction loss
                recon_loss = criterion(output, inp)

                # Association discrepancy (minimax)
                series_loss, prior_loss = self.association_discrepancy(series, prior)
                disc_loss = series_loss.mean() - prior_loss.mean()

                loss = recon_loss - disc_loss * 1e-3  # Small weight for discrepancy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_recon += recon_loss.item()
                train_disc += disc_loss.item()

            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inp, out in valid_loader:
                    inp = inp.to(self.device)
                    output, series, prior, _ = self(inp)
                    recon_loss = criterion(output, inp)
                    valid_loss += recon_loss.item()

            tl, vl = train_loss / n_train, valid_loss / n_valid
            print(f'Epoch {epoch:3d} | Train: {tl:.6f} (R:{train_recon/n_train:.6f} D:{train_disc/n_train:.6f}) | Valid: {vl:.6f}')
            es(tl, self)
            print(f'Time: {time()-t0:.1f}s | {"*"*60}')

            if es.early_stop:
                print('Early stopping')
                break

        self.load_state_dict(torch.load(self.path))
        print('Anomaly Transformer training complete.')


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data.dataset import rnn_dataset

    z_noise_summary = np.load(data_path('z_noise_summary.npy'))
    print(f'Data shape: {z_noise_summary.shape}')

    model = AnomTrans()
    model.apply(init_weights)
    print(model)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    train_set = rnn_dataset(mode='train', istransform=True)
    valid_set = rnn_dataset(mode='valid', istransform=True)
    train_loader = DataLoader(train_set, batch_size=nn_setting['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=nn_setting['batch_size'], shuffle=False)

    model.fit(train_loader, valid_loader)
