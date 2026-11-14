"""
Graph AutoEncoder for FDIA detection.

Reconstruct per-bus active and reactive power injections computed from estimated
voltage and Ybus. The encoder and decoder each use two ChebConv layers.
Filter order, hidden width, and latent width are configured in nn_setting:
  case14: K=2, hidden=16, latent=2
  case30: K=2, hidden=64, latent=5
  case57: K=3, hidden=64, latent=8

ChebConv reference: Defferrard et al. (2016), "Convolutional Neural Networks on
Graphs with Fast Localized Spectral Filtering", NeurIPS.
Author: Hawkyu
"""

import os, sys, numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch.utils.data import Dataset, DataLoader
from training.early_stopping import EarlyStopping
from time import time
from configs.nn_setting import nn_setting
from configs.paths import data_path, checkpoint_path
from configs.config import sys_config
from data.loaders import load_case

torch.manual_seed(0)


# =========================================================================
# BusPQDataset — per-bus [P_inj, Q_inj] from voltage estimates + Ybus
# =========================================================================
# Computes S = V ⊙ conj(Ybus @ V) for each timestep, returning normalized
# [n_bus, 2] tensors as (input, target) for the AE.
# Author: Hawkyu | 2026-06-19
# =========================================================================

class BusPQDataset(Dataset):
    """
    Dataset yielding per-bus [P_inj, Q_inj] features computed from
    state estimation voltage v_est via S = V ⊙ conj(Ybus @ V).

    Returns:
        (pq_norm, pq_norm) — input == target for AE training.
        pq_norm: [n_bus, 2] min-max normalized tensor (float32).
    """
    def __init__(self, mode='train'):
        se = load_case()
        self.n_bus = se.no_bus
        Ybus = torch.tensor(se.Ybus, dtype=torch.cfloat)

        # ---- Load voltage estimates ----
        v_est_all = np.load(data_path('v_est_summary.npy'))
        n_total = len(v_est_all)

        # ---- Partition ----
        n_train = int(n_total * nn_setting['train_prop'])
        n_valid = int(n_total * nn_setting['valid_prop'])
        if mode == 'train':
            self.v_est = v_est_all[:n_train]
        elif mode == 'valid':
            self.v_est = v_est_all[n_train : n_train + n_valid]
        elif mode == 'test':
            self.v_est = v_est_all[n_train + n_valid:]
        else:
            raise ValueError("mode must be 'train', 'valid', or 'test'")

        # ---- Precompute per-bus P/Q for all timesteps ----
        print(f'Computing per-bus P/Q for {mode} set ({len(self.v_est)} samples)...')
        pq_list = []
        for v in self.v_est:
            v_t = torch.tensor(v, dtype=torch.cfloat)
            I = Ybus @ v_t                          # bus injection currents
            S = v_t * torch.conj(I)                 # complex power
            pq = torch.stack([S.real.float(), S.imag.float()], dim=-1)  # [N, 2]
            pq_list.append(pq)
        self.pq = torch.stack(pq_list, dim=0)       # [n_samples, N, 2]

        # ---- Normalization (fit on train set only) ----
        if mode == 'train':
            self.pq_min = self.pq.amin(dim=(0, 1), keepdim=True)
            self.pq_max = self.pq.amax(dim=(0, 1), keepdim=True)
            self.pq_diff = self.pq_max - self.pq_min + 1e-7
            self.pq_norm = (self.pq - self.pq_min) / self.pq_diff
        else:
            # Compute normalization statistics from training-set P/Q injections.
            train_pq = self._load_train_pq()
            self.pq_min = train_pq.amin(dim=(0, 1), keepdim=True)
            self.pq_max = train_pq.amax(dim=(0, 1), keepdim=True)
            self.pq_diff = self.pq_max - self.pq_min + 1e-7
            self.pq_norm = (self.pq - self.pq_min) / self.pq_diff

    def _load_train_pq(self):
        """Compute P/Q for training partition to get normalization stats."""
        v_est_all = np.load(data_path('v_est_summary.npy'))
        n_train = int(len(v_est_all) * nn_setting['train_prop'])
        v_est_train = v_est_all[:n_train]
        se = load_case()
        Ybus = torch.tensor(se.Ybus, dtype=torch.cfloat)
        pq_list = []
        for v in v_est_train:
            v_t = torch.tensor(v, dtype=torch.cfloat)
            I = Ybus @ v_t
            S = v_t * torch.conj(I)
            pq_list.append(torch.stack([S.real.float(), S.imag.float()], dim=-1))
        return torch.stack(pq_list, dim=0)

    @property
    def f_min(self):
        return self.pq_min.squeeze(0)   # [N, 2]

    @property
    def f_max(self):
        return self.pq_max.squeeze(0)   # [N, 2]

    def __len__(self):
        return len(self.v_est)

    def __getitem__(self, idx):
        pq = self.pq_norm[idx]          # [N, 2], normalized
        return pq, pq.clone()           # AE: input == target


class GraphAE(nn.Module):
    """
    ChebConv autoencoder for per-bus P/Q with case-specific dimensions.
    """
    def __init__(self):
        super().__init__()
        self.device = nn_setting['device']
        self.lr = nn_setting['lr']
        self.delta = nn_setting['delta']
        self.path = str(checkpoint_path('GraphAE.pt'))
        self.patience = nn_setting['patience']
        self.epochs = nn_setting['epochs']

        # ---- Build adjacency from Ybus ----
        se = load_case()
        self.n_bus = se.no_bus
        A = (np.abs(se.Ybus) > 1e-6).astype(np.float32)
        np.fill_diagonal(A, 1.0)
        adj = torch.tensor(A)
        self.edge_index = torch.nonzero(adj).t().contiguous()

        # ---- GNN dimensions (from nn_setting, per-case adaptive) ----
        # Author: Hawkyu | 2026-06-19
        # case14: K=2, hidden=16, latent=2
        # case30: K=2, hidden=64, latent=5
        # case57: K=3, hidden=64, latent=8
        in_dim = 2                           # P_inj, Q_inj per bus (fixed)
        K = nn_setting['gnn_K']              # Chebyshev filter order
        h_dim = nn_setting['gnn_hidden']     # hidden dimension
        z_dim = nn_setting['gnn_latent']     # latent dimension

        # Encoder: in_dim → h_dim → z_dim (adaptive per case)
        self.enc_conv1 = ChebConv(in_dim, h_dim, K)
        self.enc_conv2 = ChebConv(h_dim, z_dim, K)
        # Decoder: z_dim → h_dim → in_dim
        self.dec_conv1 = ChebConv(z_dim, h_dim, K)
        self.dec_conv2 = ChebConv(h_dim, in_dim, K)

    def _batched_conv(self, x, conv1, conv2):
        """
        Apply two ChebConv layers on batched graphs.
        x: [B, N, d_in] → flattened internally for ChebConv.
        """
        B, N, _ = x.shape
        xf = x.reshape(B * N, -1)
        offsets = torch.arange(B, device=x.device) * N
        ei = torch.cat([self.edge_index + off for off in offsets], dim=1)
        xf = conv1(xf, ei)
        xf = F.relu(xf)
        xf = conv2(xf, ei)
        return xf.view(B, N, -1)

    def forward(self, x):
        """
        x: [B, N, 2] — normalized per-bus [P, Q]
        Returns: (recon [B, N, 2], latent [B, N, gnn_latent])
        """
        z = self._batched_conv(x, self.enc_conv1, self.enc_conv2)
        out = self._batched_conv(z, self.dec_conv1, self.dec_conv2)
        return out, z

    def fit(self, train_loader, valid_loader):
        self.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        es = EarlyStopping(patience=self.patience, verbose=True,
                           delta=self.delta, path=self.path)
        n_tr, n_va = len(train_loader), len(valid_loader)
        print('Start training GraphAE...')
        print('*' * 60)
        for epoch in range(1, self.epochs + 1):
            t0 = time()
            self.train()
            tr_loss = 0
            for inp, tgt in train_loader:
                inp, tgt = inp.to(self.device), tgt.to(self.device)
                recon, _ = self(inp)
                loss = criterion(recon, tgt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            self.eval()
            va_loss = 0
            with torch.no_grad():
                for inp, tgt in valid_loader:
                    inp, tgt = inp.to(self.device), tgt.to(self.device)
                    recon, _ = self(inp)
                    va_loss += criterion(recon, tgt).item()
            ta, va = tr_loss / n_tr, va_loss / n_va
            print(f'Train Loss: {round(ta, 7)} | Valid Loss: {round(va, 7)}')
            es(ta, self)
            print(f'Epoch {epoch} time {time() - t0:.2f}s')
            print('*' * 60)
            if es.early_stop:
                print('Early stopping')
                break
        self.load_state_dict(torch.load(self.path))
        print('GraphAE training complete.')


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device')

    model = GraphAE().to(device)
    print(model)
    print(f'Parameters: {sum(p.numel() for p in model.parameters())}')

    train_set = BusPQDataset(mode='train')
    train_loader = DataLoader(train_set, batch_size=nn_setting['batch_size'],
                              shuffle=True)
    valid_set = BusPQDataset(mode='valid')
    valid_loader = DataLoader(valid_set, batch_size=nn_setting['batch_size'],
                              shuffle=False)

    model.fit(train_loader, valid_loader)
