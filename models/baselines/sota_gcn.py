"""
GCN-based FDIA Detection Classifier
=====================================

Supervised classifier using measurement projection, residual graph convolution,
global mean pooling, and an MLP output head. Training data combine equal numbers
of normal windows and FDI-attacked windows generated with gen_fdi_att_dd().
Author: Hawkyu

[Architecture]
  Input [B, T, F]
    → Flatten [B, T*F]
    → Input MLP → [B, N, d_model]
    → ResGCN×2 → [B, N, hidden]
    → Global Mean Pool → [B, hidden]
    → MLP Classifier → [B, 1] logit → BCEWithLogitsLoss
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from training.early_stopping import EarlyStopping
from time import time
from configs.nn_setting import nn_setting
from configs.paths import data_path, checkpoint_path
from configs.config import sys_config
from data.loaders import load_case, load_measurement, load_load_pv

torch.manual_seed(0)


# =============================================================================
# ResGraphConvLayer: 带残差的归一化图卷积
# =============================================================================

class ResGraphConvLayer(nn.Module):
    """Residual GCN: H' = ReLU(D^(-1/2) Â D^(-1/2) H W + H)"""
    def __init__(self, in_features, out_features, adj_matrix, bias=True):
        super().__init__()

        adj_self = adj_matrix.clone()
        adj_self.fill_diagonal_(1.0)
        degree = adj_self.sum(dim=1)
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = (torch.diag(deg_inv_sqrt) @ adj_self @
                    torch.diag(deg_inv_sqrt))
        self.register_buffer('norm_adj', norm_adj)

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.res_proj = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)

    def forward(self, x):
        support = torch.matmul(x, self.weight)
        gcn_out = torch.matmul(self.norm_adj, support)
        if self.bias is not None:
            gcn_out = gcn_out + self.bias
        residual = self.res_proj(x) if self.res_proj is not None else x
        return F.relu(gcn_out + residual)


# =============================================================================
# FDIADataset: 正常与攻击混合数据集
# =============================================================================
# Angle attacks use ang_no in {1, 2, 3} and ang_str in {0.2, 0.3}.

class FDIADataset(Dataset):
    """
    Balanced dataset for supervised FDIA detection.

    Each sample is (measurement_window [T, F], label):
      - label = 0: normal
      - label = 1: FDI attacked

    Attacks are generated using case_class.gen_fdi_att_dd() with
    varying angle attack parameters.
    """
    def __init__(self, mode, case_class, scaler_=None):
        super().__init__()

        # ---- Config ----
        self.T = nn_setting['sample_length']
        self.F = nn_setting['feature_size']

        # ---- Load raw (unscaled) data for attack generation ----
        z_all = np.load(data_path('z_noise_summary.npy'))
        v_all = np.load(data_path('v_est_summary.npy'))
        load_active = np.load(data_path('load_active.npy'))
        load_reactive = np.load(data_path('load_reactive.npy'))
        pv_active_ = np.load(data_path('pv_active.npy'))
        pv_reactive_ = np.load(data_path('pv_reactive.npy'))

        n_total = len(z_all)
        if mode == 'train':
            start, end = 0, int(n_total * nn_setting['train_prop'])
        elif mode == 'valid':
            start = int(n_total * nn_setting['train_prop'])
            end = int(n_total * (nn_setting['train_prop'] + nn_setting['valid_prop']))
        elif mode == 'test':
            start = int(n_total * (nn_setting['train_prop'] + nn_setting['valid_prop']))
            end = n_total
        else:
            raise ValueError("mode must be 'train', 'valid', or 'test'")

        z_split = z_all[start:end]
        v_split = v_all[start:end]
        load_a = load_active[start:end]
        load_r = load_reactive[start:end]
        pv_a = pv_active_[start:end]
        pv_r = pv_reactive_[start:end]

        # ---- Build normal samples ----
        normal_samples = []
        for i in range(len(z_split) - self.T):
            window = z_split[i:i + self.T]
            window = torch.from_numpy(window).float()
            normal_samples.append(window)

        # ---- Build attack samples ----
        ang_no_list = [1, 2, 3]
        ang_str_list = [0.2, 0.3]
        attack_pool = []
        for ang_no in ang_no_list:
            for ang_str in ang_str_list:
                attack_pool.append((ang_no, 0, ang_str, 0.0))

        attack_samples = []
        rng = np.random.RandomState(42)
        for i in range(len(normal_samples)):
            ang_no, mag_no, ang_str, mag_str = attack_pool[
                rng.randint(0, len(attack_pool))
            ]
            idx = i
            window_raw = z_split[idx:idx + self.T]

            # Generate attack via load_data utilities
            z_att = self._gen_attack(
                case_class, window_raw, v_split, load_a, load_r, pv_a, pv_r,
                idx, ang_no, mag_no, ang_str, mag_str
            )
            window = torch.from_numpy(z_att).float()
            attack_samples.append(window)

        # ---- Combine and scale ----
        all_samples = normal_samples + attack_samples
        labels = [0] * len(normal_samples) + [1] * len(attack_samples)

        # ---- Apply scaler if provided ----
        if scaler_ is not None:
            scaled = []
            for s in all_samples:
                s_np = s.numpy()
                s_scaled = scaler_(s_np)
                scaled.append(torch.from_numpy(s_scaled).float())
            all_samples = scaled

        self.samples = all_samples
        self.labels = labels

    def _gen_attack(self, case_class, z_raw, v_all, load_a, load_r, pv_a, pv_r,
                    idx, ang_no, mag_no, ang_str, mag_str):
        """Generate FDI attack following the same protocol as Detection notebooks."""
        # Build measurement window as tensor
        z_window = torch.from_numpy(z_raw).float().unsqueeze(0)  # [1, T, F]
        # State estimate at last timestep
        v_last = v_all[idx + self.T - 1]
        v_last_tensor = torch.from_numpy(v_last)  # complex, do NOT cast to float

        # Use the same attack interface as the physics-guided detection notebook.
        # Pass the measurement window and last-step voltage to the attack generator.
        z_att, _ = case_class.gen_fdi_att_dd(
            z_noise=z_window,
            v_est_last=v_last_tensor,
            ang_no=ang_no,
            mag_no=mag_no,
            ang_str=ang_str,
            mag_str=mag_str
        )
        return z_att.squeeze(0).numpy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# =============================================================================
# GCN: GCN-based FDIA classifier
# =============================================================================

class GCN(nn.Module):
    """
    GCN Classifier for FDIA Detection.

    Architecture:
      [B,T,F] → flatten → InputMLP → [B,N,d_model]
      → ResGCN×2 → [B,N,hidden] → MeanPool → [B,hidden]
      → ClassifierMLP → [B,1] logit

    Training follows project conventions:
      - Optimizer: Adam
      - Loss: BCEWithLogitsLoss
      - EarlyStopping on valid loss
      - Uniform[-0.2, 0.2] weight init
    """
    def __init__(self):
        super().__init__()
        self.device = nn_setting['device']
        self.n_feature = nn_setting['feature_size']
        self.sample_length = nn_setting['sample_length']
        self.lr = nn_setting['lr']
        self.patience = nn_setting['patience']
        self.delta = nn_setting['delta']
        self.path = str(checkpoint_path('SOTA_GCN.pt'))
        self.epochs = nn_setting['epochs']

        # ---- Topology ----
        se = load_case()
        n_bus = se.no_bus
        A_np = (np.abs(se.Ybus) > 1e-6).astype(np.float32)
        np.fill_diagonal(A_np, 1.0)
        adj = torch.tensor(A_np)

        d_model = 16
        hidden_dim = 32

        # ---- Input projection: [T*F] → [N, d_model] ----
        flat_in = self.sample_length * self.n_feature
        self.input_proj = nn.Sequential(
            nn.Linear(flat_in, n_bus * d_model * 2),
            nn.LayerNorm(n_bus * d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(n_bus * d_model * 2, n_bus * d_model),
            nn.LayerNorm(n_bus * d_model),
            nn.GELU(),
        )

        # ---- Residual GCN stack ----
        self.gcn1 = ResGraphConvLayer(d_model, hidden_dim, adj)
        self.gcn2 = ResGraphConvLayer(hidden_dim, hidden_dim, adj)

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

        self.n_bus = n_bus

    def forward(self, x):
        """
        Args:
            x: [B, T, F] measurement window
        Returns:
            logit: [B] raw classification score (before sigmoid)
        """
        B = x.shape[0]
        h = x.reshape(B, -1)                             # [B, T*F]
        h = self.input_proj(h)                           # [B, N*d_model]
        h = h.view(B, self.n_bus, -1)                    # [B, N, d_model]
        h = self.gcn1(h)                                 # [B, N, hidden]
        h = self.gcn2(h)                                 # [B, N, hidden]
        h = h.mean(dim=1)                                # [B, hidden]
        logit = self.classifier(h).squeeze(-1)            # [B]
        return logit

    def fit(self, train_loader, valid_loader):
        """
        Supervised training for FDIA detection.
        """
        self.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(
            patience=self.patience, verbose=True,
            delta=self.delta, path=self.path
        )

        n_batches_train = len(train_loader)
        n_batches_valid = len(valid_loader)

        self.apply(init_weights)

        print('Start Training GCN FDIA Classifier...')
        print('*' * 60)

        for epoch in range(1, self.epochs + 1):
            t0 = time()

            # ---- Train ----
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for inp, label in train_loader:
                inp = inp.to(self.device)
                label = label.float().to(self.device)

                logit = self(inp)
                loss = criterion(logit, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = (torch.sigmoid(logit) >= 0.5).long()
                train_correct += (pred == label.long()).sum().item()
                train_total += label.size(0)

            train_acc = train_correct / train_total

            # ---- Valid ----
            self.eval()
            valid_loss = 0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for inp, label in valid_loader:
                    inp = inp.to(self.device)
                    label = label.float().to(self.device)

                    logit = self(inp)
                    loss = criterion(logit, label)
                    valid_loss += loss.item()

                    pred = (torch.sigmoid(logit) >= 0.5).long()
                    valid_correct += (pred == label.long()).sum().item()
                    valid_total += label.size(0)

            valid_acc = valid_correct / valid_total

            train_avg = train_loss / n_batches_train
            valid_avg = valid_loss / n_batches_valid

            print(f'Train Loss: {round(train_avg, 7)} | '
                  f'Valid Loss: {round(valid_avg, 7)} | '
                  f'Train Acc: {round(train_acc, 4)} | '
                  f'Valid Acc: {round(valid_acc, 4)}')

            early_stopping(valid_avg, self)
            print(f'Epoch {epoch} time {time() - t0:.2f}s')
            print('*' * 60)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        self.load_state_dict(torch.load(self.path))
        print("GCN Classifier training complete.")


# =============================================================================
# Weight Initialization
# =============================================================================
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == '__main__':
    from data.dataset import scaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # ---- Load case and scaler ----
    case_class = load_case()
    scaler_ = scaler()

    # ---- Build datasets ----
    train_set = FDIADataset(mode='train', case_class=case_class, scaler_=scaler_)
    valid_set = FDIADataset(mode='valid', case_class=case_class, scaler_=scaler_)

    print(f'Train set: {len(train_set)} samples ({sum(train_set.labels)} attacks)')
    print(f'Valid set: {len(valid_set)} samples ({sum(valid_set.labels)} attacks)')

    train_loader = DataLoader(train_set, batch_size=nn_setting['batch_size'],
                              shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=nn_setting['batch_size'],
                              shuffle=False)

    # ---- Train ----
    model = GCN().to(device)
    print(model)
    print(f'Total params: {sum(p.numel() for p in model.parameters())}')

    model.fit(train_loader, valid_loader)
