"""PGAE-tau: physics-guided autoencoder with a shared learned temperature."""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
import torch.nn as nn

from training.early_stopping import EarlyStopping
from time import time
from configs.nn_setting import nn_setting
from data.loaders import load_case
torch.manual_seed(0)


class PowerFlowPhysics(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        print("Initializing Physics Layer via SE Instance...")
        se = load_case()

        self.n_bus = se.no_bus
        self.n_branch = se.no_brh
        self.dim_total = self.n_branch * 4 + self.n_bus * 4

        def to_cplx(x): return torch.tensor(x, device=device, dtype=torch.cfloat)
        def to_real(x): return torch.tensor(x, device=device, dtype=torch.float32)

        self.register_buffer('Ybus', to_cplx(se.Ybus))
        self.register_buffer('Yf', to_cplx(se.Yf))
        self.register_buffer('Yt', to_cplx(se.Yt))
        self.register_buffer('Cf', to_cplx(se.Cf))
        self.register_buffer('Ct', to_cplx(se.Ct))
        self.register_buffer('IDX', to_real(se.IDX))

        print(f"Physics Layer Initialized: {self.n_bus} Buses, {self.n_branch} Branches.")

    def forward(self, v_est_real, v_est_imag):
        v = torch.complex(v_est_real, v_est_imag).permute(1, 0)

        v_f, i_f = torch.matmul(self.Cf, v), torch.matmul(self.Yf, v)
        sf = v_f * torch.conj(i_f)

        v_t, i_t = torch.matmul(self.Ct, v), torch.matmul(self.Yt, v)
        st = v_t * torch.conj(i_t)

        i_inj = torch.matmul(self.Ybus, v)
        si = v * torch.conj(i_inj)

        z_parts = [sf.real, st.real, si.real, torch.angle(v),
                   sf.imag, st.imag, si.imag, torch.abs(v)]
        z_full = torch.cat(z_parts, dim=0)
        z_est = torch.matmul(self.IDX, z_full).permute(1, 0)
        return z_est


class TimeCompressor(nn.Module):
    """Map a measurement window to the final bidirectional LSTM states."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        encoded = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return encoded


class LatentTopoAttention(nn.Module):
    """Aggregate bus embeddings with a soft topology bias and one shared tau."""
    def __init__(self, latent_dim, n_bus, d_model=16):
        super().__init__()
        self.n_bus = n_bus
        self.d_model = d_model
        self.lambda_tpo = 0.5

        self.latent_to_node = nn.Linear(latent_dim, n_bus * d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.tau = nn.Parameter(torch.ones(1) * 0.7)

    def forward(self, x, adj_matrix):
        B = x.shape[0]
        h = self.latent_to_node(x).view(B, self.n_bus, self.d_model)

        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)

        topo_bias = torch.where(adj_matrix > 0, self.lambda_tpo, -self.lambda_tpo)
        scores = scores + topo_bias.unsqueeze(0)

        attn_weights = torch.softmax(scores / (torch.abs(self.tau) + 1e-5), dim=-1)
        h_prime = torch.bmm(attn_weights, V)

        return h_prime.view(B, self.n_bus * self.d_model)


class FeatureCompressor(nn.Module):
    """
    将拓扑自注意力聚合后的节点特征映射为电压实部与虚部。
    Case14,Case30: hidden_dim = 256
    Case57: hidden_dim = 512
    """
    def __init__(self, input_dim, n_bus, hidden_dim=256):
        super().__init__()

        self.real_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_bus)
        )

        self.imag_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_bus)
        )

    def forward(self, x):
        v_real = self.real_head(x)
        v_imag = self.imag_head(x)
        return v_real, v_imag


class PI_Encoder(nn.Module):
    def __init__(self, n_len, n_fea, n_bus, d_model=16):
        super().__init__()
        self.n_bus = n_bus

        # 加载拓扑邻接矩阵，并注册为静态 Buffer
        se = load_case()
        A_np = (np.abs(se.Ybus) > 1e-6).astype(np.float32)
        np.fill_diagonal(A_np, 1.0)
        self.register_buffer('adj_matrix', torch.tensor(A_np))

        self.time_compressor = TimeCompressor(input_dim=n_fea, hidden_dim=64)  ## Choseable Case57: 128, Case30,Case14: 64

        self.topo_attention = LatentTopoAttention(
            latent_dim=self.time_compressor.out_dim,
            n_bus=n_bus,
            d_model=d_model
        )

        topo_out_dim = n_bus * d_model
        fc_hidden = 512 if n_bus == 57 else 256
        self.feature_compressor = FeatureCompressor(
            input_dim=topo_out_dim, n_bus=n_bus, hidden_dim=fc_hidden
        )

    def forward(self, x):
        x = self.time_compressor(x)
        x = self.topo_attention(x, self.adj_matrix)
        v_real, v_imag = self.feature_compressor(x)
        return v_real, v_imag


class PI_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = nn_setting['device']
        self.physics_layer = PowerFlowPhysics(device=self.device)
        self.n_bus = self.physics_layer.n_bus

    def forward(self, v_real, v_imag):
        z_recon = self.physics_layer(v_real, v_imag)
        return z_recon.squeeze(-1)


class PGAE_tau(nn.Module):
    """Temporal encoder, topology attention, and fixed AC measurement decoder."""

    def __init__(self):
        super().__init__()
        self.device = nn_setting['device']
        self.n_feature = nn_setting['feature_size']
        self.lr = nn_setting['lr']
        self.delta = nn_setting['delta']
        self.path = nn_setting['pgae_tau_model_path']
        self.patience = nn_setting['patience']
        self.epochs = nn_setting['epochs']
        self.sample_length = nn_setting['sample_length']

        self.decoder = PI_Decoder()
        self.encoder = PI_Encoder(n_len=self.sample_length, n_fea=self.n_feature, n_bus=self.decoder.n_bus)

    def forward(self, x):
        v_real, v_imag = self.encoder(x)
        decoded = self.decoder(v_real, v_imag)
        return decoded, v_real, v_imag

    def fit(self, train_loader, valid_loader):
        self.to(self.device)

        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        criterion = nn.MSELoss(reduction='mean')
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=self.delta, path=self.path)

        num_batches_train = len(train_loader)
        num_batches_valid = len(valid_loader)

        obj = train_loader.dataset
        try:
            scaler_min = obj.detransform.min
            scaler_max = obj.detransform.max
        except AttributeError:
            scaler_min = obj.min if hasattr(obj, 'min') else 0.0
            scaler_max = obj.max if hasattr(obj, 'max') else 1.0

        t_min = torch.tensor(scaler_min, device=self.device, dtype=torch.float32)
        t_max = torch.tensor(scaler_max, device=self.device, dtype=torch.float32)

        def transform(data_unnorm):
            return (data_unnorm - t_min)/(t_max - t_min + 1e-7)

        self.apply(init_weights_PI)
        initialize_flat_start(self.encoder, self.decoder.n_bus)

        print('Start Training PGAE-tau (Shared Learned Temperature)...')
        print('*'*60)

        for epoch in range(1, self.epochs+1):
            start_time = time()

            self.train()
            train_loss = 0
            train_mape = 0

            for input, target in train_loader:
                input, target = input.to(self.device), target.to(self.device)

                recon, _, _ = self(input)
                target_last = target[:, -1, :]
                recon_norm = transform(recon)
                loss = criterion(recon_norm, target_last)

                mape = torch.mean(torch.abs((target_last - recon) / (target_last + 1e-6)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mape += mape.item()

            self.eval()
            valid_loss = 0
            with torch.no_grad():
                for input, target in valid_loader:
                    input, target = input.to(self.device), target.to(self.device)
                    recon, _, _ = self(input)
                    target_last = target[:, -1, :]
                    recon_norm = transform(recon)
                    loss = criterion(recon_norm, target_last)
                    valid_loss += loss.item()

            train_loss_avg = train_loss/num_batches_train
            valid_loss_avg = valid_loss/num_batches_valid
            mape_avg = train_mape /num_batches_train

            print(f'Train Loss: {round(train_loss_avg, 7)} | Valid Loss: {round(valid_loss_avg, 7)} | MAPE: {round(mape_avg,7)}')

            early_stopping(valid_loss_avg, self)
            print(f'Epoch {epoch} time {time()-start_time:.2f}s')
            print('*'*60)

            scheduler.step() # 更新学习率
            print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.load_state_dict(
            torch.load(self.path, map_location=self.device, weights_only=True)
        )


def init_weights_PI(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def initialize_flat_start(encoder_model, n_bus=None):
    fc = encoder_model.feature_compressor
    last_real_layer = fc.real_head[-1]
    nn.init.normal_(last_real_layer.weight, mean=0.0, std=0.01)
    with torch.no_grad():
        last_real_layer.bias.fill_(1.0)
        last_real_layer.bias.add_(torch.randn_like(last_real_layer.bias) * 0.05)

    last_imag_layer = fc.imag_head[-1]
    nn.init.normal_(last_imag_layer.weight, mean=0.0, std=0.01)
    with torch.no_grad():
        last_imag_layer.bias.fill_(0.0)
        last_imag_layer.bias.add_(torch.randn_like(last_imag_layer.bias) * 0.05)



if __name__ == '__main__':
    from configs.config import sys_config
    from torch.utils.data import DataLoader
    from data.dataset import rnn_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    pg_ae = PGAE_tau().to(device)

    train_set = rnn_dataset(mode='train', istransform=True)
    train_loader = DataLoader(train_set, batch_size=nn_setting['batch_size'], shuffle=True)
    valid_set = rnn_dataset(mode='valid', istransform=True)
    valid_loader = DataLoader(valid_set, batch_size=nn_setting['batch_size'], shuffle=False)

    pg_ae.fit(train_loader, valid_loader)
