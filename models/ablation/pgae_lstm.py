import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import torch
import torch.nn as nn

from training.early_stopping import EarlyStopping
from time import time
from configs.nn_setting import nn_setting
from configs.paths import data_path, checkpoint_path
from data.loaders import load_case
torch.manual_seed(0)


class PowerFlowPhysics(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()

        print("Initializing Physics Layer via SE Instance...")
        se = load_case()

        Ybus_np = se.Ybus
        Yf_np   = se.Yf
        Yt_np   = se.Yt
        Cf_np   = se.Cf
        Ct_np   = se.Ct

        n_bus    = se.no_bus
        n_branch = se.no_brh

        self.n_bus = n_bus
        self.n_branch = n_branch
        self.dim_total = n_branch * 4 + n_bus * 4

        IDX_np = se.IDX

        def to_cplx(x): return torch.tensor(x, device=device, dtype=torch.cfloat)
        def to_real(x): return torch.tensor(x, device=device, dtype=torch.float32)

        self.register_buffer('Ybus', to_cplx(Ybus_np))
        self.register_buffer('Yf', to_cplx(Yf_np))
        self.register_buffer('Yt', to_cplx(Yt_np))
        self.register_buffer('Cf', to_cplx(Cf_np))
        self.register_buffer('Ct', to_cplx(Ct_np))
        self.register_buffer('IDX', to_real(IDX_np))

        print(f"Physics Layer Initialized: {n_bus} Buses, {n_branch} Branches.")

    def forward(self, v_est_real, v_est_imag):
        v = torch.complex(v_est_real, v_est_imag)
        v = v.permute(1, 0)

        v_f = torch.matmul(self.Cf, v)
        i_f = torch.matmul(self.Yf, v)
        sf = v_f * torch.conj(i_f)

        v_t = torch.matmul(self.Ct, v)
        i_t = torch.matmul(self.Yt, v)
        st = v_t * torch.conj(i_t)

        i_inj = torch.matmul(self.Ybus, v)
        si = v * torch.conj(i_inj)

        vang = torch.angle(v)
        vmag = torch.abs(v)

        z_parts = [
            sf.real, st.real, si.real, vang,
            sf.imag, st.imag, si.imag, vmag
        ]
        z_full = torch.cat(z_parts, dim=0)
        z_est = torch.matmul(self.IDX, z_full)
        z_est = z_est.permute(1, 0)

        return z_est


class TimeCompressor(nn.Module):
    """
    Time Compressor ：[Batch, Time, Feat] -> [Batch, Feat]
    """
    def __init__(self, input_dim, hidden_dim = 64):

        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout = 0.1
        )

        self.out_dim = hidden_dim * 2

    def forward(self, x):
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(x)
        encoded = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return encoded

class FeatureCompressor(nn.Module):
    """
    接收 128 维潜层特征，完成物理状态映射
    """
    def __init__(self, input_dim, n_bus, hidden_dim=256):
        super().__init__()

        self.real_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_bus)
        )

        self.imag_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_bus)
        )

    def forward(self, x):
        v_real = self.real_head(x)
        v_imag = self.imag_head(x)
        return v_real, v_imag


"""
ENCODER
"""
class PI_Encoder(nn.Module):
    def __init__(self, n_len, n_fea, n_bus):
        super().__init__()
        self.n_bus = n_bus

        self.time_compressor = TimeCompressor(input_dim = n_fea)
        self.feature_compressor = FeatureCompressor(input_dim = self.time_compressor.out_dim, n_bus = n_bus)

    def forward(self, x):
        x = self.time_compressor(x)
        v_real, v_imag = self.feature_compressor(x)
        return v_real, v_imag

"""
DECODER
"""
class PI_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = nn_setting['device']
        self.physics_layer = PowerFlowPhysics(device=self.device)
        self.n_bus = self.physics_layer.n_bus

    def forward(self, v_real, v_imag):
        z_recon = self.physics_layer(v_real, v_imag)
        return z_recon.squeeze(-1)


class PG_AE_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = nn_setting['device']
        self.n_feature = nn_setting['feature_size']
        self.lr = nn_setting['lr']
        self.delta = nn_setting['delta']
        self.path = str(checkpoint_path('PGAE_LSTM.pt'))
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

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

        def detransform(data_norm):
            return data_norm * (t_max - t_min + 1e-7) + t_min

        def transform(data_unnorm):
            return (data_unnorm - t_min)/(t_max - t_min + 1e-7)

        self.apply(init_weights_PI)
        initialize_flat_start(self.encoder, self.decoder.n_bus)

        print('Start Training PG_AE_LSTM...')
        print('*'*60)

        for epoch in range(1, self.epochs+1):
            start_time = time()

            # --- Train ---
            self.train()
            train_loss = 0
            train_mape = 0

            for batch_idx, (input, target) in enumerate(train_loader):
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

            # --- Validation ---
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

            # --- Logging ---
            train_loss_avg = train_loss/num_batches_train
            valid_loss_avg = valid_loss/num_batches_valid
            mape_avg = train_mape /num_batches_train

            print(f'Train Loss: {round(train_loss_avg, 7)}')
            print(f'mape :      {round(mape_avg,7)}')
            print(f'Valid Loss: {round(valid_loss_avg, 7)}')

            early_stopping(valid_loss_avg, self)

            print(f'Epoch {epoch} time {time()-start_time:.2f}s')
            print('*'*60)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.load_state_dict(torch.load(self.path))


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
    import numpy as np
    from configs.config import sys_config
    from torch.utils.data import DataLoader
    import sys
    sys.path.append("./")
    from data.dataset import rnn_dataset

    z_noise_summary = np.load(data_path('z_noise_summary.npy'))
    print(f'Data shape {z_noise_summary.shape}')

    device= "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    pg_ae = PG_AE_LSTM().to(device)
    print(pg_ae)

    pytorch_total_params = sum(p.numel() for p in pg_ae.parameters())
    pytorch_total_params_train = sum(p.numel() for p in pg_ae.parameters() if p.requires_grad)

    print(f'The total number of parameters is {pytorch_total_params}')
    print(f'The total trainable number of parameters is {pytorch_total_params_train}')

    train_set = rnn_dataset(mode = 'train', istransform = True)
    train_loader = DataLoader(train_set,
                            batch_size=nn_setting['batch_size'],
                            shuffle=True)

    valid_set = rnn_dataset(mode = 'valid', istransform = True)
    valid_loader = DataLoader(valid_set,
                            batch_size=nn_setting['batch_size'],
                            shuffle=False)

    print(f'Train set size: {len(train_loader.dataset)}')
    print(f'Valid set size: {len(valid_loader.dataset)}')

    pg_ae.fit(train_loader, valid_loader)
