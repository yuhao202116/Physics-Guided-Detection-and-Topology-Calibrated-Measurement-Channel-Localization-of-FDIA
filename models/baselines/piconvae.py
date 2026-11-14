import os
import sys
from time import time

import torch
import torch.nn as nn
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from training.early_stopping import EarlyStopping
from configs.nn_setting import nn_setting
from data.loaders import load_case
from configs.config import sys_config
from configs.paths import checkpoint_path, data_path
from torch.utils.data import DataLoader
from data.dataset import rnn_dataset

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


class PIConv_Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim=64):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, latent_dim, kernel_size=2, padding=0)
        self.lrelu = nn.ReLU(True)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        latent = self.conv3(x).squeeze(-1) # [B, latent_dim]
        return latent

class PIConv_Decoder_DD(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(latent_dim, 64, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(64)
        self.deconv2 = nn.ConvTranspose1d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(128)
        self.deconv3 = nn.ConvTranspose1d(128, out_channels, kernel_size=3, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.unsqueeze(-1) # [B, latent_dim, 1]
        x = self.lrelu(self.bn1(self.deconv1(x)))
        x = self.lrelu(self.bn2(self.deconv2(x)))
        decoded = self.deconv3(x) # [B, out_channels, 6]
        decoded = decoded.permute(0, 2, 1) # [B, 6, out_channels]
        return decoded

class PIConv_StateEstimator(nn.Module):
    """
    状态估计器: 从高凝聚度的时序空间推断系统的母线电压状态 (v_real, v_imag)
    """
    def __init__(self, input_dim, n_bus, hidden_dim=256):
        super().__init__()
        self.real_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_bus)
        )

        self.imag_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_bus)
        )

    def forward(self, latent):
        v_real = self.real_head(latent)
        v_imag = self.imag_head(latent)
        return v_real, v_imag

class PIConvAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = nn_setting['device']
        self.n_feature = nn_setting['feature_size']
        self.lr = nn_setting['lr']
        self.delta = nn_setting['delta']
        self.path = str(checkpoint_path('PIConvAE.pt'))
        self.patience = nn_setting['patience']
        self.epochs = nn_setting['epochs']

        # 实例化自适应物理潮流层
        self.physics_layer = PowerFlowPhysics(device=self.device)
        self.n_bus = self.physics_layer.n_bus

        # Case14: Optional Choise :
        # latent_dim = 48
        # Case57,Case30:Optional Choise
        # latent_dim = 32
        latent_dim = nn_setting['lattent_dim']
        hidden_dim_se = 256

        # 实例化子模块
        self.encoder = PIConv_Encoder(in_channels=self.n_feature, latent_dim=latent_dim)
        self.decoder_dd = PIConv_Decoder_DD(latent_dim=latent_dim, out_channels=self.n_feature)
        self.state_estimator = PIConv_StateEstimator(input_dim=latent_dim, n_bus=self.n_bus, hidden_dim=hidden_dim_se)

    def forward(self, x):
        latent = self.encoder(x)
        recon_dd = self.decoder_dd(latent) # 输出完整时序 [B, 6, 274]
        v_real, v_imag = self.state_estimator(latent)
        return recon_dd, v_real, v_imag

    def fit(self, train_loader, valid_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction='mean')

        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=self.delta, path=self.path)

        num_batches_train = len(train_loader)
        num_batches_valid = len(valid_loader)

        # 逆归一化与归一化工具类
        obj = train_loader.dataset
        scaler_min = obj.detransform.min
        scaler_max = obj.detransform.max
        t_min = torch.tensor(scaler_min, device=self.device, dtype=torch.float32)
        t_max = torch.tensor(scaler_max, device=self.device, dtype=torch.float32)

        def detransform(data_norm):
            return data_norm * (t_max - t_min + 1e-7) + t_min

        def transform(data_unnorm):
            return (data_unnorm - t_min)/(t_max - t_min + 1e-7)

        self.apply(init_weights_PI)
        initialize_flat_start(self.state_estimator)

        print('Start Training Optimized PIConvAE')
        print('*'*60)

        # 损失函数平衡系数
        alpha_d = 0.5     # 数据驱动重构权重
        alpha_phy = 0.5   # 物理规律约束权重

        for epoch in range(1, self.epochs+1):
            start_time = time()

            # --- Train ---
            self.train()
            train_loss = 0
            train_mape = 0

            for batch_idx, (input, target) in enumerate(train_loader):
                input, target = input.to(self.device), target.to(self.device)

                # 模型前向传播
                recon_dd, v_real, v_imag = self(input)
                recon_dd_norm = transform(recon_dd)

                # 物理层计算
                z_phy = self.physics_layer(v_real, v_imag)
                z_phy_norm = transform(z_phy)

                loss_ae = criterion(recon_dd_norm, target)

                loss_phy = criterion(z_phy_norm, target[:, -1, :])

                loss = alpha_d * loss_ae + alpha_phy * loss_phy

                recon_dd_detransformed = detransform(recon_dd_norm[:, -1, :])
                target_last_detransformed = detransform(target[:, -1, :])
                mape = torch.mean(torch.abs((target_last_detransformed - recon_dd_detransformed) / (target_last_detransformed + 1e-6)))

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

                    recon_dd, v_real, v_imag = self(input)
                    recon_dd_norm = transform(recon_dd)

                    z_phy = self.physics_layer(v_real, v_imag)
                    z_phy_norm = transform(z_phy)

                    l_ae = criterion(recon_dd_norm, target)
                    l_phy = criterion(z_phy_norm, target[:, -1, :])
                    v_loss = alpha_d * l_ae + alpha_phy * l_phy

                    valid_loss += v_loss.item()

            train_loss_avg = train_loss/num_batches_train
            valid_loss_avg = valid_loss/num_batches_valid
            mape_avg = train_mape /num_batches_train

            print(f'Epoch {epoch} | Time: {time()-start_time:.2f}s')
            print(f'Train Loss: {train_loss_avg:.7f} | MAPE: {mape_avg:.7f}')
            print(f'Valid Loss: {valid_loss_avg:.7f}')
            print('-'*60)

            early_stopping(valid_loss_avg, self)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        try:
            self.load_state_dict(torch.load(self.path))
        except Exception as e:
            print(f"Warning during loading best model: {e}")


def init_weights_PI(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def initialize_flat_start(estimator_model):
    """ 电力系统平坦启动初始化 (V_real ~ 1.0, V_imag ~ 0.0) """
    last_real_layer = estimator_model.real_head[-1]
    nn.init.normal_(last_real_layer.weight, mean=0.0, std=0.01)
    with torch.no_grad():
        last_real_layer.bias.fill_(1.0)
        last_real_layer.bias.add_(torch.randn_like(last_real_layer.bias) * 0.02)

    last_imag_layer = estimator_model.imag_head[-1]
    nn.init.normal_(last_imag_layer.weight, mean=0.0, std=0.01)
    with torch.no_grad():
        last_imag_layer.bias.fill_(0.0)
        last_imag_layer.bias.add_(torch.randn_like(last_imag_layer.bias) * 0.02)



if __name__ == '__main__':
    z_noise_path = data_path('z_noise_summary.npy')
    z_noise_summary = np.load(z_noise_path)
    print(f'Successfully loaded measurement data. Shape: {z_noise_summary.shape}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using executable runtime device: {device}")

    # 模型实例化
    model = PIConvAE().to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of parameters is {pytorch_total_params}')
    print(f'The total trainable number of parameters is {pytorch_total_params_train}')

    # 训练集加载
    train_set = rnn_dataset(mode='train', istransform=True)
    train_loader = DataLoader(train_set, batch_size=nn_setting['batch_size'], shuffle=True)

    # 验证集加载
    valid_set = rnn_dataset(mode='valid', istransform=True)
    valid_loader = DataLoader(valid_set, batch_size=nn_setting['batch_size'], shuffle=False)

    print(f'Train set samples: {len(train_loader.dataset)} | Valid set samples: {len(valid_loader.dataset)}')

    # 启动训练流程
    model.fit(train_loader, valid_loader)
