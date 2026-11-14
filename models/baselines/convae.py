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
from data.dataset import rnn_dataset
from torch.utils.data import DataLoader
from configs.config import sys_config
from configs.paths import checkpoint_path, data_path
import numpy as np

torch.manual_seed(0)

class ConvEncoderModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.no_feature = nn_setting['feature_size']
        self.latten_dim = nn_setting['lattent_dim']


        self.cnn_encoder = nn.Sequential(

            nn.Conv1d(in_channels=self.no_feature, out_channels=128, kernel_size=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(True),


            nn.Conv1d(in_channels=32, out_channels=self.latten_dim, kernel_size=2, padding=0),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, 274, 6)
        encoded = self.cnn_encoder(x) # (B, latent_dim, 1)
        encoded = encoded.permute(0, 2, 1) # (B, 1, latent_dim)
        return encoded

class ConvDecoderModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.no_feature = nn_setting['feature_size']
        self.latten_dim = nn_setting['lattent_dim']

        self.cnn_decoder = nn.Sequential(

            nn.ConvTranspose1d(in_channels=self.latten_dim, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(in_channels=128, out_channels=self.no_feature, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        decoded = self.cnn_decoder(x)
        decoded = decoded.permute(0, 2, 1) # (B, 6, 274)
        return decoded

class Conv_AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoderModule()
        self.decoder = ConvDecoderModule()

        self.epochs = nn_setting['epochs']
        self.lr = nn_setting['lr']
        self.patience = nn_setting['patience']
        self.delta = nn_setting['delta']
        self.path = str(checkpoint_path('ConvAE.pt'))
        self.device = nn_setting['device']

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def fit(self, train_loader, valid_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        criterion = nn.MSELoss(reduction='mean')
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, delta=self.delta, path=self.path)

        num_batches_train = len(train_loader)
        num_batches_valid = len(valid_loader)
        self.apply(init_weights)

        print(f'Start Training ConvAE for {sys_config["case_name"]}...')
        print('*'*60)

        for epoch in range(1, self.epochs+1):
            start_time = time()
            self.train()
            train_loss = 0

            for batch_idx, (input, output) in enumerate(train_loader):
                self.to(self.device)
                input, output = input.to(self.device), output.to(self.device)

                encoded, decoded = self(input)
                loss = criterion(decoded, output)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.eval()
            valid_loss = 0
            with torch.no_grad():
                for input, output in valid_loader:
                    input, output = input.to(self.device), output.to(self.device)
                    encoded, decoded = self(input)
                    loss = criterion(decoded, output)
                    valid_loss += loss.item()

            print(f'Train Loss: {round(train_loss/num_batches_train, 7)}')
            print(f'Valid Loss: {round(valid_loss/num_batches_valid, 7)}')

            end_time = time()
            early_stopping(valid_loss/num_batches_valid, self)
            print(f'Epoch {epoch} time {end_time-start_time:.2f}s')
            print('*'*60)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.load_state_dict(torch.load(self.path, map_location=self.device, weights_only=True))

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)

if __name__ == '__main__':
    from configs.config import sys_config
    z_noise_path = data_path('z_noise_summary.npy')
    z_noise_summary = np.load(z_noise_path)
    print(f'Data shape {z_noise_summary.shape}')

    CAE = Conv_AE()
    CAE.apply(init_weights)
    print(CAE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_set = rnn_dataset(mode = 'train', istransform=True)
    train_loader = DataLoader(train_set, batch_size=nn_setting['batch_size'], shuffle=True)
    valid_set = rnn_dataset(mode = 'valid', istransform=True)
    valid_loader = DataLoader(valid_set, batch_size=nn_setting['batch_size'], shuffle=False)

    CAE.fit(train_loader, valid_loader)
