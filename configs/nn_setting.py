"""
Settings for Neural Network
"""
import torch
import sys
import os
from configs.config import sys_config
from configs.paths import checkpoint_path

case_name = sys_config['case_name']

if case_name == 'case14':
    nn_setting = {
        'sample_length': 6,
        'lattent_dim': 16,
        'no_layer': 3,
        'feature_size':68,
        'epochs': 1000,
        'lr': 3e-4,
        'patience': 20,
        'delta': 0,
        'model_path': 'saved_model/case14/LSTMAE.pt',
        'pgae_tau_model_path': 'saved_model/case14/PGAE_tau.pt',
        'gnn_K': 2,
        'gnn_hidden': 16,
        'gnn_latent': 2,
        'anomtrans_d_model': 64,
        'anomtrans_n_heads': 4,
        'anomtrans_e_layers': 2,
        'anomtrans_d_ff': 256,
        'anomtrans_dropout': 0.1,
        'anomtrans_lr': 1e-4,
        'anomtrans_k': 3,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'batch_size': 32,
        'lattent_weight': 0.0,
        'train_prop': 0.6,
        'valid_prop': 0.2,
        'recover_lr': 5*1e-3,
        'beta_real': 0.1,
        'beta_imag': 0.1,
        'beta_mag': 100,
        'mode': 'pre',
        'max_step_size': 1000,
        'min_step_size': 50
    }

elif case_name == 'case57':
    nn_setting = {
        'sample_length': 6,
        'lattent_dim': 32,
        'no_layer': 3,
        'feature_size':274,
        'epochs': 1000,
        'lr': 3e-4,
        'patience': 20,
        'delta': 0,
        'model_path': 'saved_model/case57/LSTMAE.pt',
        'pgae_tau_model_path': 'saved_model/case57/PGAE_tau.pt',
        'gnn_K': 3,
        'gnn_hidden': 64,
        'gnn_latent': 8,
        'anomtrans_d_model': 128,
        'anomtrans_n_heads': 4,
        'anomtrans_e_layers': 3,
        'anomtrans_d_ff': 512,
        'anomtrans_dropout': 0.1,
        'anomtrans_lr': 1e-4,
        'anomtrans_k': 3,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'batch_size': 32,
        'lattent_weight': 0.0,
        'train_prop': 0.6,
        'valid_prop': 0.2,
        'recover_lr': 5*1e-3,
        'beta_real': 0.1,
        'beta_imag': 0.1,
        'beta_mag': 100,
        'mode': 'pre',
        'max_step_size': 1000,
        'min_step_size': 50
    }

elif case_name == 'case30':
    nn_setting = {
        'sample_length': 6,
        'lattent_dim': 32,
        'no_layer': 3,
        'feature_size':142,
        'epochs': 1000,
        'lr': 3e-4,
        'patience': 20,
        'delta': 0,
        'model_path': 'saved_model/case30/LSTMAE.pt',
        'pgae_tau_model_path': 'saved_model/case30/PGAE_tau.pt',
        'gnn_K': 2,
        'gnn_hidden': 64,
        'gnn_latent': 5,
        'anomtrans_d_model': 128,
        'anomtrans_n_heads': 4,
        'anomtrans_e_layers': 3,
        'anomtrans_d_ff': 512,
        'anomtrans_dropout': 0.1,
        'anomtrans_lr': 1e-4,
        'anomtrans_k': 3,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'batch_size': 32,
        'lattent_weight': 0.0,
        'train_prop': 0.6,
        'valid_prop': 0.2,
        'recover_lr': 5*1e-3,
        'beta_real': 0.1,
        'beta_imag': 0.1,
        'beta_mag': 100,
        'mode': 'pre',
        'max_step_size': 1000,
        'min_step_size': 50
    }

else:
    raise ValueError(f'Unsupported case: {case_name}')

nn_setting['model_path'] = str(checkpoint_path('LSTMAE.pt'))
nn_setting['pgae_tau_model_path'] = str(checkpoint_path('PGAE_tau.pt'))
nn_setting['device'] = os.environ.get('PGAE_DEVICE', nn_setting['device'])
