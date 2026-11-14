"""
Load the case class based on the specifications in sys_config
"""

import numpy as np
from configs.config import sys_config
from configs.paths import data_path
from configs.config_mea_idx import define_mea_idx_noise
from data.generation import gen_case
from physics.attacks import FDI
from data.dataset import rnn_dataset, rnn_dataset_evl, rnn_dataset_lowslow
from torch.utils.data import DataLoader

# Case Class
def load_case():
    """
    Return the instance case class
    """
    case = gen_case(sys_config['case_name']) # Modify case
    # Instance case class
    idx, no_mea, _ = define_mea_idx_noise(case, sys_config['measure_type'])
    noise_sigma = np.load(data_path('noise_sigma.npy'), allow_pickle=False)
    if noise_sigma.shape != (no_mea,) or not np.all(np.isfinite(noise_sigma)) or np.any(noise_sigma <= 0):
        raise ValueError(f"noise_sigma must contain {no_mea} finite positive values.")

    case_class = FDI(case, noise_sigma, idx, sys_config['fpr'])

    return case_class

# Load and PV

def load_load_pv():

    load_active_dir = data_path('load_active.npy')
    load_reactive_dir = data_path('load_reactive.npy')
    pv_active_dir = data_path('pv_active.npy')
    pv_reactive_dir = data_path('pv_reactive.npy')

    load_active = np.load(load_active_dir)
    load_reactive = np.load(load_reactive_dir)
    pv_active = np.load(pv_active_dir)
    pv_reactive = np.load(pv_reactive_dir)

    # set the length equal to the bus number
    pv_active_ = np.zeros((load_active.shape[0], load_reactive.shape[1]))
    pv_reactive_ = np.zeros((load_reactive.shape[0], load_reactive.shape[1]))
    pv_active_[:,sys_config['pv_bus']] = pv_active
    pv_reactive_[:,sys_config['pv_bus']] = pv_reactive

    return load_active, load_reactive, pv_active_, pv_reactive_

def load_measurement():

    # Load measurement
    z_noise_summary = np.load(data_path('z_noise_summary.npy'))
    v_est_summary = np.load(data_path('v_est_summary.npy'))
    print(f'z noise size: {z_noise_summary.shape}')
    print(f'v est size: {v_est_summary.shape}')

    return z_noise_summary, v_est_summary

def load_dataset():

    is_shuffle = True  # Allow shuffle in dataloader

    # Test dataset dataloader: SCALED
    test_dataset_scaled = rnn_dataset(mode = 'test', istransform=True)
    test_dataloader_scaled = DataLoader(dataset = test_dataset_scaled,
                                    shuffle=is_shuffle,
                                    batch_size=1)
    print(f'test dataset size scaled: {len(test_dataloader_scaled.dataset)}')

    # Test dataset dataloader: UNSCALED
    # With Random
    test_dataset_unscaled = rnn_dataset_evl(mode = 'test', istransform=False)
    test_dataloader_unscaled = DataLoader(dataset = test_dataset_unscaled,
                                shuffle=is_shuffle,
                                batch_size=1)
    print(f'test dataset size unscaled: {len(test_dataloader_unscaled.dataset)}')

    # Valid dataset: SCALED
    valid_dataset_scaled = rnn_dataset(mode = 'valid', istransform=True)
    valid_dataloader_scaled = DataLoader(dataset = valid_dataset_scaled,
                                    shuffle=is_shuffle,
                                    batch_size=1)
    print(f'valid dataset size scaled: {len(valid_dataloader_scaled.dataset)}')

    # Validation dataset dataloader: UNSCALED
    # No Random
    valid_dataset_unscaled = rnn_dataset_evl(mode = 'valid', istransform=False)
    valid_dataloader_unscaled = DataLoader(dataset = valid_dataset_unscaled,
                                shuffle=is_shuffle,
                                batch_size=1)
    print(f'valid dataset size unscaled: {len(valid_dataloader_unscaled.dataset)}')

    return test_dataloader_scaled, test_dataloader_unscaled, valid_dataloader_scaled, valid_dataloader_unscaled


def load_dataset_lowslow():
    """
    Load the test dataset for Low-and-Slow FDIA attack evaluation.
    """
    is_shuffle = True

    test_dataset_lowslow = rnn_dataset_lowslow(mode='test', istransform=False)
    test_dataloader_lowslow = DataLoader(
        dataset=test_dataset_lowslow,
        shuffle=is_shuffle,
        batch_size=1
    )
    print(f'[Low-Slow] test dataset size (unscaled): {len(test_dataloader_lowslow.dataset)}')

    return test_dataloader_lowslow
