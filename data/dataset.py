"""
Contains PyTorch dataset and dataloader for RNN
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import sys
from configs.nn_setting import nn_setting
from configs.config import sys_config
from configs.paths import data_path

"""
Inherit a RNN PyTorch Dataset
"""

class rnn_dataset(Dataset):
    def __init__(self, mode, istransform):

        # load measurement
        z_noise_summary = np.load(data_path('z_noise_summary.npy'), allow_pickle=False)
        feature_size = len(z_noise_summary)
        if z_noise_summary.ndim != 2 or z_noise_summary.shape[1] != nn_setting['feature_size']:
            raise ValueError(f"Measurements must have shape (N, {nn_setting['feature_size']}).")

        if mode == 'train':
            feature = z_noise_summary[:int(feature_size * nn_setting['train_prop'])]
        elif mode == 'valid':
            feature = z_noise_summary[int(feature_size * nn_setting['train_prop']):int(feature_size * (nn_setting['train_prop'] + nn_setting['valid_prop']))]
        elif mode == 'test':
            feature = z_noise_summary[int(feature_size * (nn_setting['train_prop'] + nn_setting['valid_prop'])):]
        else:
            raise ValueError("mode must be 'train', 'valid', or 'test'")

        self.feature = feature
        self.sample_length = nn_setting['sample_length']    # sample length
        if len(feature) <= self.sample_length:
            raise ValueError(f"The {mode} partition needs more than {self.sample_length} measurements.")

        # Scaling
        self.transform = scaler()  # Instance
        self.detransform = inverse_scaler()  # Instance
        self.istransform = istransform
        self.feature_size = feature_size

    def __len__(self):
        return len(self.feature) - self.sample_length + 1

    def __getitem__(self, idx):

        # transformation
        if self.istransform:
            input = self.transform(self.feature[idx:idx+self.sample_length])
        else:
            input = self.feature[idx:idx+self.sample_length]

        input = torch.from_numpy(input).float()  # Output tensor is float32
        output = input.clone()

        return (input, output)

class rnn_dataset_evl(rnn_dataset):
    """
    A dataset to output
    1. The current measurement
    1. nn input measurement: (1,sample_length,feature_size);
    2. voltage of the previous measurement;
    3. voltage of the current measurement
    """
    def __init__(self, mode, istransform):
        super().__init__(mode, istransform)  # Inherit from rnn_dataset

        # Load estimated state
        v_est = np.load(data_path('v_est_summary.npy'), allow_pickle=False)
        if len(v_est) != self.feature_size:
            raise ValueError("Measurements and state estimates must have the same number of rows.")

        if mode == 'train':
            self.v_est = v_est[:int(self.feature_size * nn_setting['train_prop'])]
        elif mode == 'valid':
            self.v_est = v_est[int(self.feature_size * nn_setting['train_prop']):int(self.feature_size * (nn_setting['train_prop'] + nn_setting['valid_prop']))]
        elif mode == 'test':
            self.v_est = v_est[int(self.feature_size * (nn_setting['train_prop'] + nn_setting['valid_prop'])):]
        else:
            raise ValueError("mode must be 'train', 'valid', or 'test'")

    def __len__(self):
        return len(self.feature) - self.sample_length + 1 - 1 # Start from the second measurement in each train, valid, and test dataset

    def __getitem__(self, idx):    # Rewrite

        # Transformation on measurement
        if self.istransform:
            input = self.transform(self.feature[idx+1:idx+1+self.sample_length])
        else:
            input = self.feature[idx+1:idx+1+self.sample_length]

        # Current Measurement
        input = torch.from_numpy(input).float()  # Output tensor is float32

        # State of the current measurement
        v_est_last = self.v_est[idx + 1 + nn_setting['sample_length']-1]

        # State of the previous measurement
        v_est_pre = self.v_est[idx + nn_setting['sample_length'] - 1]

        #print(f'v_est_pre shape: {v_est_pre.shape}')
        #print(f'v_est_last shape: {v_est_last.shape}')
        #print(f'input shape: {input.shape}')

        # The measurement in (1,sample_length,feature_size) and voltage reference at the last measurement
        return (idx, input, v_est_pre, v_est_last)

"""
Scaling transform
"""

class scaler:
    def __init__(self):
        # Find the statistic on the training dataset

        # load measurement
        z_noise_summary = np.load(data_path('z_noise_summary.npy'), allow_pickle=False)
        feature_size = len(z_noise_summary)
        z_noise_summary = z_noise_summary[:int(feature_size * nn_setting['train_prop'])]

        self.train_feature = z_noise_summary

        # Min-Max Normalization
        self.min = np.min(self.train_feature, axis = 0)
        self.max = np.max(self.train_feature, axis = 0)

    def __call__(self, sample):
        # print(f'sample shape: {sample.shape}')
        # sample.shape = (sample_length, feature_size)
        input = sample
        return (input-self.min)/(self.max - self.min + 1e-7)  # To avoid numerical instable

class inverse_scaler(scaler):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        input = sample
        return input*(self.max - self.min + 1e-7)+self.min


# =========================================================================
# Low-and-Slow FDIA Evaluation Dataset (Author: Hawkyu)
# =========================================================================
# Extends rnn_dataset to return the full 6-step state estimation sequence
# and net-load profile, so gen_low_slow_att() can ramp the attack across
# the entire temporal window instead of only the last step.
# =========================================================================

class rnn_dataset_lowslow(rnn_dataset):
    """
    Dataset for Low-and-Slow FDIA attack evaluation.

    Returns (idx, input, v_est_seq, load_seq) per sample:
      - input      : (sample_length, feature_size) measurement window
      - v_est_seq  : (sample_length, n_bus) complex-valued state estimates
      - load_seq   : (sample_length,) total system net-load per timestep
    """
    def __init__(self, mode, istransform):
        super().__init__(mode, istransform)

        # Load estimated states (complex voltage, shape: (n_timesteps, n_bus))
        v_est_all = np.load(data_path('v_est_summary.npy'))
        # Load active load and PV for net-load computation
        load_active = np.load(data_path('load_active.npy'))
        pv_active = np.load(data_path('pv_active.npy'))

        # Partition according to the same train/valid/test split
        n = self.feature_size
        if mode == 'train':
            self.v_est = v_est_all[:int(n * nn_setting['train_prop'])]
            self.load_active = load_active[:int(n * nn_setting['train_prop'])]
            self.pv_active = pv_active[:int(n * nn_setting['train_prop'])]
        elif mode == 'valid':
            self.v_est = v_est_all[int(n * nn_setting['train_prop']):
                                   int(n * (nn_setting['train_prop'] + nn_setting['valid_prop']))]
            self.load_active = load_active[int(n * nn_setting['train_prop']):
                                           int(n * (nn_setting['train_prop'] + nn_setting['valid_prop']))]
            self.pv_active = pv_active[int(n * nn_setting['train_prop']):
                                       int(n * (nn_setting['train_prop'] + nn_setting['valid_prop']))]
        elif mode == 'test':
            self.v_est = v_est_all[int(n * (nn_setting['train_prop'] + nn_setting['valid_prop'])):]
            self.load_active = load_active[int(n * (nn_setting['train_prop'] + nn_setting['valid_prop'])):]
            self.pv_active = pv_active[int(n * (nn_setting['train_prop'] + nn_setting['valid_prop'])):]
        else:
            raise ValueError("mode must be 'train', 'valid', or 'test'")

    def __len__(self):
        # Same sliding-window offset as rnn_dataset_evl
        return len(self.feature) - self.sample_length

    def __getitem__(self, idx):
        # ---- Measurement window: (sample_length, feature_size) ----
        # Use idx+1 offset (consistent with rnn_dataset_evl)
        if self.istransform:
            input_data = self.transform(self.feature[idx+1 : idx+1+self.sample_length])
        else:
            input_data = self.feature[idx+1 : idx+1+self.sample_length]
        input_data = torch.from_numpy(input_data).float()

        # ---- Full state sequence: (sample_length, n_bus) ----
        v_est_seq = self.v_est[idx+1 : idx+1+self.sample_length]

        # ---- Net-load profile: (sample_length,) ----
        # Net load = total active load minus total PV generation at each step
        load_slice = self.load_active[idx+1 : idx+1+self.sample_length]
        pv_slice = self.pv_active[idx+1 : idx+1+self.sample_length]
        load_seq = np.sum(load_slice, axis=1) - np.sum(pv_slice, axis=1)  # -> (sample_length,)

        return (idx, input_data, v_est_seq, load_seq)


"""
Test Dataset
"""
if __name__ == '__main__':
    import sys
    from torch.utils.data import DataLoader

    # Test dataloader
    train_set = rnn_dataset(mode = 'train', istransform=True)
    train_loader = DataLoader(train_set,
                            batch_size=nn_setting['batch_size'],
                            shuffle=False)

    for batch_idx, (input, output) in enumerate(train_loader):
        print(f'input shape: {input.shape}')
        assert torch.equal(input, output)
        max_ = torch.amax(input, dim = (0,1))
        min_ = torch.amin(input, dim = (0,1))
        print(max_)
        print(min_)

        break
