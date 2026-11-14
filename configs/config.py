"""
Configuration for:
1. State Estimation
2. Optimal Power Flow
"""

import os
from pathlib import Path
import numpy as np

# Compatibility for PYPOWER routines on NumPy 2.x.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pypower.api import ppoption

# State Estimation Settings
se_config = {
    'tol': 1e-3,
    'max_it': 100,
    'verbose': 0
}

# OPF Settings
opt = ppoption()
opt['VERBOSE'] = 0
opt['OUT_ALL'] = 0
opt['OPF_FLOW_LIM'] = 1
opt['OPF_VIOLATION'] = 1e-3


# Template: all case configs
sys_config_template = {}

sys_config_template['case14'] = {
    'case_name': 'case14',
    'load_resolution': '5min',
    'fpr': 0.02,
    'noise_ratio_power': 0.02,
    'noise_ratio_voltage': 0.001,
    'pv_bus': np.array([4, 5, 11, 13]) - 1,
    'measure_type': 'HALF_RTU',
}

sys_config_template['case57'] = {
    'case_name': 'case57',
    'load_resolution': '5min',
    'fpr': 0.02,
    'noise_ratio_power': 0.02,
    'noise_ratio_voltage': 0.001,
    'pv_bus': np.array([2, 3, 6, 8, 9, 12]) - 1,
    'measure_type': 'HALF_RTU',
}

sys_config_template['case30'] = {
    'case_name': 'case30',
    'load_resolution': '5min',
    'fpr': 0.02,
    'noise_ratio_power': 0.02,
    'noise_ratio_voltage': 0.001,
    'pv_bus': np.array([2, 5, 8, 11, 13]) - 1,
    'measure_type': 'HALF_RTU',
}


# Select the case before importing model or dataset modules.
default_case = os.environ.get('PGAE_CASE', 'case30')
if default_case not in sys_config_template:
    raise ValueError(f'Unsupported case: {default_case}')
sys_config = sys_config_template[default_case]


def save_metric(address, **kwargs):
    Path(address).parent.mkdir(parents=True, exist_ok=True)
    metric = {}
    for key in kwargs.keys():
        metric[key] = kwargs[key]
    np.save(address, metric, allow_pickle=True)
