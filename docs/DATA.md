# Data Guide / 数据说明

Prepare experiment arrays under `data/<case>/`. The `data/` package provides
grid generation, chronological datasets, and loaders. Data arrays and trained
checkpoints are stored locally and excluded from Git.

实验数组存放在 `data/<case>/`。`data/` 包提供电网生成、按时间划分的数据集与加载器。
数据数组和训练权重在本地保存，并由 Git 忽略。

## Directory and Arrays / 目录与数组

By default, place arrays in `data/<case>/`, with `<case>` equal to `case14`,
`case30`, or `case57`. `PGAE_DATA_DIR` may point to another root directory; the
loader appends `<case>/<filename>`. Save arrays with numeric NumPy dtypes.

| File | Shape | Meaning |
| --- | --- | --- |
| `z_noise_summary.npy` | `(N, F)` | Noisy measurements in original physical units |
| `noise_sigma.npy` | `(F,)` | Finite positive standard deviations, aligned with measurement columns |
| `v_est_summary.npy` | `(N, B)` | Complex voltage state estimates, aligned in time |
| `load_active.npy` | `(N, B)` | Active bus loads |
| `load_reactive.npy` | `(N, B)` | Reactive bus loads |
| `pv_active.npy` | `(N, P)` | PV active power at configured PV buses |
| `pv_reactive.npy` | `(N, P)` | PV reactive power at configured PV buses |
| `success_summary.npy` | `(N,)` | Optional OPF convergence flags |

`B` is the bus count, `P` is the number of entries in
`configs/config.py::sys_config["pv_bus"]`, and `F` is 68, 142, or 274 for the
14-, 30-, or 57-bus system. Main model training needs the first two files;
attack-based localization evaluation also needs `v_est_summary.npy`.
Load/PV arrays are required by profile-based experiments and data generation.
All rows must refer to the same chronological operating points.

`B` 表示节点数，`P` 表示配置中的光伏节点数，`F` 在 14、30、57 节点系统中分别为
68、142、274。核心训练使用前两个文件；基于攻击构造的定位评估还使用电压状态数组。
负荷与光伏数组用于原始数据处理及量测生成，各数组逐行对应同一时刻。

## Measurement Convention / 量测约定

The HALF-RTU configuration concatenates sending-end active branch flows, active
bus injections, sending-end reactive branch flows, and reactive bus injections.
The general ordering is
`pf, pt, pi, vang, qf, qt, qi, vmag`, with unselected groups empty.
`configs/config_mea_idx.py::define_mea_idx_noise` is authoritative.
Power measurements use the per-unit convention of `physics/state_estimation.py`, with
voltage magnitudes in per unit and voltage angles in radians.

Measurements are split chronologically at 60% and 80%. Sliding windows have six
steps, stride one, and do not cross partition boundaries. Min/max statistics are
computed from the entire training partition. Validation and test partitions use
those same statistics. Each partition must contain more than six rows.

HALF-RTU 配置按 `pf, pi, qf, qi` 排列，即首端有功潮流、有功注入、首端无功潮流、
无功注入。功率和电压幅值采用标幺值，电压角度采用弧度。数据按时间划分为
60%/20%/20%，窗口长度为 6，步长为 1，归一化统计量来自训练分区。

## Preparing Profiles and Measurements / 原始数据与量测生成

The manuscript describes UCI Electricity Load Diagrams 2011 2014 profiles and
Elia photovoltaic profiles, resampling from 15-minute to 5-minute resolution and
30% PV penetration. Obtain the load and PV profiles from their respective
providers and prepare the source tables below.

1. Prepare `data/raw_data/load.csv` and `pv.csv` with a `DateTime` column
   and numeric profile columns. Set the date ranges and selected columns in the
   preprocessing notebook to match the profiles.
2. Run `experiments/01_setup/prepare_profiles.ipynb` from the repository root to adapt the profiles to a
   selected grid and save the load/PV arrays. Its date ranges and channel
   selection must match the profiles you provide.
3. Use `data/generation.py` to generate measurements and state estimates:
   `PGAE_CASE=case30 python -m data.generation`.
4. Inspect OPF convergence, array shapes, noise settings, and time alignment
   before training.

完整数据需自行准备：按表格约定保存数组，确保量测、状态及负荷/光伏数据逐行时间一致。
数据准备 Notebook 中的日期与列选择按实际来源设置。主数据加载器支持
`PGAE_DATA_DIR`，预处理 Notebook 使用默认 `data/` 路径。

`gen_case` modifies the bundled PYPOWER grids. Use these modified grids
consistently for generation, model initialization, and attack construction.

## Noise Settings / 噪声设置

`define_mea_idx_noise` uses 0.01 p.u. RTU standard deviation and 0.001 for PMU
channels. The measurement-generation module saves this vector as
`noise_sigma.npy`. The profile notebook also provides a magnitude-dependent
calculation, `abs(z * noise_ratio_power) + 1e-3`. Model loaders read the saved
vector for the selected case.

`define_mea_idx_noise` 将 RTU 标准差设为 0.01 p.u.，PMU 通道设为 0.001。量测生成
模块将该向量保存为 `noise_sigma.npy`。原始数据 Notebook 还提供随量测幅值变化的
计算方式 `abs(z * noise_ratio_power) + 1e-3`。模型加载器读取所选系统的已保存向量。
