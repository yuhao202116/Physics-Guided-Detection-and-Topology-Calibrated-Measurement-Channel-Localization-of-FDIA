# Paper-to-Code Map / 论文与代码对照

The tables follow the manuscript's section order and link its methodology
and experiments to source files. Paths are relative to the repository root.

下表按论文顺序对应方法模块、实验入口与源码位置，路径均相对于仓库根目录。

## III. Methodology / 方法

| Paper subsection / 论文小节 | Code / 代码 | Notes / 说明 |
| --- | --- | --- |
| III-A. Detection and Localization Workflow | `train.py`; `experiments/02_detection/`; `experiments/03_localization/` | Dedicated detection and localization entries / 检测与定位分别设有入口。 |
| III-B-1. Temporal Feature Encoding | `models/detection/pgae_tau.py::TimeCompressor` | Bidirectional LSTM / 双向 LSTM。 |
| III-B-2. Spatial-Topology Attention | Same file: `LatentTopoAttention`, `PI_Encoder` | Soft adjacency bias, all bus pairs participate, shared scalar tau. / 软拓扑偏置，所有节点对参与计算，共享标量温度。 |
| III-B-3. Voltage-State Projection and Flat-Start Initialization | Same file: `FeatureCompressor`, `initialize_flat_start`, `init_weights_PI` | Separate real/imaginary heads / 实虚部独立输出头。 |
| III-B-4. Fixed AC State-to-Measurement Decoder | Same file: `PowerFlowPhysics`, `PI_Decoder`; `physics/state_estimation.py`; `configs/config_mea_idx.py` | Fixed grid matrices and measurement selection / 固定电网矩阵及量测选择。 |
| III-B-5. Reconstruction Training and Attack Alarm | Same file: `PGAE_tau.fit`; `train.py`; `experiments/02_detection/physics_guided.ipynb` | Reconstruction MSE training and quantile-based evaluation / 重建均方误差训练与分位数阈值评估。 |
| III-C-1. ConvAE Channel-Residual Extraction | `models/baselines/convae.py::Conv_AE`; `models/localization/topoloc_convae.py::TopoLocConvAE` | ConvAE is frozen during prior training / 先验训练时冻结 ConvAE。 |
| III-C-2. Node-Topology Prior Learning | `models/localization/topoloc_convae.py::NodeTopologyPrior` | Eight pooled features, local/neighbor/global fusion / 八维汇聚特征与三类上下文融合。 |
| III-C-3. Measurement-Space Mapping and Residual Calibration | Same file: `bus_measurement_matrix`, `NodeTopologyPrior.to_measurements`, `TopoLocConvAE.forward`; fusion in `experiments/03_localization/evaluate_topoloc.py` | Node prior to channel prior, multiplicative residual calibration / 节点到通道先验映射与乘法校准。 |
| III-C-4. Self-Supervised Training with Pseudo-Attacks | `models/localization/self_supervision.py::corrupt`; `experiments/03_localization/train_topoloc.py::run_epoch` | Normal windows, surrogate node targets, clean-window loss / 正常窗口、代理节点标签与正常样本损失。 |

## IV. Experimental Setup / 实验设置

| Topic / 内容 | Code / 代码 |
| --- | --- |
| IEEE cases and measurement setup / 系统与量测配置 | `configs/config.py`, `configs/config_mea_idx.py`, `data/generation.py` |
| Profile preparation / 负荷与光伏准备 | `experiments/01_setup/prepare_profiles.ipynb`; see [DATA.md](DATA.md) |
| Chronological split and normalization / 时间划分与归一化 | `data/dataset.py`, `data/loaders.py` |
| FDI construction and state estimation / FDI 构造与状态估计 | `physics/attacks.py`, `physics/state_estimation.py` |
| Model configuration and storage / 网络与存储设置 | `configs/nn_setting.py`, `configs/paths.py` |
| Detection metrics / 检测指标 | `experiments/02_detection/` notebooks and BDD script |
| Localization relevance and ranking metrics / 定位相关性与排序指标 | `evaluation/localization.py::_build_attack_scenario`, `_metrics_for_sample`, `_aggregate_metrics` |
| Compared methods / 比较方法 | `models/baselines/`; optional PyTorch Geometric for GraphAE and PING |

## V. Experimental Results / 实验结果

| Paper subsection / 论文小节 | Source / 源码 | Execution / 运行方式 |
| --- | --- | --- |
| V-A. Detection Performance | `experiments/02_detection/` | Per-method notebooks; score and threshold defaults are listed in [experiment settings](REPRODUCIBILITY.md). / 按方法运行 Notebook，分数与默认阈值见实验设置。 |
| V-B. Measurement-Channel Localization | `experiments/03_localization/benchmark.ipynb`, `train_topoloc.py`, `evaluate_topoloc.py`, `merge_results.py` | Main scenarios and intensity sweeps; constants define sample counts and a single seed. / 主场景与强度扫描，样本数与单种子由脚本常量控制。 |
| V-C-1. Component Ablation | `models/ablation/pgae_lstm.py`; residual/prior/fusion comparisons in `evaluate_topoloc.py`; `experiments/04_ablation/localization_controls.py` | PGAE-LSTM comparison and fixed-checkpoint localization controls / PGAE-LSTM 对照与使用固定权重的定位对照。 |
| V-C-2. Sensitivity to Topology-Bias Strength | `LatentTopoAttention.lambda_tpo` in `models/detection/pgae_tau.py` | Set each bias strength in the model, then train and evaluate it separately / 在模型中设置各偏置强度，分别训练和评估。 |
| V-C-3. Sensitivity to Fusion Weight | `experiments/03_localization/evaluate_topoloc.py` | Validation NDCG@15 selects the weight; AUPRC is also recorded. / 根据验证集 NDCG@15 选权重，同时记录 AUPRC。 |
| V-D. Efficiency Analysis | `experiments/05_efficiency/profile.ipynb` | Parameter count and per-window inference timing / 参数量与单窗口推理计时。 |

## Smoke Test / 最小运行检查

`smoke_test.py` checks synthetic inputs, core physics equations, gradients,
and checkpoint save/load. See [the experiment guide](../experiments/README.md)
for the run order and [experiment settings](REPRODUCIBILITY.md) for defaults.

`smoke_test.py` 检查合成输入、核心物理计算、梯度和权重保存与加载。
实验顺序见实验指南，默认参数见实验设置。
