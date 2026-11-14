# Experiments in Paper Order / 按论文顺序组织的实验

Run shell commands below from the repository root. Notebooks locate that root
automatically when opened from inside the repository. Set `PGAE_CASE` before
imports and restart the kernel after changing it. Prepare the input arrays
and train the required models in the order below. Run `python smoke_test.py`
for a synthetic CPU check.

下列命令在项目根目录执行。Notebook 从项目内打开时会自动定位根目录。请在导入模块前
选择 `PGAE_CASE`，切换后重启内核。依次准备数据并训练实验所需模型。
`python smoke_test.py` 使用临时合成输入检查 CPU 运行。

## 01. Experimental Setup / 实验设置

Paper Section IV. Read [the data contract](../docs/DATA.md), then adapt
[prepare_profiles.ipynb](01_setup/prepare_profiles.ipynb) to your source load/PV
tables. It writes case arrays under `data/`; inspect the profile dates, column
selection, noise recipe, and OPF convergence before training. The generation
module can be invoked as `PGAE_CASE=case30 python -m data.generation`.

对应论文第四节。先阅读[数据格式说明](../docs/DATA.md)，再根据自己的负荷与光伏原始
表格调整预处理 Notebook。它默认在 `data/` 下生成各系统数组；训练前需确认日期、
列选择、噪声规则与最优潮流收敛情况。

## 02. Detection Performance / 检测性能

Train the two core reconstruction models:

先训练主检测模型和供定位器使用的 ConvAE：

```bash
python train.py --model PGAE-tau --case case30 --seed 0
python train.py --model ConvAE --case case30 --seed 0
```

| Entry / 入口 | Purpose / 用途 |
| --- | --- |
| [physics_guided.ipynb](02_detection/physics_guided.ipynb) | PGAE-tau detection evaluation / 主模型检测评估 |
| [data_driven.ipynb](02_detection/data_driven.ipynb) | Data-driven reconstruction evaluation, currently configured for ConvAE / 数据驱动评估，当前配置为 ConvAE |
| [ocsvm.ipynb](02_detection/ocsvm.ipynb) | One-class SVM evaluation / 单类 SVM 评估 |
| [graph_ae.ipynb](02_detection/graph_ae.ipynb) | GraphAE evaluation; requires PyTorch Geometric / 图自编码器评估，需额外图依赖 |
| [anomaly_transformer.ipynb](02_detection/anomaly_transformer.ipynb) | Anomaly Transformer evaluation / 异常 Transformer 评估 |
| [sota_gcn.ipynb](02_detection/sota_gcn.ipynb) | GCN evaluation / GCN 评估 |
| [evaluate_bdd.py](02_detection/evaluate_bdd.py) | BDD baseline / 坏数据检测基线 |

Baseline training modules are in `models/baselines/`. Set the case and load
the corresponding checkpoint before evaluation. The
[detection settings](../docs/REPRODUCIBILITY.md#detection-settings--检测设置)
list each method's score and threshold defaults.

基线训练模块位于 `models/baselines/`。评估时选择系统并加载对应权重，各方法的分数
与默认阈值见检测设置。

## 03. Measurement-Channel Localization / 量测通道定位

After training ConvAE, train the node prior and evaluate TopoLoc:

训练 ConvAE 后，再训练拓扑先验并评估 TopoLoc：

```bash
PGAE_CASE=case30 python experiments/03_localization/train_topoloc.py
PGAE_CASE=case30 python experiments/03_localization/evaluate_topoloc.py
```

`evaluate_topoloc.py` selects fusion strength by validation NDCG@15, then
evaluates test attack scenarios. It also reports residual-only and prior-only
controls and the validation fusion sweep. Outputs go to
`outputs/topoloc/<case>/`. Constants near the script tops define seeds, sample
counts, and attack scenarios. The default run uses seed 42 and up to 200
windows per validation/test partition.

评估程序先用验证集 NDCG@15 选择融合权重，再评估测试攻击场景，同时生成仅残差、
仅先验及融合权重扫描结果。输出位于 `outputs/topoloc/<case>/`。脚本顶部定义种子、
样本数和攻击场景，默认种子为 42，验证和测试分区各选取最多 200 个窗口。

[benchmark.ipynb](03_localization/benchmark.ipynb) calls
`evaluation/localization.py` to compare measurement-channel rankings. It writes
`outputs/detection5/<case>/localization_results.json`. Missing/incompatible
checkpoints are recorded in the output's `model_status` field.

定位基线 Notebook 调用共享评估模块，在 `model_status` 中记录各模型的加载状态。
完成基线、TopoLoc 与下一节对照实验后可合并结果：

```bash
PGAE_CASE=case30 python experiments/03_localization/merge_results.py
```

Merged output / 合并结果：`outputs/detection5_topoloc/<case>/localization_results.json`。

## 04. Ablation Study / 消融分析

```bash
PGAE_CASE=case30 python experiments/04_ablation/localization_controls.py
```

This script requires the trained localizer and its evaluation JSON. It disables
graph/temporal/residual context at inference, shuffles priors, and compares
fusion rules. All controls use the same trained checkpoint. The fusion-weight sweep is in
`03_localization/evaluate_topoloc.py`.

该脚本依赖已训练定位器及其评估 JSON，通过在推理时关闭上下文、打乱先验或改变融合
规则形成对照。所有对照共用同一训练权重。融合权重扫描
位于上一节的定位评估程序中。

`models/ablation/pgae_lstm.py` provides the PGAE-LSTM detector comparison.
For topology-bias sensitivity, set `LatentTopoAttention.lambda_tpo` in
`models/detection/pgae_tau.py`, then train and evaluate each setting.
Source locations are listed in [the paper-to-code map](../docs/PAPER_TO_CODE.md).

PGAE-LSTM 检测器对照位于 `models/ablation/pgae_lstm.py`。拓扑偏置敏感性实验通过设置
`LatentTopoAttention.lambda_tpo`，分别训练和评估各参数值。源码位置见论文与代码对照表。

## 05. Efficiency Analysis / 效率分析

[profile.ipynb](05_efficiency/profile.ipynb) profiles model parameter
counts and per-window inference latency; OCSVM timing requires its fitted
checkpoint. Install optional dependencies before running the complete notebook.
It uses random inputs for PyTorch timing and writes parameter counts and
per-window latency to `outputs/inference_profile.xlsx`.

效率 Notebook 统计模型参数量与单窗口推理耗时，PyTorch 计时使用随机输入，OCSVM
计时使用已训练模型。安装可选依赖后运行，输出为 `outputs/inference_profile.xlsx`。

## Diagnostics / 辅助诊断

`diagnostics/` contains Jacobian checks and PGAE residual visualizations.
Run the corresponding script or notebook to generate each analysis.

辅助诊断目录包含雅可比检查和 PGAE 残差可视化，运行相应脚本或 Notebook 生成分析结果。
