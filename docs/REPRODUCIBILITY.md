# Experiment Settings / 实验设置

## Environment / 运行环境

Use Python 3.13 with the core dependencies in `requirements.txt`. Install
`requirements-optional.txt` for notebooks, plotting, spreadsheet export, and
the GraphAE/PING baselines that use PyTorch Geometric.

使用 Python 3.13 和 `requirements.txt` 中的核心依赖。Notebook、统计绘图、表格导出
及使用 PyTorch Geometric 的 GraphAE/PING 基线依赖见 `requirements-optional.txt`。

Local CPU smoke checks passed on all three cases with Python 3.13.13,
PyTorch 2.12.1, NumPy 2.5.0, SciPy 1.17.1, PYPOWER 5.1.19, pandas 3.0.3,
matplotlib 3.10.9, scikit-learn 1.9.0, tqdm 4.67.3, and joblib 1.5.3.
The GitHub Actions workflow runs the IEEE 14-bus CPU smoke test on pushes
and pull requests.

上述版本为本地 CPU 检查环境。GitHub Actions 在推送和拉取请求时运行 IEEE 14 节点
CPU smoke test。

## Configuration / 配置

| Variable | Default | Purpose |
| --- | --- | --- |
| `PGAE_CASE` | `case30` | Select `case14`, `case30`, or `case57` before imports |
| `PGAE_DEVICE` | CUDA if available, otherwise CPU | PyTorch device for core models |
| `PGAE_DATA_DIR` | `<repository>/data` | Root of per-case input arrays |
| `PGAE_CHECKPOINT_DIR` | `<repository>/saved_model` | Root of per-case checkpoints |
| `PGAE_OUTPUT_DIR` | `<repository>/outputs` | Root of generated results |

Run documented commands from the repository root. Notebooks find the root
when opened from within an experiment subdirectory and set it as their working
directory. Restart the notebook kernel after changing the case. Main training,
localization scripts, and the smoke test also resolve their imports when invoked
by absolute path. Notebook checkpoint and result paths are relative to the
repository root. Core loaders and training scripts use the storage variables
above. `.gitignore` excludes generated data, checkpoints, and results.

Notebook 从项目内部打开时自动定位根目录，切换系统后重启内核。Notebook 的权重与
结果路径相对于根目录，核心加载器和训练脚本使用上表中的存储环境变量。

## Core Training and Localization / 核心训练与定位

```bash
python train.py --model PGAE-tau --case case30 --seed 0
python train.py --model ConvAE --case case30 --seed 0
PGAE_CASE=case30 python experiments/03_localization/train_topoloc.py
PGAE_CASE=case30 python experiments/03_localization/evaluate_topoloc.py
PGAE_CASE=case30 python experiments/04_ablation/localization_controls.py
```

Repeat with separate processes for the other systems. The detector checkpoint
is `saved_model/<case>/PGAE_tau.pt`; the localizer uses
`saved_model/<case>/ConvAE.pt` and saves only its prior network to
`saved_model/<case>/TopoLoc_ConvAE.pt`.

The localizer trains on normal windows with pseudo-corruptions. Its constants
(seed, subset sizes, batch size, patience, and epoch limit) are near the top of
`experiments/03_localization/train_topoloc.py`. The evaluation script selects the fusion
weight on validation scenarios by NDCG@15 and then evaluates test scenarios.
Generated training, fusion, and control results go to `outputs/topoloc/<case>/`.
Inference controls in `experiments/04_ablation/localization_controls.py` reuse
a fixed trained localizer and modify its inference inputs or fusion rule.

定位器在正常窗口上通过伪攻击学习节点先验。训练参数位于训练脚本顶部，评估程序根据
验证集 NDCG@15 选择融合权重。推理对照共用同一定位权重，分别改变输入信息或融合规则。

`experiments/03_localization/benchmark.ipynb` invokes the shared baseline benchmark and writes
`outputs/detection5/<case>/localization_results.json`. Train compatible baseline
checkpoints before running it. The output's `model_status` field records each
model's checkpoint path, loading status, and reason for any skipped entry.
After both that benchmark and TopoLoc evaluation/controls finish, run:

```bash
PGAE_CASE=case30 python experiments/03_localization/merge_results.py
```

The merged payload is written to
`outputs/detection5_topoloc/<case>/localization_results.json`.

基线定位结果记录在 `outputs/detection5/<case>/`，TopoLoc 结果记录在
`outputs/topoloc/<case>/`。完成两类评估及推理对照后，运行合并程序生成七模型结果。

## Data and Training Settings / 数据与训练设置

- Chronological train/validation/test split: 60%/20%/20%.
- Window length: 6; stride: 1.
- Normalization: per-channel min/max over the full training partition.
- PGAE-tau: one shared scalar temperature, a single-layer bidirectional LSTM,
  and normalized reconstruction MSE at the last time step.
- ConvAE: validation-loss early stopping and a dedicated `ConvAE.pt` checkpoint.
- TopoLoc prior: seed 42, up to 8,000 training windows and 1,600 validation
  windows, batch size 128, at most 20 epochs, patience 4.
- TopoLoc loss: attack-node cross-entropy with temperature 0.25 plus
  clean-window binary cross-entropy weighted by 0.1.

数据按时间划分为 60%/20%/20%，窗口长度为 6，步长为 1；归一化统计量来自整个训练
分区。ConvAE 根据验证损失早停并单独保存权重。TopoLoc 的训练种子、子集规模和损失
权重均在训练脚本中显式给出。

## Detection Settings / 检测设置

The manuscript reports Recall, F1-score, and MCC at FPR=5%. The table below
lists the supplied detection entries and their executable defaults.

论文在 FPR=5% 下报告 Recall、F1-score 和 MCC。各检测入口的默认评分与阈值设置如下：

| Entry / 入口 | Score / 分数 | Default threshold / 默认阈值 |
| --- | --- | --- |
| `physics_guided.ipynb` | Normalized MSE between the last-step reconstruction and the full input window, broadcast over time / 最后时刻重建值沿时间维广播后，与完整输入窗口计算归一化 MSE | Normal-validation 0.92 quantile, `quantile_idx=6` / 正常验证集 0.92 分位数 |
| `data_driven.ipynb` | ConvAE window reconstruction MSE / ConvAE 窗口重建均方误差 | Normal-validation 0.92 quantile / 正常验证集 0.92 分位数 |
| `ocsvm.ipynb` | Negative one-class SVM decision score / 单类 SVM 决策分数的相反数 | Normal-validation 0.92 quantile / 正常验证集 0.92 分位数 |
| `graph_ae.ipynb` | Normalized bus P/Q reconstruction MSE / 归一化节点 P/Q 重建均方误差 | Normal-validation 0.92 quantile / 正常验证集 0.92 分位数 |
| `sota_gcn.ipynb` | Sigmoid attack probability / Sigmoid 攻击概率 | Normal-validation 0.92 quantile / 正常验证集 0.92 分位数 |
| `anomaly_transformer.ipynb` | Association-weighted reconstruction error / 关联差异加权重建误差 | Validation quantile search with target normal-test FPR 0.05 / 按正常测试集 FPR 选择验证集分位数，目标为 0.05 |
| `evaluate_bdd.py` | BDD residual / 坏数据检测残差 | Clean-test 0.92 quantile / 正常测试集 0.92 分位数 |

Quantile arrays and selected indices are defined in each detection notebook.
Anomaly Transformer selects the candidate closest to 0.05 from below; when
all candidates exceed the target, it selects the smallest available FPR.

阈值分位数列表及选定索引位于各检测 Notebook 的评估器中。
Anomaly Transformer 选择最接近且不超过 0.05 的候选 FPR；若全部超过目标，则选择
FPR 最小的候选值。

## Localization Settings / 定位设置

The evaluation scripts use seed 42 and up to 200 windows per partition. Attack
cardinalities are 1, 2, and 3 buses, with intensities from 0.05 to 0.30. Main
scenarios use intensities 0.20 and 0.30. Fusion weights are selected by maximum
validation NDCG@15 from `{0, 0.2, 1/3, 0.5, 2/3, 0.8, 0.9, 0.95, 1}`.

定位评估使用种子 42，每个分区最多选取 200 个窗口。攻击节点数为 1、2、3，强度
范围为 0.05 至 0.30，主场景强度为 0.20 和 0.30。融合权重在上述集合中依据验证集
NDCG@15 最大值选择。

`localization_controls.py` evaluates graph, temporal, residual, mapping, and
fusion controls using the trained prior. Detector topology sensitivity is
configured through `LatentTopoAttention.lambda_tpo`; train and evaluate a
separate model for each selected value.

定位对照使用已训练先验评估拓扑、时间、残差、映射与融合设置。检测器的拓扑偏置强度
由 `LatentTopoAttention.lambda_tpo` 控制，每个参数值分别训练和评估。

## Smoke Test / 最小运行检查

`python smoke_test.py` checks data loading, normalization, the AC decoder
against independent NumPy equations, shared-temperature gradients, checkpoint
save/load, and frozen-backbone localization gradients. Use `--case case30` or
`--case case57` to select another IEEE system.

smoke test 覆盖数据加载、归一化、交流解码器与独立 NumPy 公式的一致性、共享温度
梯度、权重保存与加载，以及冻结骨干后的定位梯度。测试在临时目录生成输入和权重，
退出时自动清理。
