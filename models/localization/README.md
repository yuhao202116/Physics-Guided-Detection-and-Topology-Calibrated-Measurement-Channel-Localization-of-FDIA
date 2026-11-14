# TopoLoc-ConvAE

`topoloc_convae.py` combines a frozen normal-data ConvAE with a learned node
prior. The prior pools current measurements, temporal differences, and
reconstruction residuals, aggregates topology information, and maps node
weights back to measurement channels.

该定位器以冻结的 ConvAE 提供通道残差，结合节点局部、邻居和全局信息学习拓扑先验，
再将节点权重映射回量测通道。自监督扰动函数位于 `self_supervision.py`。

Prepare the arrays in [the data contract](../../docs/DATA.md), then run from the
repository root:

```bash
python train.py --model ConvAE --case case30
PGAE_CASE=case30 python experiments/03_localization/train_topoloc.py
PGAE_CASE=case30 python experiments/03_localization/evaluate_topoloc.py
PGAE_CASE=case30 python experiments/04_ablation/localization_controls.py
```

The frozen backbone is loaded from `saved_model/<case>/ConvAE.pt`. Training
saves the localizer to `saved_model/<case>/TopoLoc_ConvAE.pt`. Result files go
to `outputs/topoloc/<case>/`; all are ignored by Git.

Evaluation selects fusion strength using validation NDCG@15. The extra control
script disables inference inputs or changes fusion without retraining the
localizer. See [experiment settings](../../docs/REPRODUCIBILITY.md) for
parameters and the result-merging procedure.

Run `python smoke_test.py` for a synthetic CPU check of the model's forward
and backward computations.

先按数据说明准备数组并训练 ConvAE，再训练定位先验、选择融合权重并评估。额外对照脚本
共用已训练权重，分别改变推理输入与融合规则。根目录的 `smoke_test.py` 使用临时合成
输入检查 CPU 前向与反向计算。
