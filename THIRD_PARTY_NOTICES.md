# Third-Party Credits / 第三方来源

The project uses the following upstream implementations and libraries.
Source files retain their author and upstream references.

本项目使用以下上游实现与依赖，源码中保留了作者及来源信息。

- `physics/state_estimation.py` credits **W XU** and identifies
  `https://github.com/xuwkk/steady-state-power-system` as its upstream.
  It also references MATPOWER, MATPOWER-SE, and PYPOWER.
- `training/early_stopping.py` identifies the early-stopping implementation from
  `https://github.com/Bjarten/early-stopping-pytorch` as its source.
- `models/baselines/lstmae.py` identifies
  `https://github.com/hellojinwoo/TorchCoder/blob/master/autoencoders/rae.py`
  as the source of its recurrent autoencoder implementation.
- `models/baselines/anomaly_transformer.py` adapts Xu et al.'s Anomaly
  Transformer and references `https://github.com/thuml/Anomaly-Transformer`.
- GraphAE uses the ChebConv implementation provided by the separately
  installed PyTorch Geometric package.
- Standard IEEE test cases are obtained at runtime from the separately
  installed PYPOWER package.

The method illustrations are supplied with the associated manuscript; their
source filenames and positions are documented in [the figure notes](docs/assets/README.md).
Repositories consulted for documentation organization are listed separately in
[repository references](docs/REPOSITORY_REFERENCES.md).

论文配图来源见配图说明，文档组织参考见参考仓库。
