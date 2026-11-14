# Repository References / 参考仓库

The following repositories provide examples of method presentation,
quick-start instructions, and source organization. Upstream code credits
are listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

以下仓库提供方法展示、快速运行与代码组织的参考。上游代码来源见
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

| Repository / 仓库 | Conventions used here / 本仓库借鉴方式 |
| --- | --- |
| [Anomaly Transformer](https://github.com/thuml/Anomaly-Transformer) | Introduce the method with an architecture image, then give a short runnable entry and data preparation instructions. / 用结构图介绍方法，再提供简短运行入口与数据说明。 |
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | Separate model implementations from examples and make installation/dependency boundaries explicit. / 区分模型实现与示例入口，清楚标注安装及可选依赖。 |

PGAE-tau additionally uses numbered experiment directories to follow this
paper's own order: setup, detection, localization, ablation, efficiency.
The bilingual descriptions and all four README figures refer to the
associated PGAE-tau manuscript.

本项目在此基础上，以编号实验目录对应本论文的设置、检测、定位、消融与效率分析。
中英文方法说明及四张 README 配图均针对本项目论文。
