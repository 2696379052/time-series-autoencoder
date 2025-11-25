# 项目结构概览

本文档提供了 time-series autoencoder 项目的目录结构速览，帮助贡献者快速定位核心模块，并说明哪些组件来自外部项目。

## 顶层文件

- `README.md`、`LICENSE`、`requirements.txt` 和 `pyproject.toml`：分别用于项目说明、许可协议以及依赖与打包配置。
- `autoenc_architecture.png`：展示自编码器的整体模型结构示意图。

## 核心包：`tsa/`

项目的主要代码位于 `tsa` 包下：

- `__init__.py`：包初始化逻辑。
- `dataset.py`：多变量时间序列数据的加载与预处理工具。
- `model.py`：模型定义，包括编码器、解码器及注意力模块。
- `train.py`：训练流程封装，涵盖检查点与优化器配置。
- `eval.py`：评估函数与推理辅助工具。
- `utils.py`：通用工具函数集合。

## LOBSTER 工具包：`lobster_toolkit/`

- `lobster_toolkit/core/`：核心数据处理模块（预处理、打标、归一化等），与深度学习框架解耦。
- `lobster_toolkit/loaders/`：针对 PyTorch 等框架的数据加载器封装。
- `lobster_toolkit/adapters/`：与 aeon 等时间序列框架的适配层。

该工具包源自外部项目，但已经直接集成到本仓库中，用于为 `tsa` 提供 LOBSTER 数据预处理与标签生成能力。

## 示例

完整的端到端示例位于 `examples/` 目录：

- `examples/reconstruction/`：包含 `config.yaml` 与 `run_reconstruction.py`，演示重建任务实验。
- `examples/forecasting/`：包含 `config.yaml` 与 `run_forecasting.py`，展示预测任务的配置与脚本。
- `examples/lobster_toolkit/`：展示如何单独使用 `lobster_toolkit` 进行数据构建与模型训练。

## 测试

所有测试脚本集中在 `tests/` 目录：

- `test_toolkit_basic.py`、`test_lobster_pipeline.py` 等用于验证 LOBSTER 工具包的核心功能与端到端流程。
- `test_window_normalization.py` 用于验证 `lobster_toolkit` 与 `tsa.dataset` 之间的窗口归一化联动是否正确。

## 外部来源说明

`lobster_toolkit/` 源自外部项目（例如 aeon-remote），已经作为源码直接集成到本仓库中。用户不再需要单独引入 `export/` 目录或额外克隆外部仓库即可使用相关数据处理功能。
