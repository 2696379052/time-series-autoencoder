# 项目结构概览

本文档提供了 time-series autoencoder 项目的目录结构速览，帮助贡献者快速定位核心模块，并说明哪些组件来自外部依赖。

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

## 示例

完整的端到端示例位于 `examples/` 目录：

- `examples/reconstruction/`：包含 `config.yaml` 与 `run_reconstruction.py`，演示重建任务实验。
- `examples/forecasting/`：包含 `config.yaml` 与 `run_forecasting.py`，展示预测任务的配置与脚本。

## 外部数据处理模块

本仓库依赖来自外部项目的 `export/` 目录以提供部分数据处理功能。需要复现实验时，请从相应项目引入该目录或将其作为依赖添加。
