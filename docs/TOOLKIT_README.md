# LOBSTER Toolkit

一个优雅、简洁的 LOBSTER 高频交易数据处理工具包。

## 特性

- 🚀 **高效处理**：快速加载和处理 LOBSTER CSV 数据
- 🎯 **灵活标签**：支持多种标签生成策略（long/short）
- 🔧 **模块化设计**：核心逻辑与框架解耦
- 📦 **多框架支持**：适配 PyTorch、aeon 等主流框架
- 🧪 **易于测试**：完整的单元测试覆盖

## 快速开始

### 安装与环境

本工具包随 `time-series-autoencoder-agent` 仓库一起分发，不再单独打包发布为独立 Python 包。通常只需在仓库根目录创建虚拟环境并安装项目依赖：

```bash
# 在仓库根目录
uv venv               # 或 python -m venv .venv
uv pip sync pyproject.toml  # 或 pip install -r requirements.txt
```

在仓库根目录下运行示例脚本或测试时，可以直接 `import lobster_toolkit`。

### 基础使用

```python
from lobster_toolkit import build_dataset

# 一行代码构建数据集
datasets = build_dataset("./data")
# 返回: {'train': [...], 'val': [...], 'test': [...]}
```

### PyTorch 训练

```python
from lobster_toolkit.loaders.pytorch import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    data_path="./data",
    batch_size=32,
    seq_len=128,
    label_type='long'
)

# 训练循环
for features, labels in train_loader:
    # features: (batch, seq_len, n_features)
    # labels: (batch,)
    pass
```

### aeon 时间序列分类

```python
from lobster_toolkit.adapters.aeon import build_datasets

datasets = build_datasets(
    data_path="./data",
    window_size=64,
    step=32,
    label_kind='long'
)

# 使用 aeon 模型
from aeon.classification.convolution_based import MultiRocketClassifier

X_train = datasets['train'].X  # (n_instances, n_channels, window_size)
y_train = datasets['train'].y  # (n_instances,)

clf = MultiRocketClassifier()
clf.fit(X_train, y_train)
```

## 自定义使用

### 自定义标签策略

```python
from lobster_toolkit.core import LOBSTERDataBuilder

builder = LOBSTERDataBuilder(
    split_rates=(0.7, 0.15, 0.15),
    stop_loss_ticks=10,
    take_profit_ratio=2.0,
    tick_size=0.1,
    max_steps=64
)

datasets = builder.build_dataset("./data")
```

### 手动控制流程

```python
from lobster_toolkit.core import (
    MessagePreprocessor, 
    OrderbookPreprocessor,
    StopLossLabelStrategy
)

# 加载数据
message_df = MessagePreprocessor.load_csv("message.csv")
orderbook_df = OrderbookPreprocessor.load_csv("orderbook.csv")

# 预处理
message_df = MessagePreprocessor.preprocess(message_df)
orderbook_df = OrderbookPreprocessor.preprocess(orderbook_df)

# 生成标签
strategy = StopLossLabelStrategy(stop_loss_ticks=20)
labels = strategy.generate(message_df, orderbook_df)
```

## 项目结构

```
 time-series-autoencoder-agent/
├── lobster_toolkit/           # LOBSTER 工具包源码
│   ├── __init__.py           # 顶层 API
│   ├── core/                 # 核心模块（无框架依赖）
│   │   ├── builder.py        # 数据构建器
│   │   ├── preprocessor.py   # 预处理器
│   │   ├── labeler.py        # 标签生成器
│   │   └── normalizer.py     # 归一化工具
│   ├── loaders/              # 数据加载器
│   │   └── pytorch.py        # PyTorch DataLoader
│   └── adapters/             # 框架适配器
│       └── aeon.py           # aeon 适配器
├── examples/
│   └── lobster_toolkit/      # 使用示例脚本
├── tests/                    # 与工具包相关的测试
│   ├── test_toolkit_basic.py
│   └── test_lobster_pipeline.py
├── docs/
│   ├── TOOLKIT_README.md
│   └── QUICKSTART_TOOLKIT.md
└── pyproject.toml            # 整体项目的打包与依赖配置
```

## API 文档

### 核心模块

#### LOBSTERDataBuilder

主要的数据构建器，负责完整的数据处理流程。

```python
builder = LOBSTERDataBuilder(
    split_rates=(0.8, 0.1, 0.1),  # 数据集划分比例
    stop_loss_ticks=6,             # 止损阈值
    take_profit_ratio=1.5,         # 止盈倍数
    tick_size=0.1,                 # tick 大小
    max_steps=64                   # 前向观察窗口
)

datasets = builder.build_dataset("./data")
```

### 适配器

#### PyTorch Loader

```python
train_loader, val_loader, test_loader = get_dataloaders(
    data_path="./data",
    batch_size=32,
    seq_len=128,
    label_type='long',
    num_workers=0,
    shuffle_train=True
)
```

#### aeon Adapter

```python
datasets = build_datasets(
    data_path="./data",
    window_size=64,
    step=32,
    label_kind='long'
)
```

## 数据格式

### 输入格式

LOBSTER CSV 文件：
- `*_message_*.csv`: 6列（time, type, order_id, size, price, direction）
- `*_orderbook_*.csv`: 40列（10档 ask/bid 价格和挂单量）

### 输出格式

- `train.pkl / val.pkl / test.pkl`: 序列化的数据列表
- `norm_params.csv`: 归一化参数

## 依赖

核心依赖：
- numpy >= 1.21.0
- pandas >= 1.3.0

可选依赖：
- torch >= 1.10.0 (PyTorch 支持)
- aeon >= 0.5.0 (aeon 支持)

## 许可

MIT License

## 作者

Extracted from aeon-remote project
