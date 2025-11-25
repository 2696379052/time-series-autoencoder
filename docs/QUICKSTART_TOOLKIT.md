# 快速入门指南

## 安装

本工具包作为 `time-series-autoencoder-agent` 仓库的一部分分发，不再单独打包发布。通常只需在仓库根目录创建虚拟环境并安装项目依赖：

```bash
# 在仓库根目录
uv venv               # 或 python -m venv .venv
uv pip sync pyproject.toml  # 或 pip install -r requirements.txt
```

安装可选依赖：
```bash
# PyTorch 支持
pip install torch

# aeon 支持
pip install aeon scikit-learn
```

## 5 分钟上手

### 1. 构建数据集（30秒）

```python
from lobster_toolkit import build_dataset

# 一行代码完成数据处理
datasets = build_dataset("./data")
# 生成: train.pkl, val.pkl, test.pkl, norm_params.csv
```

### 2. PyTorch 训练（2分钟）

```python
from lobster_toolkit.loaders.pytorch import get_dataloaders
import torch
import torch.nn as nn

# 创建数据加载器
train_loader, val_loader, test_loader = get_dataloaders(
    data_path="./data",
    batch_size=32,
    seq_len=128,
    label_type='long'
)

# 定义简单模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# 训练
model = SimpleLSTM(input_size=train_loader.dataset[0][0].shape[1])
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} 完成")
```

### 3. aeon 时间序列分类（2分钟）

```python
from lobster_toolkit.adapters.aeon import build_datasets
from aeon.classification.convolution_based import MultiRocketClassifier

# 构建 aeon 格式数据
datasets = build_datasets(
    data_path="./data",
    window_size=64,
    step=32,
    label_kind='long'
)

# 训练分类器
clf = MultiRocketClassifier(n_jobs=-1, random_state=42)
clf.fit(datasets['train'].X, datasets['train'].y)

# 评估
score = clf.score(datasets['test'].X, datasets['test'].y)
print(f"测试集准确率: {score:.4f}")
```

## 常见用法

### 自定义参数

```python
from lobster_toolkit.core import LOBSTERDataBuilder

builder = LOBSTERDataBuilder(
    split_rates=(0.7, 0.15, 0.15),
    stop_loss_ticks=10,        # 调整止损阈值
    take_profit_ratio=2.0,     # 调整止盈倍数
    tick_size=0.1,
    max_steps=128              # 调整观察窗口
)

datasets = builder.build_dataset("./data")
```

### 手动控制每个步骤

```python
from lobster_toolkit.core import (
    MessagePreprocessor,
    OrderbookPreprocessor,
    StopLossLabelStrategy,
    ZScoreNormalizer
)

# 1. 加载
message_df = MessagePreprocessor.load_csv("path/to/message.csv")
orderbook_df = OrderbookPreprocessor.load_csv("path/to/orderbook.csv")

# 2. 预处理
message_df = MessagePreprocessor.preprocess(message_df)
orderbook_df = OrderbookPreprocessor.preprocess(orderbook_df)

# 3. 生成标签
strategy = StopLossLabelStrategy(stop_loss_ticks=6, is_long=True)
labels = strategy.generate(message_df, orderbook_df)

# 4. 归一化
normalizer = ZScoreNormalizer()
data_items = [{'message': message_df, 'orderbook': orderbook_df}]
normalizer.fit(data_items)
normalized_items = normalizer.transform(data_items)
```

## 目录结构

```
 time-series-autoencoder-agent/
├── lobster_toolkit/           # 主包源码
│   ├── __init__.py           # 顶层 API
│   ├── core/                 # 核心模块（无框架依赖）
│   ├── loaders/              # 数据加载器
│   └── adapters/             # 框架适配器
├── examples/
│   └── lobster_toolkit/      # 使用示例脚本
│       ├── 01_basic_usage.py
│       ├── 02_pytorch_loader.py
│       ├── 03_aeon_adapter.py
│       └── 04_custom_labeler.py
├── tests/                    # 与工具包相关的测试
│   ├── test_toolkit_basic.py
│   └── test_lobster_pipeline.py
├── docs/
│   ├── TOOLKIT_README.md     # 完整文档
│   └── QUICKSTART_TOOLKIT.md # 本文件
├── pyproject.toml            # 整体项目依赖配置
└── requirements.txt          # 由 uv 生成的锁定依赖
```

## 运行示例

```bash
cd examples/lobster_toolkit

# 基础使用
python 01_basic_usage.py

# PyTorch 数据加载
python 02_pytorch_loader.py

# aeon 时间序列分类
python 03_aeon_adapter.py

# 自定义标签策略
python 04_custom_labeler.py
```

## 运行测试

```bash
# 在仓库根目录
pytest tests -v
```

## 下一步

1. 查看 `README.md` 了解完整 API 文档
2. 查看 `examples/` 目录学习更多用法
3. 根据你的需求自定义标签策略和数据处理流程

## 故障排查

### 问题：找不到数据文件
确保数据目录结构正确：
```
data/
└── GC/
    ├── GC_2023-01-02_50_message.csv
    ├── GC_2023-01-02_50_orderbook.csv
    └── ...
```

### 问题：导入错误
通常需要保证：
```bash
# 1. 当前工作目录在仓库根目录
# 2. 已安装项目依赖（例如）
pip install -r requirements.txt
```

### 问题：PyTorch/aeon 不可用
安装可选依赖：
```bash
pip install torch  # PyTorch
pip install aeon   # aeon
```
