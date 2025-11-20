# Message.Price 窗口归一化说明

## 📋 概述

本项目对 LOBSTER 数据的 `message.price` 列实现了**窗口均值+全局标准差归一化**，与其他特征的全局 Z-Score 归一化策略不同。

---

## 🔍 归一化流程

### 1. 预处理阶段（已存在）

在 `export/lobster_toolkit/core/preprocessor.py` 中：

```python
# 每个交易日的 price 已经减去当日开盘价
start_price = msg_proc['price'].iloc[0]
df['price'] -= start_price  # 相对价格
```

**结果**：每天的 price 值变成"相对于开盘价的偏差"

### 2. 全局归一化阶段（已存在）

在 `export/lobster_toolkit/core/normalizer.py` 中：

```python
# 对 time, size 等列进行全局 Z-Score 归一化
normalized = (value - global_mean) / global_std
```

**但是**：price 列在这一步**不进行归一化**，而是保存其全局标准差供后续使用。

### 3. 窗口归一化阶段（新增功能）

在 `tsa/dataset.py` 的 `LOBSTERTimeSeriesDataset._frame_split` 中：

```python
# 对每个窗口的 price 列
window_prices = feature_window[:, price_idx]
window_mean = window_prices.mean()
feature_window[:, price_idx] = (window_prices - window_mean) / price_global_std
```

**公式**：
```
normalized_price = (price - window_mean) / global_std
```

---

## 💡 设计理由

### 为什么用窗口均值？

- **去除窗口内价格水平偏移**：不同窗口可能处于不同的价格区间
- **关注窗口内的价格结构**：模型学习的是"相对于窗口中心价格的偏离"

### 为什么用全局标准差？

- **保持尺度一致性**：一个 5 tick 的上涨在任何窗口中都映射到相似的归一化值
- **跨窗口可比性**：不同波动幅度的窗口仍然可以比较
- **避免噪声放大**：小波动窗口不会被过度缩放

### 与其他方案的对比

| 方案 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| **当前方案** | `(x - window_mean) / global_std` | 尺度一致、可比性强 | - |
| 窗口 Z-Score | `(x - window_mean) / window_std` | 突出局部模式 | 跨窗口不可比、放大噪声 |
| 相对首值 | `(x - x[0]) / x[0]` | 简单直观 | 依赖起点选择 |
| 全局 Z-Score | `(x - global_mean) / global_std` | 最一致 | 不适合已相对化的价格 |

---

## 🔧 实现细节

### 代码修改点

1. **`export/lobster_toolkit/core/normalizer.py`**
   - 在 `fit()` 方法中额外保存 `message.price` 的全局统计量

2. **`tsa/dataset.py`**
   - `LOBSTERTimeSeriesDataset.__init__`：加载 price 全局标准差
   - `_combine_features`：返回 price 列在合并特征中的索引
   - `_frame_split`：对每个窗口的 price 列应用窗口归一化

### 关键特性

- ✅ **交易日隔离**：窗口不会跨越不同交易日
- ✅ **滑动窗口**：步长为 1，相邻窗口重叠 `seq_length - 1` 个时间步
- ✅ **原地归一化**：在窗口化时动态归一化，不修改原始数据
- ✅ **向后兼容**：如果 `price_global_std` 为 None，跳过窗口归一化

---

## 🧪 测试方法

运行测试脚本：

```bash
python test_window_normalization.py
```

**测试内容**：

1. ✓ 检查数据文件是否存在
2. ✓ 验证 price 全局标准差是否被正确保存和加载
3. ✓ 创建 LOBSTERTimeSeriesDataset 并加载数据
4. ✓ 检查窗口归一化后的 price 统计特性：
   - 各窗口的均值应该接近 0
   - 标准差应该在合理范围内

**预期输出示例**：

```
步骤 2: 检查归一化参数
  ✓ Message.Price 全局统计量:
    均值: 0.000123
    标准差: 2.456789

步骤 3: 测试 LOBSTERTimeSeriesDataset
  ✓ 数据集实例化成功
  ✓ 成功加载 price 全局标准差: 2.456789

步骤 4: 检查窗口归一化效果
  各窗口的均值分布:
    平均窗口均值: -0.000012
    窗口均值标准差: 0.000345
    (理论上应该接近0，因为做了窗口去均值)
```

---

## 📊 数据流总结

```
原始 LOBSTER CSV
    ↓
预处理：price -= 开盘价
    ↓
全局归一化：time, size (Z-Score)
保存：price 全局 std
    ↓
保存 pkl 文件
    ↓
窗口化：
  - 滑动窗口提取
  - price: (x - window_mean) / global_std
  - 其他: 直接使用全局归一化值
    ↓
DataLoader → 模型训练
```

---

## ⚙️ 配置

无需特殊配置，功能自动启用：

- 如果 `norm_params.csv` 中包含 `message.price` 的统计量，自动加载
- 如果加载失败或不存在，打印警告但不影响其他功能
- 可以在日志中看到：`✓ 加载 price 全局标准差: X.XXXXXX`

---

## 🔮 未来扩展

如需对其他价格列（如 `ask_price_*`, `bid_price_*`）也应用窗口归一化：

1. 修改 `_combine_features` 识别所有价格列
2. 在 `normalizer.py` 中保存这些列的全局标准差
3. 在 `_frame_split` 中对所有价格列应用窗口归一化

当前实现保持简洁，只处理 `message.price`，便于理解和调试。

---

## 📝 相关文件

- 核心实现：`tsa/dataset.py::LOBSTERTimeSeriesDataset`
- 归一化器：`export/lobster_toolkit/core/normalizer.py::ZScoreNormalizer`
- 预处理器：`export/lobster_toolkit/core/preprocessor.py`
- 测试脚本：`test_window_normalization.py`

---

**版本**: v1.0  
**最后更新**: 2025-01-15
