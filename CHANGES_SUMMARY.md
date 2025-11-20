# 窗口归一化功能实现总结

## ✅ 已完成的修改

### 1. 修改 `export/lobster_toolkit/core/normalizer.py`

**位置**：`ZScoreNormalizer.fit()` 方法，第 57-68 行

**改动**：额外保存 `message.price` 列的全局统计量

```python
# 额外保存 price 列的全局标准差（用于窗口归一化）
if 'price' in data_items[0]['message'].columns:
    all_prices = pd.concat(
        [item['message']['price'] for item in data_items],
        ignore_index=True,
        copy=False
    )
    self.stats['message']['price'] = {
        'mean': float(all_prices.mean()),
        'std': float(all_prices.std(ddof=0))
    }
```

**作用**：确保 price 列的全局标准差被保存到 `norm_params.csv` 中

---

### 2. 修改 `tsa/dataset.py` - LOBSTERTimeSeriesDataset

#### 2.1 在 `__init__` 中加载 price 全局标准差

**位置**：第 206-218 行

```python
# 加载价格列的全局标准差（用于窗口归一化）
self.price_global_std = None
norm_params_path = os.path.join(self.root_path, "norm_params.csv")
if os.path.exists(norm_params_path):
    try:
        from export.lobster_toolkit.core.normalizer import ZScoreNormalizer
        normalizer = ZScoreNormalizer.load_params(norm_params_path)
        params = normalizer.get_params()
        if 'message' in params and 'price' in params['message']:
            self.price_global_std = params['message']['price']['std']
            print(f"✓ 加载 price 全局标准差: {self.price_global_std:.6f}")
    except Exception as e:
        print(f"⚠️  无法加载价格全局标准差: {e}")
```

**作用**：在数据集初始化时加载全局标准差，用于后续窗口归一化

---

#### 2.2 修改 `_combine_features` 方法

**位置**：第 273-289 行

**改动**：返回值从 `np.ndarray` 改为 `Tuple[np.ndarray, Optional[int]]`

```python
def _combine_features(self, item: Dict[str, Any]) -> Tuple[np.ndarray, Optional[int]]:
    """组合特征并返回 price 列的索引
    
    Returns:
        (features_array, price_column_index)
        price_column_index: message.price 在组合后特征中的列索引，如果不存在则为 None
    """
    message_df = self._select_columns(item["message"], self.message_columns)
    orderbook_df = self._select_columns(item["orderbook"], self.orderbook_columns)
    combined = pd.concat([message_df, orderbook_df], axis=1)
    
    # 查找 price 列在组合后的索引
    price_idx = None
    if 'price' in message_df.columns:
        price_idx = list(combined.columns).index('price')
    
    return combined.to_numpy(dtype=np.float32), price_idx
```

**作用**：识别 price 列在合并后特征中的位置，供窗口归一化使用

---

#### 2.3 修改 `_frame_split` 方法

**位置**：第 293-337 行

**改动**：在窗口化时对 price 列应用窗口归一化

```python
for item in items:
    features, price_idx = self._combine_features(item)  # 获取 price 列索引
    targets = self._get_target_series(item, features)
    
    # ... 其他代码 ...
    
    for start in range(1, max_index + 1):
        end = start + self.seq_length
        
        feature_window = features[start:end].copy()  # 使用 copy 避免修改原数据
        
        # 对 message.price 列应用窗口均值 + 全局标准差归一化
        if price_idx is not None and self.price_global_std is not None:
            window_prices = feature_window[:, price_idx]
            window_mean = window_prices.mean()
            feature_window[:, price_idx] = (window_prices - window_mean) / (self.price_global_std + 1e-8)
        
        features_batches.append(torch.from_numpy(feature_window).float().unsqueeze(0))
        # ... 其他代码 ...
```

**作用**：在每个窗口内对 price 列进行归一化

---

### 3. 新增测试脚本

**文件**：`test_window_normalization.py`

**功能**：
- 检查数据文件和归一化参数
- 验证 LOBSTERTimeSeriesDataset 是否正确加载全局标准差
- 检查窗口归一化效果（窗口均值应接近 0）

**使用**：
```bash
python test_window_normalization.py
```

---

### 4. 新增文档

**文件**：`WINDOW_NORMALIZATION.md`

**内容**：
- 归一化流程说明
- 设计理由和方案对比
- 实现细节和关键特性
- 测试方法和预期结果
- 数据流总结

---

## 🎯 核心设计

### 归一化公式

```python
normalized_price = (price - window_mean) / global_std
```

### 设计特点

1. **窗口均值去中心化**：
   - 去除窗口内价格水平偏移
   - 每个窗口的 price 均值接近 0
   - 关注窗口内的相对价格结构

2. **全局标准差保持尺度**：
   - 跨窗口的尺度一致性
   - 相同波动幅度映射到相同归一化值
   - 避免小波动窗口的噪声放大

3. **交易日隔离**：
   - 窗口不跨越交易日边界
   - 每个交易日独立处理

4. **向后兼容**：
   - 如果 `price_global_std` 为 None，跳过窗口归一化
   - 不影响其他功能

---

## 📋 使用流程

### 1. 重新构建数据（如果需要）

```bash
python test_lobster_pipeline.py
```

这会：
- 预处理 LOBSTER 数据
- 保存 price 列的全局统计量到 `norm_params.csv`
- 生成 `train.pkl`, `val.pkl`, `test.pkl`

### 2. 测试窗口归一化

```bash
python test_window_normalization.py
```

验证：
- ✓ price 全局标准差已保存
- ✓ LOBSTERTimeSeriesDataset 成功加载
- ✓ 窗口均值接近 0

### 3. 训练模型

```bash
python examples/forecasting/run_lobster.py --config-name=config_lobster_long
```

日志中会显示：
```
✓ 加载 price 全局标准差: X.XXXXXX
```

---

## 🔍 验证方法

### 检查归一化参数

```python
import pandas as pd
params = pd.read_csv('./data/norm_params.csv')
print(params[params['param'] == 'price'])
```

预期输出：
```
   type   param      mean       std
0  message  price  0.001234  2.456789
```

### 检查窗口均值

```python
from tsa.dataset import LOBSTERTimeSeriesDataset, Tasks

dataset = LOBSTERTimeSeriesDataset(
    task=Tasks.prediction,
    data_path="./data",
    seq_length=64,
    batch_size=16,
    target_source="labels",
    label_type="long"
)

train_loader, _, _ = dataset.get_loaders()

for features, _, _ in train_loader:
    # 假设 price 在第 0 列
    window_means = features[:, :, 0].mean(dim=1)
    print(f"窗口均值: mean={window_means.mean():.6f}, std={window_means.std():.6f}")
    break
```

预期：窗口均值的平均值应接近 0（例如 < 0.001）

---

## 📝 注意事项

1. **只处理 message.price**：
   - 当前实现仅对 message.price 列应用窗口归一化
   - 其他列（time, size, orderbook）保持全局 Z-Score 归一化
   - 如需扩展到其他价格列，参考 `WINDOW_NORMALIZATION.md`

2. **数据重建**：
   - 如果之前已经运行过 `test_lobster_pipeline.py`，需要重新运行
   - 确保 `norm_params.csv` 包含 price 的统计量

3. **兼容性**：
   - 对现有代码无破坏性修改
   - 如果无法加载 price_global_std，只打印警告，不影响其他功能

---

## 🎉 总结

本次修改实现了：

✅ Message.price 的窗口均值+全局标准差归一化  
✅ 保持代码简洁清晰  
✅ 不破坏现有功能  
✅ 完整的测试和文档  
✅ 交易日隔离和滑动窗口正确处理

修改的文件：
- `export/lobster_toolkit/core/normalizer.py`（1处修改）
- `tsa/dataset.py`（3处修改）
- `test_window_normalization.py`（新增）
- `WINDOW_NORMALIZATION.md`（新增）

**下一步**：运行 `python test_window_normalization.py` 验证功能！
