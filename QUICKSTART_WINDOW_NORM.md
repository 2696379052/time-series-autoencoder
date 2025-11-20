# 🚀 窗口归一化功能快速启动

## 一句话总结

对 message.price 列实现了**窗口均值+全局标准差**归一化，保持代码简洁清晰。

---

## ✅ 验证修改

### 1. 重新构建数据（必须）

```bash
python test_lobster_pipeline.py
```

**检查输出**：确保最后显示"所有测试通过"

---

### 2. 测试窗口归一化

```bash
python test_window_normalization.py
```

**预期输出**：

```
步骤 2: 检查归一化参数
  ✓ Message.Price 全局统计量:
    均值: 0.XXXXXX
    标准差: X.XXXXXX

步骤 3: 测试 LOBSTERTimeSeriesDataset
  ✓ 数据集实例化成功
  ✓ 成功加载 price 全局标准差: X.XXXXXX

步骤 4: 检查窗口归一化效果
  各窗口的均值分布:
    平均窗口均值: ~0.000XXX  ← 应该接近0
    窗口均值标准差: 0.00XXXX
```

**关键检查点**：
- ✓ `price_global_std` 成功加载
- ✓ 平均窗口均值接近 0（说明窗口去均值成功）

---

### 3. 训练测试

```bash
python examples/forecasting/run_lobster.py --config-name=config_lobster_long
```

**检查日志**：启动时应显示
```
✓ 加载 price 全局标准差: X.XXXXXX
```

---

## 📋 改了什么

### 核心公式

```python
normalized_price = (price - window_mean) / global_std
```

### 修改的文件

1. **`export/lobster_toolkit/core/normalizer.py`** (1处)
   - 额外保存 message.price 全局统计量

2. **`tsa/dataset.py`** (3处)
   - 加载 price_global_std
   - 识别 price 列位置
   - 窗口化时应用归一化

---

## 🎯 关键特性

- ✅ 每个窗口的 price 均值 ≈ 0
- ✅ 用全局标准差保持尺度一致性
- ✅ 窗口不跨越交易日
- ✅ 步长为1的滑动窗口
- ✅ 向后兼容（如果加载失败只警告）

---

## ❓ 常见问题

**Q: 为什么要用窗口均值+全局标准差？**
A: 既去除窗口内价格水平偏移，又保持跨窗口的尺度可比性。详见 `WINDOW_NORMALIZATION.md`

**Q: 其他价格列（ask_price, bid_price）呢？**
A: 当前只处理 message.price，保持简洁。如需扩展参考文档。

**Q: 如果测试失败怎么办？**
A: 
1. 确保已重新运行 `test_lobster_pipeline.py`
2. 检查 `data/norm_params.csv` 是否包含 price 行
3. 查看完整错误信息

---

## 📚 详细文档

- **实现细节**：`CHANGES_SUMMARY.md`
- **设计说明**：`WINDOW_NORMALIZATION.md`
- **完整流程**：LOBSTER 数据处理流程

---

**完成！** 🎉

现在可以训练模型了，窗口归一化会自动应用于 message.price 列。
