# 更新日志

## [0.1.0] - 2024-11-05

### 新增
- ✅ 核心数据处理模块
  - `LOBSTERDataBuilder`: 完整的数据构建流程
  - `MessagePreprocessor`: Message 数据预处理
  - `OrderbookPreprocessor`: Orderbook 数据预处理
  - `StopLossLabelStrategy`: 止损/止盈标签生成
  - `ZScoreNormalizer`: Z-score 归一化

- ✅ 框架适配器
  - `PyTorch DataLoader`: PyTorch 训练支持
  - `aeon Adapter`: aeon 时间序列分类支持

- ✅ 文档和示例
  - README.md: 完整的 API 文档
  - QUICKSTART.md: 快速入门指南
  - 4 个使用示例

### 修复
- 🐛 修复 Python 3.8 类型提示兼容性问题
  - 使用 `Tuple` 替代 `tuple[...]` 语法
  
- 🐛 修复 CSV 格式以匹配原项目
  - 从 `type,param,stat,value` 改为 `type,param,mean,std`
  
- 🐛 移除未使用的导入
  - 移除 `pytorch.py` 中未使用的 `ZScoreNormalizer` 导入

### 改进
- ✨ 添加边界检查
  - `ZScoreNormalizer.fit()` 检查空数据
  
- ✨ 使数据目录可配置
  - 添加 `data_dir_name` 参数，默认 "GC"
  
- ✨ 改进错误消息
  - 提供更详细的错误信息和解决建议
  
- 📝 改进文档
  - 更详细的参数说明
  - 添加使用示例

## 已知问题

暂无

## 下一版本计划 [0.2.0]

- [ ] 添加更多单元测试
- [ ] 添加性能基准测试
- [ ] 支持更多标签策略
- [ ] 添加数据可视化工具
- [ ] 支持并行处理
