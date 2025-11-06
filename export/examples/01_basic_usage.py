"""示例 1: 基础使用 - 构建数据集"""

import sys
sys.path.insert(0, '..')

from lobster_toolkit import build_dataset

# 最简单的使用方式
datasets = build_dataset("../../data")

print(f"训练集: {len(datasets['train'])} 天")
print(f"验证集: {len(datasets['val'])} 天")
print(f"测试集: {len(datasets['test'])} 天")

# 查看第一天的数据
first_day = datasets['train'][0]
print(f"\n第一天数据:")
print(f"日期: {first_day['date']}")
print(f"Message 形状: {first_day['message'].shape}")
print(f"Orderbook 形状: {first_day['orderbook'].shape}")
print(f"Long 标签: {first_day['labels_long'].sum()} / {len(first_day['labels_long'])}")
print(f"Short 标签: {first_day['labels_short'].sum()} / {len(first_day['labels_short'])}")
