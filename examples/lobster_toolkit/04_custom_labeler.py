"""示例 4: 自定义标签生成策略"""

import sys
sys.path.insert(0, '..')

from lobster_toolkit.core import (
    LOBSTERDataBuilder,
    MessagePreprocessor,
    OrderbookPreprocessor,
    StopLossLabelStrategy
)

# 方式 1: 使用自定义参数的构建器
print("=== 方式 1: 自定义参数构建器 ===")
builder = LOBSTERDataBuilder(
    split_rates=(0.7, 0.15, 0.15),  # 不同的划分比例
    stop_loss_ticks=10,              # 更大的止损阈值
    take_profit_ratio=2.0,           # 更高的止盈倍数
    tick_size=0.1,
    max_steps=128                    # 更长的观察窗口
)

# 注意: 实际使用时需要提供数据路径
# datasets = builder.build_dataset("../../data")

# 方式 2: 手动处理单个文件
print("\n=== 方式 2: 手动处理流程 ===")

# 加载数据（示例路径）
data_path = "../../data/GC"
import os
if os.path.exists(data_path):
    files = sorted([f for f in os.listdir(data_path) if '_message_' in f])
    if files:
        msg_file = os.path.join(data_path, files[0])
        ob_file = msg_file.replace('_message_', '_orderbook_')
        
        print(f"加载文件: {os.path.basename(msg_file)}")
        
        # 加载
        message_df = MessagePreprocessor.load_csv(msg_file)
        orderbook_df = OrderbookPreprocessor.load_csv(ob_file)
        
        # 预处理
        message_df = MessagePreprocessor.preprocess(message_df)
        orderbook_df = OrderbookPreprocessor.preprocess(orderbook_df)
        
        # 相对价格归一化
        if len(message_df) > 0:
            start_price = message_df['price'].iloc[0]
            message_df = MessagePreprocessor.normalize_price_relative(message_df)
            orderbook_df = OrderbookPreprocessor.normalize_price_relative(
                orderbook_df, start_price
            )
        
        # 使用不同的标签策略
        strategies = [
            ("保守策略", StopLossLabelStrategy(stop_loss_ticks=3, is_long=True)),
            ("标准策略", StopLossLabelStrategy(stop_loss_ticks=6, is_long=True)),
            ("激进策略", StopLossLabelStrategy(stop_loss_ticks=12, is_long=True)),
        ]
        
        print(f"\n比较不同标签策略 (共 {len(message_df)} 个样本):")
        for name, strategy in strategies:
            labels = strategy.generate(message_df, orderbook_df)
            positive_rate = labels.sum() / len(labels) * 100
            print(f"{name}: 正样本率 {positive_rate:.2f}% ({labels.sum()}/{len(labels)})")
else:
    print("数据路径不存在，跳过演示")

print("\n提示: 根据不同的交易策略和市场环境，可以调整标签生成参数")
