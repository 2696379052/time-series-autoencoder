"""LOBSTER 数据处理流程综合测试脚本"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import pytest

# 添加 export 目录到路径
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'export'))

from lobster_toolkit.core.preprocessor import MessagePreprocessor, OrderbookPreprocessor, preprocess_pair
from lobster_toolkit.core.labeler import generate_dual_labels, StopLossLabelStrategy
from lobster_toolkit.core.normalizer import ZScoreNormalizer
from lobster_toolkit.core.builder import LOBSTERDataBuilder


def print_section(title):
    """打印测试章节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_file_discovery():
    """测试1: 文件发现和配对"""
    print_section("测试 1: 文件发现和配对")
    
    data_path = "./data"
    builder = LOBSTERDataBuilder()
    
    try:
        file_pairs = builder._discover_files(data_path)
        print(f"✓ 发现 {len(file_pairs)} 个文件对")
        
        if file_pairs:
            print(f"\n前3个文件对:")
            for i, pair in enumerate(file_pairs[:3], 1):
                print(f"  {i}. 日期: {pair['date']}")
                print(f"     Message:   {os.path.basename(pair['message'])}")
                print(f"     Orderbook: {os.path.basename(pair['orderbook'])}")
        
        return True, file_pairs
    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False, []


def test_data_loading(file_pairs):
    """测试2: 数据加载"""
    print_section("测试 2: 数据加载")
    
    if not file_pairs:
        print("✗ 没有可用的文件对")
        return False, None, None
    
    try:
        pair = file_pairs[0]
        print(f"加载文件: {pair['date']}")
        
        # 加载 message
        start = time.time()
        message_df = MessagePreprocessor.load_csv(pair['message'])
        msg_time = time.time() - start
        print(f"✓ Message 加载完成: {message_df.shape} - {msg_time:.3f}秒")
        print(f"  列: {list(message_df.columns)}")
        print(f"  内存: {message_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # 加载 orderbook
        start = time.time()
        orderbook_df = OrderbookPreprocessor.load_csv(pair['orderbook'])
        ob_time = time.time() - start
        print(f"✓ Orderbook 加载完成: {orderbook_df.shape} - {ob_time:.3f}秒")
        print(f"  列数: {len(orderbook_df.columns)}")
        print(f"  内存: {orderbook_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # 数据预览
        print(f"\nMessage 前3行:")
        print(message_df.head(3))
        print(f"\nOrderbook 前3行 (部分列):")
        print(orderbook_df.iloc[:3, :8])
        
        return True, message_df, orderbook_df
    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False, None, None


def test_preprocessing(message_df, orderbook_df):
    """测试3: 数据预处理"""
    print_section("测试 3: 数据预处理")
    
    if message_df is None or orderbook_df is None:
        print("✗ 没有可用的数据")
        return False, None, None
    
    try:
        start = time.time()
        msg_proc, ob_proc = preprocess_pair(message_df, orderbook_df)
        proc_time = time.time() - start
        
        print(f"✓ 预处理完成 - {proc_time:.3f}秒")
        print(f"  Message 形状: {msg_proc.shape}")
        print(f"  Orderbook 形状: {ob_proc.shape}")
        
        # 检查数据类型
        print(f"\nMessage 数据类型:")
        print(msg_proc.dtypes)
        
        print(f"\nMessage 预处理后数据统计:")
        print(msg_proc.describe())
        
        # 检查是否有 NaN
        msg_nan = msg_proc.isna().sum().sum()
        ob_nan = ob_proc.isna().sum().sum()
        print(f"\nNaN 检查:")
        print(f"  Message: {msg_nan} NaN值")
        print(f"  Orderbook: {ob_nan} NaN值")
        
        if msg_nan > 0 or ob_nan > 0:
            print(f"  ⚠ 警告: 存在 NaN 值")
        
        return True, msg_proc, ob_proc
    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False, None, None


def test_label_generation(message_df, orderbook_df):
    """测试4: 标签生成"""
    print_section("测试 4: 标签生成")
    
    if message_df is None or orderbook_df is None:
        print("✗ 没有可用的数据")
        return False, None, None
    
    try:
        start = time.time()
        labels_long, labels_short = generate_dual_labels(
            message_df, orderbook_df,
            stop_loss_ticks=6,
            take_profit_ratio=1.5,
            tick_size=0.1,
            max_steps=64
        )
        label_time = time.time() - start
        
        print(f"✓ 标签生成完成 - {label_time:.3f}秒")
        print(f"  Long 标签形状: {labels_long.shape}")
        print(f"  Short 标签形状: {labels_short.shape}")
        
        # 标签分布
        long_pos = (labels_long == 1).sum()
        short_pos = (labels_short == 1).sum()
        total = len(labels_long)
        
        print(f"\n标签分布:")
        print(f"  Long - 止盈: {long_pos} ({long_pos/total*100:.1f}%), "
              f"止损: {total-long_pos} ({(total-long_pos)/total*100:.1f}%)")
        print(f"  Short - 止盈: {short_pos} ({short_pos/total*100:.1f}%), "
              f"止损: {total-short_pos} ({(total-short_pos)/total*100:.1f}%)")
        
        # 检查标签平衡性
        long_ratio = long_pos / total
        short_ratio = short_pos / total
        
        if long_ratio < 0.3 or long_ratio > 0.7:
            print(f"  ⚠ 警告: Long标签不平衡 ({long_ratio:.1%})")
        if short_ratio < 0.3 or short_ratio > 0.7:
            print(f"  ⚠ 警告: Short标签不平衡 ({short_ratio:.1%})")
        
        return True, labels_long, labels_short
    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False, None, None


def test_normalization(file_pairs):
    """测试5: 归一化"""
    print_section("测试 5: 归一化")
    
    if not file_pairs:
        print("✗ 没有可用的文件对")
        return False
    
    try:
        # 加载前2个文件用于测试
        test_items = []
        for pair in file_pairs[:2]:
            message_df = MessagePreprocessor.load_csv(pair['message'])
            orderbook_df = OrderbookPreprocessor.load_csv(pair['orderbook'])
            msg_proc, ob_proc = preprocess_pair(message_df, orderbook_df)
            
            test_items.append({
                'date': pair['date'],
                'message': msg_proc,
                'orderbook': ob_proc
            })
        
        print(f"使用 {len(test_items)} 个文件进行归一化测试")
        
        # 归一化
        start = time.time()
        normalizer = ZScoreNormalizer()
        normalizer.fit(test_items)
        normalized_items = normalizer.transform(test_items)
        norm_time = time.time() - start
        
        print(f"✓ 归一化完成 - {norm_time:.3f}秒")
        
        # 检查归一化参数
        params = normalizer.get_params()
        print(f"\n归一化参数:")
        print(f"  Message 参数:")
        for col, stats in params['message'].items():
            print(f"    {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        print(f"  Orderbook 参数 (前3列):")
        for col, stats in list(params['orderbook'].items())[:3]:
            print(f"    {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # 检查归一化后的数据
        first_item = normalized_items[0]
        print(f"\n归一化后数据统计:")
        print(first_item['message'].describe())
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试6: 完整数据构建流程"""
    print_section("测试 6: 完整数据构建流程")
    
    try:
        # 使用小规模参数加速测试
        builder = LOBSTERDataBuilder(
            split_rates=(0.6, 0.2, 0.2),
            stop_loss_ticks=6,
            take_profit_ratio=1.5,
            tick_size=0.1,
            max_steps=64,
            data_dir_name="GC"
        )
        
        print("开始构建数据集...")
        start = time.time()
        
        datasets = builder.build_dataset("./data")
        
        build_time = time.time() - start
        
        print(f"\n✓ 数据集构建完成 - {build_time:.2f}秒")
        print(f"\n数据集统计:")
        for split_name in ['train', 'val', 'test']:
            if split_name in datasets:
                items = datasets[split_name]
                print(f"  {split_name}: {len(items)} 天")
                
                if items:
                    first_item = items[0]
                    print(f"    示例形状 - Message: {first_item['message'].shape}, "
                          f"Orderbook: {first_item['orderbook'].shape}")
        
        # 验证保存的文件
        print(f"\n检查保存的文件:")
        for fname in ['train.pkl', 'val.pkl', 'test.pkl', 'norm_params.csv']:
            fpath = os.path.join("./data", fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath) / 1024 / 1024
                print(f"  ✓ {fname}: {size:.2f} MB")
            else:
                print(f"  ✗ {fname}: 未找到")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False


def test_dataloader():
    """测试7: PyTorch DataLoader"""
    print_section("测试 7: PyTorch DataLoader")
    
    try:
        from lobster_toolkit.loaders.pytorch import get_dataloaders
        
        print("创建 DataLoader...")
        train_loader, val_loader, test_loader = get_dataloaders(
            data_path="./data",
            batch_size=16,
            seq_len=64,
            label_type='long',
            shuffle_train=False
        )
        
        print(f"✓ DataLoader 创建成功")
        print(f"  训练集: {len(train_loader.dataset)} 个序列")
        print(f"  验证集: {len(val_loader.dataset)} 个序列")
        print(f"  测试集: {len(test_loader.dataset)} 个序列")
        
        # 测试一个 batch
        print(f"\n获取一个 batch...")
        for features, labels in train_loader:
            print(f"  Features: {features.shape} - dtype: {features.dtype}")
            print(f"  Labels: {labels.shape} - dtype: {labels.dtype}")
            print(f"  特征范围: [{features.min():.4f}, {features.max():.4f}]")
            print(f"  标签分布: {labels.unique(return_counts=True)}")
            break
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        traceback.print_exc()
        return False


# 将上面的分步函数标记为“帮助函数”，避免被 pytest 直接收集为测试
test_file_discovery.__test__ = False
test_data_loading.__test__ = False
test_preprocessing.__test__ = False
test_label_generation.__test__ = False
test_normalization.__test__ = False
test_full_pipeline.__test__ = False
test_dataloader.__test__ = False


@pytest.mark.integration
def test_lobster_pipeline():
    """使用完整流水线跑一遍 LOBSTER 处理，用于 pytest 集成测试。"""
    # 测试1: 文件发现
    success, file_pairs = test_file_discovery()
    if not success:
        pytest.skip("LOBSTER 文件发现失败，可能是 ./data 下没有可用文件对")

    # 测试2: 数据加载
    success, message_df, orderbook_df = test_data_loading(file_pairs)
    assert success, "数据加载失败"

    # 测试3: 预处理
    success, msg_proc, ob_proc = test_preprocessing(message_df, orderbook_df)
    assert success, "数据预处理失败"

    # 测试4: 标签生成
    success, labels_long, labels_short = test_label_generation(msg_proc, ob_proc)
    assert success, "标签生成失败"
    assert len(labels_long) == len(labels_short) > 0

    # 测试5: 归一化
    success = test_normalization(file_pairs)
    assert success, "归一化流程失败"

    # 测试6: 完整数据构建流程
    success = test_full_pipeline()
    assert success, "完整数据构建流程失败"

    # 测试7: DataLoader
    success = test_dataloader()
    assert success, "DataLoader 构建或迭代失败"


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("  LOBSTER 数据处理流程综合测试")
    print("="*60)
    
    results = {}
    
    # 测试1: 文件发现
    success, file_pairs = test_file_discovery()
    results['文件发现'] = success
    
    if not success:
        print("\n❌ 文件发现失败，终止测试")
        return
    
    # 测试2: 数据加载
    success, message_df, orderbook_df = test_data_loading(file_pairs)
    results['数据加载'] = success
    
    # 测试3: 预处理
    if success:
        success, msg_proc, ob_proc = test_preprocessing(message_df, orderbook_df)
        results['数据预处理'] = success
        
        # 测试4: 标签生成
        if success:
            success, labels_long, labels_short = test_label_generation(msg_proc, ob_proc)
            results['标签生成'] = success
    
    # 测试5: 归一化
    success = test_normalization(file_pairs)
    results['归一化'] = success
    
    # 测试6: 完整流程
    success = test_full_pipeline()
    results['完整流程'] = success
    
    # 测试7: DataLoader
    if success:
        success = test_dataloader()
        results['DataLoader'] = success
    
    # 总结
    print_section("测试总结")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ 通过" if passed_flag else "✗ 失败"
        print(f"  {test_name:15s} {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠ {total - passed} 个测试失败")


if __name__ == "__main__":
    main()
