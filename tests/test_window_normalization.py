"""测试 message.price 窗口归一化功能"""

import numpy as np
import sys
import os

# 添加 export 目录到路径
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'export'))

from lobster_toolkit.core.builder import LOBSTERDataBuilder


def test_window_normalization():
    """测试窗口归一化是否正确工作"""
    
    print("\n" + "=" * 60)
    print("  测试 Message.Price 窗口归一化功能")
    print("=" * 60 + "\n")
    
    # 1. 检查数据是否存在
    data_path = "./data"
    required_files = ['train.pkl', 'val.pkl', 'test.pkl', 'norm_params.csv']
    
    print("步骤 1: 检查数据文件")
    all_exist = True
    for fname in required_files:
        fpath = os.path.join(data_path, fname)
        if os.path.exists(fpath):
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} - 不存在")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  缺少必要文件，请先运行 test_lobster_pipeline.py")
        return False
    
    # 2. 加载归一化参数并检查 price 列
    print("\n步骤 2: 检查归一化参数")
    try:
        from lobster_toolkit.core.normalizer import ZScoreNormalizer
        normalizer = ZScoreNormalizer.load_params(os.path.join(data_path, "norm_params.csv"))
        params = normalizer.get_params()
        
        if 'message' in params and 'price' in params['message']:
            price_mean = params['message']['price']['mean']
            price_std = params['message']['price']['std']
            print(f"  ✓ Message.Price 全局统计量:")
            print(f"    均值: {price_mean:.6f}")
            print(f"    标准差: {price_std:.6f}")
        else:
            print(f"  ✗ 未找到 message.price 的全局统计量")
            return False
    except Exception as e:
        print(f"  ✗ 加载归一化参数失败: {e}")
        return False
    
    # 3. 测试 LOBSTERTimeSeriesDataset
    print("\n步骤 3: 测试 LOBSTERTimeSeriesDataset")
    try:
        from tsa.dataset import LOBSTERTimeSeriesDataset, Tasks
        
        dataset = LOBSTERTimeSeriesDataset(
            task=Tasks.prediction,
            data_path=data_path,
            seq_length=64,
            batch_size=16,
            prediction_window=1,
            target_source="labels",
            label_type="long",
            build_if_missing=False
        )
        
        print(f"  ✓ 数据集实例化成功")
        
        if dataset.price_global_std is not None:
            print(f"  ✓ 成功加载 price 全局标准差: {dataset.price_global_std:.6f}")
        else:
            print(f"  ⚠️  price_global_std 为 None")
        
        # 获取 DataLoader
        train_loader, test_loader, nb_features = dataset.get_loaders()
        print(f"  ✓ DataLoader 创建成功")
        print(f"    特征维度: {nb_features}")
        print(f"    训练样本数: {len(train_loader.dataset)}")
        print(f"    测试样本数: {len(test_loader.dataset)}")
        
        # 查看特征列名
        train_item = dataset.datasets['train'][0]
        message_cols = list(train_item['message'].columns)
        orderbook_cols = list(train_item['orderbook'].columns)
        all_cols = message_cols + orderbook_cols
        print(f"\n  特征列名:")
        print(f"    Message 列: {message_cols}")
        print(f"    Orderbook 列(前5): {orderbook_cols[:5]}")
        if 'price' in message_cols:
            price_idx = message_cols.index('price')
            print(f"    Price 在 message 中的索引: {price_idx}")
            print(f"    Price 在合并后的索引: {price_idx} (假设 message 在前)")
        
        # 检查一个 batch
        print("\n步骤 4: 检查窗口归一化效果")
        for features, y_hist, labels in train_loader:
            print(f"  Batch shape: {features.shape}")
            
            # 假设 price 是第一个特征列（需要根据实际情况调整）
            # 如果你知道 price 在哪一列，可以直接索引
            print(f"\n  特征统计 (所有列):")
            for i in range(min(5, features.shape[-1])):  # 只显示前5列
                col_data = features[:, :, i].flatten()
                print(f"    列 {i}: mean={col_data.mean():.6f}, std={col_data.std():.6f}, "
                      f"min={col_data.min():.6f}, max={col_data.max():.6f}")
            
            # 检查 price 列（在第 2 列）
            if nb_features > 2:
                price_col_idx = 2  # price 在 message 中的索引
                price_col = features[:, :, price_col_idx].flatten()
                print(f"\n  列 {price_col_idx} (price) 窗口归一化后:")
                print(f"    全局均值: {price_col.mean():.6f}  ← 应接近 0")
                print(f"    全局标准差: {price_col.std():.6f}")
                print(f"    范围: [{price_col.min():.6f}, {price_col.max():.6f}]")
                
                # 检查单个窗口的均值是否接近0
                window_means = features[:, :, price_col_idx].mean(dim=1)
                print(f"\n  各窗口的 price 均值分布:")
                print(f"    平均窗口均值: {window_means.mean():.6f}  ← 应接近 0")
                print(f"    窗口均值标准差: {window_means.std():.6f}")
                print(f"    检查: {'\u2713 通过' if abs(window_means.mean()) < 0.001 else '✗ 失败'} (应 < 0.001)")
            
            break  # 只检查第一个 batch
        
        print("\n" + "=" * 60)
        print("  ✓ 测试完成")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_window_normalization()
    sys.exit(0 if success else 1)
