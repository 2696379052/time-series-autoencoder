"""示例 2: PyTorch 数据加载器"""

import sys
sys.path.insert(0, '..')

from lobster_toolkit.loaders.pytorch import get_dataloaders

# 创建 DataLoader
train_loader, val_loader, test_loader = get_dataloaders(
    data_path="../../data",
    batch_size=32,
    seq_len=128,
    label_type='long',
    shuffle_train=True
)

print(f"训练集: {len(train_loader.dataset)} 个序列")
print(f"验证集: {len(val_loader.dataset)} 个序列")
print(f"测试集: {len(test_loader.dataset)} 个序列")

# 查看一个 batch
for features, labels in train_loader:
    print(f"\nBatch 形状:")
    print(f"Features: {features.shape}")  # (batch, seq_len, n_features)
    print(f"Labels: {labels.shape}")      # (batch,)
    print(f"标签分布: {labels.unique(return_counts=True)}")
    break

# 模拟训练循环
print("\n模拟训练...")
for epoch in range(2):
    for batch_idx, (features, labels) in enumerate(train_loader):
        if batch_idx >= 3:  # 只训练几个 batch
            break
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
              f"features {features.shape}, labels {labels.shape}")
