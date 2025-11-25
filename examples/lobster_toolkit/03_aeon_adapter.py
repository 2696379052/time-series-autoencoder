"""示例 3: aeon 框架适配器"""

import sys
sys.path.insert(0, '..')

from lobster_toolkit.adapters.aeon import build_datasets

# 构建 aeon 格式数据
datasets = build_datasets(
    data_path="../../data",
    window_size=64,
    step=32,
    label_kind='long'
)

# 查看数据形状
for split_name, split_data in datasets.items():
    print(f"\n{split_name}:")
    print(f"  X: {split_data.X.shape}")  # (n_instances, n_channels, window_size)
    print(f"  y: {split_data.y.shape}")  # (n_instances,)
    print(f"  标签分布: {split_data.y.sum()} / {len(split_data.y)}")
    print(f"  元数据: {split_data.metadata.shape}")

# 查看元数据
print("\n前5个窗口的元数据:")
print(datasets['train'].metadata.head())

# 如果安装了 aeon，可以训练模型
try:
    from aeon.classification.convolution_based import MultiRocketClassifier
    
    print("\n训练 MultiRocket 分类器...")
    X_train = datasets['train'].X
    y_train = datasets['train'].y
    
    clf = MultiRocketClassifier(n_jobs=-1, random_state=42)
    clf.fit(X_train[:1000], y_train[:1000])  # 只用前1000个样本演示
    
    # 评估
    X_val = datasets['val'].X
    y_val = datasets['val'].y
    score = clf.score(X_val[:500], y_val[:500])
    print(f"验证集准确率: {score:.4f}")
    
except ImportError:
    print("\n提示: 安装 aeon 以使用时间序列分类模型")
    print("pip install aeon")
