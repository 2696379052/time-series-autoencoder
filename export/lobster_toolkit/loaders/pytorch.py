"""PyTorch 数据加载器"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional

from ..core.builder import LOBSTERDataBuilder


class LOBSTERDataset(Dataset):
    """LOBSTER PyTorch 数据集
    
    支持固定窗口长度的序列提取，自动处理跨文件边界。
    """
    
    def __init__(self,
                 data_items: List[Dict[str, Any]],
                 seq_len: int = 128,
                 label_type: str = 'long'):
        """初始化数据集
        
        Args:
            data_items: 数据项列表
            seq_len: 序列窗口长度
            label_type: 标签类型 ('long' 或 'short')
        """
        if label_type not in ['long', 'short']:
            raise ValueError("label_type 必须是 'long' 或 'short'")
        
        self.data_items = data_items
        self.seq_len = seq_len
        self.label_type = label_type
        
        # 构建全局索引映射
        self.index_mapping = self._build_index_mapping()
        
        print(f"数据集: {len(data_items)} 个文件, "
              f"{len(self)} 个序列, 标签类型: {label_type}")
    
    def _build_index_mapping(self) -> List[Tuple[int, int]]:
        """构建全局索引到 (文件索引, 序列索引) 的映射
        
        Returns:
            索引映射列表
        """
        mapping = []
        
        for file_idx, item in enumerate(self.data_items):
            # 获取最小长度
            min_len = min(
                len(item['message']),
                len(item['orderbook']),
                len(item['labels_long'])
            )
            
            # 计算可生成的序列数
            num_sequences = max(0, min_len - self.seq_len + 1)
            
            for seq_idx in range(num_sequences):
                mapping.append((file_idx, seq_idx))
        
        return mapping
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本
        
        Args:
            index: 全局索引
            
        Returns:
            (features, label)
            - features: (seq_len, n_features) 拼接的特征
            - label: 标量标签
        """
        file_idx, seq_idx = self.index_mapping[index]
        item = self.data_items[file_idx]
        
        # 提取序列
        start_idx = seq_idx
        end_idx = start_idx + self.seq_len
        
        message_seq = item['message'].iloc[start_idx:end_idx].values
        orderbook_seq = item['orderbook'].iloc[start_idx:end_idx].values
        
        # 拼接特征
        features = np.concatenate([message_seq, orderbook_seq], axis=1)
        
        # 获取标签（使用序列末尾的标签）
        labels = item['labels_long'] if self.label_type == 'long' else item['labels_short']
        label = labels[end_idx - 1]
        
        return torch.from_numpy(features).float(), torch.tensor(label, dtype=torch.long)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.index_mapping)


def get_dataloaders(data_path: str,
                   batch_size: int = 32,
                   seq_len: int = 128,
                   label_type: str = 'long',
                   num_workers: int = 0,
                   shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建 train/val/test DataLoader
    
    这是一个便捷函数，自动加载数据并创建 DataLoader。
    
    Args:
        data_path: 数据目录路径
        batch_size: 批次大小
        seq_len: 序列长度
        label_type: 标签类型 ('long' 或 'short')
        num_workers: 数据加载进程数
        shuffle_train: 是否打乱训练集
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 加载数据
    datasets = LOBSTERDataBuilder.load_processed_data(data_path)
    
    if not datasets:
        raise ValueError(f"未找到处理后的数据: {data_path}")
    
    # 创建 Dataset
    train_dataset = LOBSTERDataset(datasets['train'], seq_len, label_type)
    val_dataset = LOBSTERDataset(datasets.get('val', []), seq_len, label_type)
    test_dataset = LOBSTERDataset(datasets.get('test', []), seq_len, label_type)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
