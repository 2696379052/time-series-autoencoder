"""数据归一化工具模块"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import os


class ZScoreNormalizer:
    """Z-Score 归一化器
    
    对指定列进行 z-score 归一化：(x - mean) / std
    """
    
    def __init__(self):
        """初始化归一化器"""
        self.stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.fitted = False
    
    def fit(self, 
            data_items: List[Dict],
            message_cols: Optional[List[str]] = None,
            orderbook_cols: Optional[List[str]] = None) -> 'ZScoreNormalizer':
        """从数据中计算归一化参数
        
        Args:
            data_items: 数据项列表，每项包含 'message' 和 'orderbook' DataFrame
            message_cols: 要归一化的 message 列（默认: ['time', 'size']）
            orderbook_cols: 要归一化的 orderbook 列（默认: 所有 *_size_* 列）
            
        Returns:
            self (支持链式调用)
            
        Raises:
            ValueError: 当 data_items 为空时
        """
        if not data_items:
            raise ValueError("data_items 不能为空")
        
        if message_cols is None:
            message_cols = ['time', 'size']
        
        # 收集所有 message 数据
        all_messages = pd.concat(
            [item['message'][message_cols] for item in data_items],
            ignore_index=True,
            copy=False
        )
        
        # 计算 message 统计量
        self.stats['message'] = {}
        for col in message_cols:
            self.stats['message'][col] = {
                'mean': float(all_messages[col].mean()),
                'std': float(all_messages[col].std(ddof=0))
            }
        
        # 收集所有 orderbook 数据
        if orderbook_cols is None:
            # 默认归一化所有 size 列
            sample_ob = data_items[0]['orderbook']
            orderbook_cols = [col for col in sample_ob.columns if '_size_' in col]
        
        all_orderbooks = pd.concat(
            [item['orderbook'][orderbook_cols] for item in data_items],
            ignore_index=True,
            copy=False
        )
        
        # 计算 orderbook 统计量
        self.stats['orderbook'] = {}
        for col in orderbook_cols:
            self.stats['orderbook'][col] = {
                'mean': float(all_orderbooks[col].mean()),
                'std': float(all_orderbooks[col].std(ddof=0))
            }
        
        self.fitted = True
        return self
    
    def transform(self, 
                 data_items: List[Dict],
                 eps: float = 1e-8) -> List[Dict]:
        """应用归一化到数据
        
        Args:
            data_items: 要归一化的数据项列表
            eps: 防止除零的小常数
            
        Returns:
            归一化后的数据项列表
        """
        if not self.fitted:
            raise RuntimeError("必须先调用 fit() 方法")
        
        normalized_items = []
        
        for item in data_items:
            normalized_item = item.copy()
            
            # 归一化 message
            msg_df = item['message'].copy()
            for col, params in self.stats['message'].items():
                if col in msg_df.columns:
                    mean = params['mean']
                    std = params['std'] if params['std'] > eps else 1.0
                    msg_df[col] = ((msg_df[col] - mean) / std).astype(np.float32)
            normalized_item['message'] = msg_df
            
            # 归一化 orderbook
            ob_df = item['orderbook'].copy()
            for col, params in self.stats['orderbook'].items():
                if col in ob_df.columns:
                    mean = params['mean']
                    std = params['std'] if params['std'] > eps else 1.0
                    ob_df[col] = ((ob_df[col] - mean) / std).astype(np.float32)
            normalized_item['orderbook'] = ob_df
            
            normalized_items.append(normalized_item)
        
        return normalized_items
    
    def fit_transform(self, 
                     data_items: List[Dict],
                     **kwargs) -> List[Dict]:
        """计算并应用归一化（便捷方法）
        
        Args:
            data_items: 数据项列表
            **kwargs: 传递给 fit() 的参数
            
        Returns:
            归一化后的数据项列表
        """
        self.fit(data_items, **kwargs)
        return self.transform(data_items)
    
    def save_params(self, filepath: str) -> None:
        """保存归一化参数到 CSV 文件
        
        Args:
            filepath: 保存路径
        """
        if not self.fitted:
            raise RuntimeError("没有可保存的参数，请先调用 fit()")
        
        rows = []
        for domain in ['message', 'orderbook']:
            for col, params in self.stats[domain].items():
                rows.append({
                    'type': domain,
                    'param': col,
                    'mean': params['mean'],
                    'std': params['std']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    @classmethod
    def load_params(cls, filepath: str) -> 'ZScoreNormalizer':
        """从 CSV 文件加载归一化参数
        
        Args:
            filepath: CSV 文件路径
            
        Returns:
            加载了参数的归一化器实例
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"参数文件不存在: {filepath}")
        
        df = pd.read_csv(filepath)
        normalizer = cls()
        normalizer.stats = {'message': {}, 'orderbook': {}}
        
        for _, row in df.iterrows():
            domain = row['type']
            col = row['param']
            normalizer.stats[domain][col] = {
                'mean': float(row['mean']),
                'std': float(row['std'])
            }
        
        normalizer.fitted = True
        return normalizer
    
    def get_params(self) -> Dict:
        """获取归一化参数
        
        Returns:
            参数字典
        """
        if not self.fitted:
            raise RuntimeError("没有可用的参数，请先调用 fit()")
        return self.stats.copy()
