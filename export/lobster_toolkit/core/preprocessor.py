"""数据预处理模块"""

import pandas as pd
import numpy as np
from typing import Tuple


class MessagePreprocessor:
    """Message 数据预处理器"""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """加载 message CSV 文件
        
        Args:
            filepath: CSV 文件路径
            
        Returns:
            包含标准列名的 DataFrame
        """
        columns = ['time', 'type', 'order_id', 'size', 'price', 'direction']
        return pd.read_csv(filepath, header=None, names=columns)
    
    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """预处理 message 数据
        
        处理步骤：
        1. 计算时间差分（相对时间）
        2. 价格缩放（除以10000）
        3. 方向标记反转
        4. 移除不需要的列
        
        Args:
            df: 原始 message DataFrame
            
        Returns:
            预处理后的 DataFrame
        """
        df = df.copy()
        
        # 类型转换
        df['time'] = df['time'].astype(np.float32)
        df['size'] = df['size'].astype(np.float32)
        df['price'] = df['price'].astype(np.float32) / 10000
        df['direction'] = df['direction'].astype(np.float32) * -1
        
        # 时间差分（第一行为0）
        df['time'] = df['time'].diff().fillna(0).astype(np.float32)
        
        # 移除不需要的列
        df = df.drop(columns=['type', 'order_id'])
        
        return df
    
    @staticmethod
    def normalize_price_relative(df: pd.DataFrame) -> pd.DataFrame:
        """相对价格归一化（减去起始价格）
        
        Args:
            df: message DataFrame
            
        Returns:
            价格归一化后的 DataFrame
        """
        df = df.copy()
        if len(df) > 0:
            start_price = df['price'].iloc[0]
            df['price'] -= start_price
        return df


class OrderbookPreprocessor:
    """Orderbook 数据预处理器"""
    
    @staticmethod
    def load_csv(filepath: str, levels: int = 10) -> pd.DataFrame:
        """加载 orderbook CSV 文件
        
        Args:
            filepath: CSV 文件路径
            levels: 盘口档位数量（默认10档）
            
        Returns:
            包含标准列名的 DataFrame
        """
        columns = []
        for i in range(1, levels + 1):
            columns.extend([
                f'ask_price_{i}', f'ask_size_{i}',
                f'bid_price_{i}', f'bid_size_{i}'
            ])
        return pd.read_csv(filepath, header=None, names=columns)
    
    @staticmethod
    def preprocess(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
        """预处理 orderbook 数据
        
        处理步骤：
        1. 价格缩放（除以10000）
        2. 类型转换为 float32
        
        Args:
            df: 原始 orderbook DataFrame
            levels: 盘口档位数量
            
        Returns:
            预处理后的 DataFrame
        """
        df = df.copy()
        
        for i in range(1, levels + 1):
            df[f'ask_price_{i}'] = df[f'ask_price_{i}'].astype(np.float32) / 10000
            df[f'ask_size_{i}'] = df[f'ask_size_{i}'].astype(np.float32)
            df[f'bid_price_{i}'] = df[f'bid_price_{i}'].astype(np.float32) / 10000
            df[f'bid_size_{i}'] = df[f'bid_size_{i}'].astype(np.float32)
        
        return df
    
    @staticmethod
    def normalize_price_relative(df: pd.DataFrame, reference_price: float, 
                                levels: int = 10) -> pd.DataFrame:
        """相对价格归一化
        
        Args:
            df: orderbook DataFrame
            reference_price: 参考价格（通常是起始价格）
            levels: 盘口档位数量
            
        Returns:
            价格归一化后的 DataFrame
        """
        df = df.copy()
        for i in range(1, levels + 1):
            df[f'ask_price_{i}'] -= reference_price
            df[f'bid_price_{i}'] -= reference_price
        return df


def preprocess_pair(message_df: pd.DataFrame, 
                   orderbook_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """预处理 message 和 orderbook 数据对
    
    这是一个便捷函数，执行完整的预处理流程：
    1. 预处理 message 和 orderbook
    2. 应用相对价格归一化
    
    Args:
        message_df: 原始 message DataFrame
        orderbook_df: 原始 orderbook DataFrame
        
    Returns:
        (预处理后的 message, 预处理后的 orderbook)
    """
    # 预处理
    msg_proc = MessagePreprocessor.preprocess(message_df)
    ob_proc = OrderbookPreprocessor.preprocess(orderbook_df)
    
    # 相对价格归一化
    if len(msg_proc) > 0:
        start_price = msg_proc['price'].iloc[0]
        msg_proc = MessagePreprocessor.normalize_price_relative(msg_proc)
        ob_proc = OrderbookPreprocessor.normalize_price_relative(ob_proc, start_price)
    
    return msg_proc, ob_proc
