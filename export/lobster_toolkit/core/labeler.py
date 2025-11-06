"""标签生成策略模块"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class LabelStrategy(ABC):
    """标签生成策略基类"""
    
    @abstractmethod
    def generate(self, message_df: pd.DataFrame, 
                orderbook_df: pd.DataFrame) -> np.ndarray:
        """生成标签
        
        Args:
            message_df: message 数据
            orderbook_df: orderbook 数据
            
        Returns:
            标签数组
        """
        pass


class StopLossLabelStrategy(LabelStrategy):
    """基于止损/止盈的二分类标签策略
    
    标签定义：
    - 0: 止损先触发
    - 1: 止盈先触发
    """
    
    def __init__(self,
                 stop_loss_ticks: int = 6,
                 take_profit_ratio: float = 1.5,
                 tick_size: float = 0.1,
                 max_steps: Optional[int] = 64,
                 is_long: bool = True):
        """初始化标签策略
        
        Args:
            stop_loss_ticks: 止损阈值（tick数）
            take_profit_ratio: 止盈距离倍数（相对于止损）
            tick_size: 最小价格变动单位
            max_steps: 前向观察窗口大小（None表示观察到末尾）
            is_long: 是否为多头策略
                - True: 价格上涨盈利，下跌止损
                - False: 价格下跌盈利，上涨止损
        """
        self.stop_loss_ticks = stop_loss_ticks
        self.take_profit_ratio = take_profit_ratio
        self.tick_size = tick_size
        self.max_steps = max_steps
        self.is_long = is_long
        
        # 计算距离
        self.stop_loss_distance = tick_size * stop_loss_ticks
        self.take_profit_distance = take_profit_ratio * self.stop_loss_distance
    
    def generate(self, message_df: pd.DataFrame, 
                orderbook_df: pd.DataFrame) -> np.ndarray:
        """生成二分类标签
        
        Args:
            message_df: message 数据（未直接使用）
            orderbook_df: orderbook 数据
            
        Returns:
            标签数组 (n_samples,)
            
        Raises:
            ValueError: 当数据长度不一致时
        """
        if len(message_df) != len(orderbook_df):
            raise ValueError("message 与 orderbook 长度不一致")
        
        # 提取最优买卖价
        best_ask = orderbook_df['ask_price_1'].to_numpy(dtype=np.float32)
        best_bid = orderbook_df['bid_price_1'].to_numpy(dtype=np.float32)
        
        n_samples = len(best_bid)
        labels = np.zeros(n_samples, dtype=np.int8)
        
        # 为每个时间点生成标签
        for i in range(n_samples):
            end = None if self.max_steps is None else min(n_samples, i + 1 + self.max_steps)
            future_bids = best_bid[i + 1:end]
            future_asks = best_ask[i + 1:end]
            
            if future_bids.size == 0:
                continue
            
            if self.is_long:
                # 多头：价格上涨盈利，下跌止损
                stop_loss = best_bid[i] - self.stop_loss_distance
                take_profit = best_ask[i] + self.take_profit_distance
                
                t_sl = self._first_touch_idx(future_bids <= stop_loss)
                t_tp = self._first_touch_idx(future_asks >= take_profit)
            else:
                # 空头：价格下跌盈利，上涨止损
                stop_loss = best_ask[i] + self.stop_loss_distance
                take_profit = best_bid[i] - self.take_profit_distance
                
                t_sl = self._first_touch_idx(future_asks >= stop_loss)
                t_tp = self._first_touch_idx(future_bids <= take_profit)
            
            # 止盈先触发则标记为1
            if np.isfinite(t_tp) and (not np.isfinite(t_sl) or t_tp < t_sl):
                labels[i] = 1
        
        return labels
    
    @staticmethod
    def _first_touch_idx(condition: np.ndarray) -> float:
        """返回首次满足条件的索引，若不存在则返回 inf
        
        Args:
            condition: 布尔数组
            
        Returns:
            首次为True的索引，或 np.inf
        """
        idx = np.flatnonzero(condition)
        return float(idx[0]) if idx.size > 0 else np.inf


def generate_dual_labels(message_df: pd.DataFrame,
                        orderbook_df: pd.DataFrame,
                        stop_loss_ticks: int = 6,
                        take_profit_ratio: float = 1.5,
                        tick_size: float = 0.1,
                        max_steps: Optional[int] = 64) -> Tuple[np.ndarray, np.ndarray]:
    """生成多头和空头的二分类标签
    
    这是一个便捷函数，同时生成 long 和 short 标签。
    
    Args:
        message_df: message 数据
        orderbook_df: orderbook 数据
        stop_loss_ticks: 止损阈值
        take_profit_ratio: 止盈倍数
        tick_size: tick 大小
        max_steps: 前向观察窗口
        
    Returns:
        (labels_long, labels_short)
    """
    long_strategy = StopLossLabelStrategy(
        stop_loss_ticks=stop_loss_ticks,
        take_profit_ratio=take_profit_ratio,
        tick_size=tick_size,
        max_steps=max_steps,
        is_long=True
    )
    
    short_strategy = StopLossLabelStrategy(
        stop_loss_ticks=stop_loss_ticks,
        take_profit_ratio=take_profit_ratio,
        tick_size=tick_size,
        max_steps=max_steps,
        is_long=False
    )
    
    labels_long = long_strategy.generate(message_df, orderbook_df)
    labels_short = short_strategy.generate(message_df, orderbook_df)
    
    return labels_long, labels_short
