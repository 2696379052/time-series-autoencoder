"""LOBSTER Toolkit - 高频交易数据处理工具包

一个用于处理 LOBSTER (Limit Order Book) 数据的 Python 工具包。

快速开始:
    >>> from lobster_toolkit import build_dataset
    >>> datasets = build_dataset("./data")
"""

__version__ = "0.1.0"

from .core.builder import LOBSTERDataBuilder

__all__ = ["LOBSTERDataBuilder", "build_dataset"]


def build_dataset(path: str, **kwargs):
    """快速构建数据集的便捷函数
    
    Args:
        path: 数据根目录
        **kwargs: 传递给 LOBSTERDataBuilder 的参数，例如：
            - split_rates: 数据集划分比例 (默认 (0.8, 0.1, 0.1))
            - stop_loss_ticks: 止损阈值 (默认 6)
            - take_profit_ratio: 止盈倍数 (默认 1.5)
            - tick_size: tick 大小 (默认 0.1)
            - max_steps: 前向观察窗口 (默认 64)
            - data_dir_name: 数据子目录名称 (默认 "GC")
        
    Returns:
        包含 train/val/test 的数据字典
        
    Examples:
        >>> # 基础使用
        >>> datasets = build_dataset("./data")
        
        >>> # 自定义参数
        >>> datasets = build_dataset(
        ...     "./data",
        ...     stop_loss_ticks=10,
        ...     data_dir_name="AAPL"
        ... )
    """
    builder = LOBSTERDataBuilder(**kwargs)
    return builder.build_dataset(path)
