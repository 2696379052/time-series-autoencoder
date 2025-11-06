"""aeon 时间序列框架适配器"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple

from ..core.builder import LOBSTERDataBuilder


LabelKind = Literal["long", "short"]


@dataclass
class WindowedSplit:
    """窗口化数据容器
    
    Attributes:
        X: 特征数组 (n_instances, n_channels, window_size)
        y: 标签数组 (n_instances,)
        metadata: 窗口元信息 DataFrame
    """
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


class AeonAdapter:
    """aeon 框架适配器
    
    将 LOBSTER 数据转换为 aeon 兼容的格式：
    - 滑动窗口提取
    - 价格列滑动归一化
    - channels-first 布局
    """
    
    def __init__(self, window_size: int = 64, step: int = 32):
        """初始化适配器
        
        Args:
            window_size: 滑动窗口大小
            step: 滑动窗口步长
        """
        if window_size <= 0:
            raise ValueError("window_size 必须为正数")
        if step <= 0:
            raise ValueError("step 必须为正数")
        
        self.window_size = window_size
        self.step = step
    
    def convert_split(self,
                     data_items: List[Dict],
                     label_kind: LabelKind = "long") -> WindowedSplit:
        """转换一个数据分片
        
        Args:
            data_items: 数据项列表
            label_kind: 标签类型
            
        Returns:
            WindowedSplit 容器
        """
        if label_kind not in ("long", "short"):
            raise ValueError("label_kind 必须是 'long' 或 'short'")
        
        all_X = []
        all_y = []
        all_meta = []
        
        for idx, item in enumerate(data_items):
            # 应用滑动窗口价格归一化
            message_norm = self._sliding_normalize_prices(item['message'])
            
            # 准备特征（仅使用 message 数据）
            features = message_norm.to_numpy(dtype=np.float32)
            column_names = list(message_norm.columns)
            
            # 选择标签
            labels = item[f'labels_{label_kind}']
            
            # 生成滑动窗口
            X_item, y_item, meta_item = self._create_windows(
                features=features,
                labels=labels,
                column_names=column_names,
                date_tag=item.get('date', f'item_{idx}')
            )
            
            if X_item.size > 0:
                all_X.append(X_item)
                all_y.append(y_item)
                all_meta.append(meta_item)
        
        # 合并所有数据
        if not all_X:
            n_channels = len(data_items[0]['message'].columns)
            empty_X = np.empty((0, n_channels, self.window_size), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int64)
            empty_meta = pd.DataFrame(columns=["date", "start", "end", "channels"])
            return WindowedSplit(X=empty_X, y=empty_y, metadata=empty_meta)
        
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        metadata = pd.concat(all_meta, ignore_index=True)
        
        return WindowedSplit(X=X, y=y, metadata=metadata)
    
    def _sliding_normalize_prices(self, message_df: pd.DataFrame) -> pd.DataFrame:
        """对 price 列进行滑动窗口 z-score 归一化
        
        Args:
            message_df: message DataFrame
            
        Returns:
            归一化后的 DataFrame
        """
        if message_df.empty or 'price' not in message_df.columns:
            return message_df.copy()
        
        df_norm = message_df.copy()
        
        # 使用 pandas rolling 计算滑动统计量
        price_series = df_norm['price']
        rolling_mean = price_series.rolling(
            window=self.window_size, 
            min_periods=1
        ).mean()
        rolling_std = price_series.rolling(
            window=self.window_size, 
            min_periods=1
        ).std(ddof=0)
        
        # 处理标准差为0或NaN
        rolling_std = rolling_std.fillna(1.0)
        rolling_std = np.where(rolling_std < 1e-8, 1.0, rolling_std)
        
        # 应用归一化
        df_norm['price'] = (price_series - rolling_mean) / rolling_std
        
        return df_norm
    
    def _create_windows(self,
                       features: np.ndarray,
                       labels: np.ndarray,
                       column_names: List[str],
                       date_tag: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """创建滑动窗口
        
        Args:
            features: 特征数组 (n_samples, n_features)
            labels: 标签数组 (n_samples,)
            column_names: 特征列名
            date_tag: 日期标识
            
        Returns:
            (X, y, metadata)
        """
        if features.shape[0] != labels.shape[0]:
            raise ValueError("特征和标签长度不一致")
        
        instances = []
        instance_labels = []
        metadata_rows = []
        
        for start in range(0, features.shape[0] - self.window_size + 1, self.step):
            end = start + self.window_size
            
            # 提取窗口并转置为 (n_features, window_size)
            window = features[start:end].T
            instances.append(window)
            
            # 使用窗口末尾的标签
            instance_labels.append(int(labels[end - 1]))
            
            metadata_rows.append({
                "date": date_tag,
                "start": start,
                "end": end
            })
        
        if not instances:
            n_channels = features.shape[1]
            empty_X = np.empty((0, n_channels, self.window_size), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int64)
            empty_meta = pd.DataFrame(columns=["date", "start", "end"])
            return empty_X, empty_y, empty_meta
        
        X = np.stack(instances).astype(np.float32)
        y = np.asarray(instance_labels, dtype=np.int64)
        metadata = pd.DataFrame(metadata_rows)
        metadata["channels"] = ",".join(column_names)
        
        return X, y, metadata


def build_datasets(data_path: str,
                  window_size: int = 64,
                  step: int = 32,
                  label_kind: LabelKind = "long") -> Dict[str, WindowedSplit]:
    """构建 aeon 格式数据集（便捷函数）
    
    Args:
        data_path: 数据目录路径
        window_size: 滑动窗口大小
        step: 滑动窗口步长
        label_kind: 标签类型
        
    Returns:
        包含 train/val/test 的 WindowedSplit 字典
    """
    # 加载数据
    datasets = LOBSTERDataBuilder.load_processed_data(data_path)
    
    if not datasets:
        raise ValueError(f"未找到处理后的数据: {data_path}")
    
    # 创建适配器
    adapter = AeonAdapter(window_size=window_size, step=step)
    
    # 转换每个分片
    windowed = {}
    for split_name, items in datasets.items():
        windowed[split_name] = adapter.convert_split(items, label_kind=label_kind)
        print(f"{split_name}: {windowed[split_name].X.shape[0]} 个窗口")
    
    return windowed
