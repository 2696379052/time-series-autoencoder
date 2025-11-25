"""LOBSTER 数据构建器核心模块"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Any

from .preprocessor import MessagePreprocessor, OrderbookPreprocessor, preprocess_pair
from .labeler import generate_dual_labels
from .normalizer import ZScoreNormalizer


class LOBSTERDataBuilder:
    """LOBSTER 数据集构建器
    
    负责从原始 CSV 文件构建完整的训练数据集，包括：
    - 数据加载与配对
    - 预处理（价格归一化、时间差分）
    - 标签生成
    - 数据归一化
    - 数据集划分
    """
    
    def __init__(self,
                 split_rates: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 stop_loss_ticks: int = 6,
                 take_profit_ratio: float = 1.5,
                 tick_size: float = 0.1,
                 max_steps: int = 64,
                 data_dir_name: str = "GC"):
        """初始化数据构建器
        
        Args:
            split_rates: 数据集划分比例 (train, val, test)
            stop_loss_ticks: 止损阈值（tick数）
            take_profit_ratio: 止盈距离倍数
            tick_size: 最小价格变动单位
            max_steps: 标签生成的前向观察窗口
            data_dir_name: 数据子目录名称（默认 "GC"）
        """
        self.split_rates = split_rates
        self.stop_loss_ticks = stop_loss_ticks
        self.take_profit_ratio = take_profit_ratio
        self.tick_size = tick_size
        self.max_steps = max_steps
        self.data_dir_name = data_dir_name
        self.num_trading_days = 0
    
    def build_dataset(self, path: str) -> Dict[str, List[Dict[str, Any]]]:
        """构建完整数据集（主入口）
        
        Args:
            path: 包含 "GC" 子目录的根数据路径
            
        Returns:
            包含 train/val/test 的数据字典
        """
        # 1. 加载并预处理数据
        file_pairs = self._discover_files(path)
        self.num_trading_days = len(file_pairs)
        
        # 2. 划分数据集
        train_pairs, val_pairs, test_pairs = self._split_files(file_pairs)
        
        print(f"总交易天数: {self.num_trading_days}")
        print(f"训练集: {len(train_pairs)} 天, "
              f"验证集: {len(val_pairs)} 天, "
              f"测试集: {len(test_pairs)} 天")
        
        # 3. 处理每个数据集
        datasets = {
            'train': self._process_files(train_pairs),
            'val': self._process_files(val_pairs),
            'test': self._process_files(test_pairs)
        }
        
        # 4. 归一化
        normalizer = ZScoreNormalizer()
        normalizer.fit(datasets['train'])
        
        for split_name in ['train', 'val', 'test']:
            datasets[split_name] = normalizer.transform(datasets[split_name])
        
        # 5. 保存
        normalizer.save_params(os.path.join(path, "norm_params.csv"))
        self._save_datasets(datasets, path)
        
        return datasets
    
    def _discover_files(self, path: str) -> List[Dict[str, str]]:
        """发现并配对 message 和 orderbook 文件
        
        Args:
            path: 数据根目录
            
        Returns:
            文件对列表，每项包含 'message' 和 'orderbook' 路径
        """
        gc_path = os.path.join(path, self.data_dir_name)
        
        if not os.path.exists(gc_path):
            raise FileNotFoundError(f"路径不存在: {gc_path}")
        
        all_files = set(os.listdir(gc_path))
        message_files = sorted([f for f in all_files 
                               if '_message_' in f and f.endswith('.csv')])
        
        if not message_files:
            raise FileNotFoundError(f"未找到 message CSV 文件: {gc_path}")
        
        # 配对文件
        file_pairs = []
        for msg_file in message_files:
            ob_file = msg_file.replace('_message_', '_orderbook_')
            if ob_file in all_files:
                file_pairs.append({
                    'message': os.path.join(gc_path, msg_file),
                    'orderbook': os.path.join(gc_path, ob_file),
                    'date': self._extract_date(msg_file)
                })
            else:
                print(f"警告: 未找到匹配的 orderbook 文件: {msg_file}")
        
        if not file_pairs:
            raise ValueError(
                f"未找到任何有效的文件对。\n"
                f"检查路径: {gc_path}\n"
                f"期望文件格式: *_message_*.csv 和 *_orderbook_*.csv"
            )
        
        return file_pairs
    
    def _split_files(self, 
                    file_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """按比例划分文件
        
        Args:
            file_pairs: 文件对列表
            
        Returns:
            (train_pairs, val_pairs, test_pairs)
        """
        n_total = len(file_pairs)
        n_train = int(n_total * self.split_rates[0])
        n_val = int(n_total * self.split_rates[1])
        
        train_pairs = file_pairs[:n_train]
        val_pairs = file_pairs[n_train:n_train + n_val]
        test_pairs = file_pairs[n_train + n_val:]
        
        return train_pairs, val_pairs, test_pairs
    
    def _process_files(self, file_pairs: List[Dict]) -> List[Dict[str, Any]]:
        """处理文件对列表
        
        Args:
            file_pairs: 文件对列表
            
        Returns:
            处理后的数据项列表
        """
        data_items = []
        
        for pair in file_pairs:
            # 加载
            message_df = MessagePreprocessor.load_csv(pair['message'])
            orderbook_df = OrderbookPreprocessor.load_csv(pair['orderbook'])
            
            # 预处理
            message_df, orderbook_df = preprocess_pair(message_df, orderbook_df)
            
            # 生成标签
            labels_long, labels_short = generate_dual_labels(
                message_df, orderbook_df,
                stop_loss_ticks=self.stop_loss_ticks,
                take_profit_ratio=self.take_profit_ratio,
                tick_size=self.tick_size,
                max_steps=self.max_steps
            )
            
            data_items.append({
                'date': pair['date'],
                'message': message_df,
                'orderbook': orderbook_df,
                'labels_long': labels_long,
                'labels_short': labels_short
            })
        
        return data_items
    
    def _save_datasets(self, datasets: Dict[str, List], path: str) -> None:
        """保存数据集为 pkl 文件
        
        Args:
            datasets: 数据集字典
            path: 保存路径
        """
        for name, data_list in datasets.items():
            save_path = os.path.join(path, f"{name}.pkl")
            pd.to_pickle(data_list, save_path)
            print(f"已保存 {name} 数据至 {save_path}")
    
    @staticmethod
    def _extract_date(filename: str) -> str:
        """从文件名提取日期标识
        
        Args:
            filename: 文件名
            
        Returns:
            日期字符串
        """
        stem = os.path.splitext(os.path.basename(filename))[0]
        return stem.replace('_message', '')
    
    @staticmethod
    def load_processed_data(path: str) -> Dict[str, List[Dict[str, Any]]]:
        """加载已处理的 pkl 数据
        
        Args:
            path: 数据目录
            
        Returns:
            包含 train/val/test 的数据字典
        """
        datasets = {}
        for name in ['train', 'val', 'test']:
            file_path = os.path.join(path, f"{name}.pkl")
            if os.path.exists(file_path):
                datasets[name] = pd.read_pickle(file_path)
                print(f"已从 {file_path} 加载 {name} 数据")
        return datasets
