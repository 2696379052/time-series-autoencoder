"""基础功能测试"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd

from lobster_toolkit.core import (
    MessagePreprocessor,
    OrderbookPreprocessor,
    StopLossLabelStrategy,
    ZScoreNormalizer
)


class TestPreprocessor:
    """测试预处理器"""
    
    def test_message_preprocessor(self):
        """测试 message 预处理"""
        # 创建测试数据
        df = pd.DataFrame({
            'time': [1000.0, 2000.0, 3000.0],
            'type': [1, 2, 1],
            'order_id': [100, 101, 102],
            'size': [10.0, 20.0, 15.0],
            'price': [100000.0, 101000.0, 100500.0],
            'direction': [1.0, -1.0, 1.0]
        })
        
        processed = MessagePreprocessor.preprocess(df)
        
        # 检查列
        assert 'time' in processed.columns
        assert 'size' in processed.columns
        assert 'price' in processed.columns
        assert 'direction' in processed.columns
        assert 'type' not in processed.columns
        assert 'order_id' not in processed.columns
        
        # 检查时间差分
        assert processed['time'].iloc[0] == 0.0
        assert processed['time'].iloc[1] == 1000.0
        
        # 检查价格缩放
        assert processed['price'].iloc[0] == 10.0
    
    def test_orderbook_preprocessor(self):
        """测试 orderbook 预处理"""
        # 创建测试数据
        data = {}
        for i in range(1, 11):
            data[f'ask_price_{i}'] = [100000.0 + i*100]
            data[f'ask_size_{i}'] = [10.0 + i]
            data[f'bid_price_{i}'] = [99900.0 - i*100]
            data[f'bid_size_{i}'] = [15.0 + i]
        
        df = pd.DataFrame(data)
        processed = OrderbookPreprocessor.preprocess(df)
        
        # 检查价格缩放
        assert processed['ask_price_1'].iloc[0] == 10.01
        assert processed['bid_price_1'].iloc[0] == 9.99


class TestLabeler:
    """测试标签生成器"""
    
    def test_stop_loss_strategy(self):
        """测试止损/止盈标签策略"""
        # 创建模拟数据
        message_df = pd.DataFrame({
            'time': [0.0, 1.0, 2.0, 3.0, 4.0],
            'size': [10.0] * 5,
            'price': [0.0] * 5,
            'direction': [1.0] * 5
        })
        
        orderbook_df = pd.DataFrame({
            'ask_price_1': [10.0, 10.1, 10.2, 11.0, 10.0],
            'ask_size_1': [100.0] * 5,
            'bid_price_1': [9.9, 9.8, 9.7, 9.0, 9.9],
            'bid_size_1': [100.0] * 5,
        })
        
        strategy = StopLossLabelStrategy(
            stop_loss_ticks=6,
            take_profit_ratio=1.5,
            tick_size=0.1,
            max_steps=4,
            is_long=True
        )
        
        labels = strategy.generate(message_df, orderbook_df)
        
        assert len(labels) == 5
        assert labels.dtype == np.int8
        assert all(label in [0, 1] for label in labels)


class TestNormalizer:
    """测试归一化器"""
    
    def test_zscore_normalizer(self):
        """测试 Z-score 归一化"""
        # 创建测试数据
        data_items = [
            {
                'message': pd.DataFrame({
                    'time': [1.0, 2.0, 3.0],
                    'size': [10.0, 20.0, 30.0]
                }),
                'orderbook': pd.DataFrame({
                    'ask_size_1': [100.0, 200.0, 150.0],
                    'bid_size_1': [90.0, 180.0, 135.0]
                })
            }
        ]
        
        normalizer = ZScoreNormalizer()
        normalizer.fit(data_items)
        
        # 检查参数
        assert 'message' in normalizer.stats
        assert 'time' in normalizer.stats['message']
        assert 'mean' in normalizer.stats['message']['time']
        assert 'std' in normalizer.stats['message']['time']
        
        # 测试转换
        normalized = normalizer.transform(data_items)
        assert len(normalized) == 1
        
        # 检查均值接近0（允许小误差）
        mean_time = normalized[0]['message']['time'].mean()
        assert abs(mean_time) < 1e-6
    
    def test_normalizer_empty_data(self):
        """测试空数据的边界情况"""
        normalizer = ZScoreNormalizer()
        with pytest.raises(ValueError, match="data_items 不能为空"):
            normalizer.fit([])
    
    def test_csv_format(self):
        """测试 CSV 格式是否正确"""
        import tempfile
        import os
        
        # 创建测试数据
        data_items = [
            {
                'message': pd.DataFrame({
                    'time': [1.0, 2.0],
                    'size': [10.0, 20.0]
                }),
                'orderbook': pd.DataFrame({
                    'ask_size_1': [100.0, 200.0]
                })
            }
        ]
        
        normalizer = ZScoreNormalizer()
        normalizer.fit(data_items)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name
        
        try:
            normalizer.save_params(temp_path)
            
            # 读取并验证格式
            df = pd.read_csv(temp_path)
            assert list(df.columns) == ['type', 'param', 'mean', 'std']
            assert 'message' in df['type'].values
            assert 'time' in df['param'].values
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
