"""LOBSTER数据训练脚本

支持多种训练模式：
- 重构任务：学习订单簿数据表示
- Long标签预测：预测做多信号
- Short标签预测：预测做空信号
"""
import os
import sys
import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tsa import AutoEncForecast, train, evaluate
from tsa.utils import load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def check_lobster_data(data_path: str):
    """检查LOBSTER数据是否已处理"""
    required_files = ['train.pkl', 'val.pkl', 'test.pkl', 'norm_params.csv']
    missing_files = []
    
    for fname in required_files:
        fpath = os.path.join(data_path, fname)
        if not os.path.exists(fpath):
            missing_files.append(fname)
    
    if missing_files:
        print(f"⚠️  缺少处理后的数据文件: {', '.join(missing_files)}")
        print(f"ℹ️  将自动运行数据预处理流程...")
        return False
    else:
        print(f"✓ 找到所有必需的数据文件")
        return True


def print_config_summary(cfg: DictConfig):
    """打印配置摘要"""
    print("\n" + "=" * 60)
    print("  LOBSTER 训练配置")
    print("=" * 60)
    
    print(f"\n【数据配置】")
    print(f"  任务类型: {cfg.data.task.value}")
    print(f"  数据路径: {cfg.data.data_path}")
    print(f"  批次大小: {cfg.data.batch_size}")
    print(f"  序列长度: {cfg.data.seq_length}")
    print(f"  目标来源: {cfg.data.target_source}")
    if cfg.data.target_source in ['labels', 'labels_long', 'labels_short']:
        print(f"  标签类型: {cfg.data.label_type}")
    
    print(f"\n【模型配置】")
    print(f"  编码器隐藏层: {cfg.training.hidden_size_encoder}")
    print(f"  解码器隐藏层: {cfg.training.hidden_size_decoder}")
    print(f"  输入注意力: {cfg.training.input_att}")
    print(f"  时间注意力: {cfg.training.temporal_att}")
    print(f"  去噪: {cfg.training.denoising}")
    
    print(f"\n【训练配置】")
    print(f"  学习率: {cfg.training.lr}")
    print(f"  训练轮数: {cfg.training.num_epochs}")
    print(f"  L1正则化: {cfg.training.reg1} (系数: {cfg.training.reg_factor1})")
    print(f"  L2正则化: {cfg.training.reg2} (系数: {cfg.training.reg_factor2})")
    
    print(f"\n【输出配置】")
    print(f"  输出目录: {cfg.general.output_dir}")
    print(f"  保存步数: {cfg.general.save_steps}")
    print(f"  日志步数: {cfg.general.logging_steps}")
    print(f"  训练时评估: {cfg.general.eval_during_training}")
    
    print("=" * 60 + "\n")


def print_dataset_info(train_iter, test_iter, nb_features, ts):
    """打印数据集信息"""
    print("\n" + "=" * 60)
    print("  数据集统计")
    print("=" * 60)
    
    print(f"\n训练集: {len(train_iter.dataset)} 个样本")
    print(f"测试集: {len(test_iter.dataset)} 个样本")
    print(f"特征维度: {nb_features}")
    
    if hasattr(ts, 'target_size'):
        print(f"目标维度: {ts.target_size}")
    
    # 获取一个batch查看形状
    for features, y_hist, labels in train_iter:
        print(f"\n样本形状:")
        print(f"  特征: {features.shape}")
        print(f"  历史: {y_hist.shape}")
        print(f"  标签: {labels.shape}")
        break
    
    print("=" * 60 + "\n")


@hydra.main(config_path="./", config_name="config_lobster", version_base=None)
def run(cfg: DictConfig):
    """主训练函数"""
    
    # 打印配置
    print_config_summary(cfg)
    
    # 检查数据
    data_path = cfg.data.data_path
    if data_path.startswith('..'):
        # 处理相对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.normpath(os.path.join(script_dir, data_path))
    
    print(f"检查数据目录: {data_path}")
    check_lobster_data(data_path)
    
    # 初始化数据集
    print("\n加载数据集...")
    try:
        ts = instantiate(cfg.data)
        train_iter, test_iter, nb_features = ts.get_loaders()
        print("✓ 数据加载成功")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        print(f"\n请确保:")
        print(f"  1. 数据路径正确: {data_path}")
        print(f"  2. 已运行 test_lobster_pipeline.py 进行数据预处理")
        print(f"  3. 或设置 build_if_missing: true 自动构建数据")
        raise
    
    # 打印数据集信息
    print_dataset_info(train_iter, test_iter, nb_features, ts)
    
    # 自动设置输出维度
    if hasattr(ts, 'target_size') and ts.target_size:
        cfg.training.output_size = ts.target_size
        print(f"自动设置输出维度为: {ts.target_size}\n")
    
    # 初始化模型
    print("初始化模型...")
    model = AutoEncForecast(cfg.training, input_size=nb_features).to(device)
    print(f"✓ 模型已创建并移至 {device}")
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}\n")
    
    # 设置损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    
    # 训练
    if cfg.general.do_train:
        print("=" * 60)
        print("  开始训练")
        print("=" * 60 + "\n")
        train(train_iter, test_iter, model, criterion, optimizer, cfg, ts)
        print("\n✓ 训练完成")
    
    # 评估
    if cfg.general.do_eval:
        if cfg.general.get("ckpt", False):
            print(f"\n加载检查点: {cfg.general.ckpt}")
            model, _, loss, epoch = load_checkpoint(
                cfg.general.ckpt, model, optimizer, device
            )
            print(f"✓ 已加载 epoch {epoch} 的模型")
        
        print("\n" + "=" * 60)
        print("  开始评估")
        print("=" * 60 + "\n")
        results = evaluate(test_iter, criterion, model, cfg, ts)
        
        print("\n评估结果:")
        for key, val in results.items():
            print(f"  {key}: {val:.6f}")
        print("\n✓ 评估完成")
    
    print("\n" + "=" * 60)
    print("  训练流程结束")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
