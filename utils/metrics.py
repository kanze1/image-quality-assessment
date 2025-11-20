"""
评估指标
"""
import numpy as np
from scipy import stats


def calculate_plcc(predictions, targets):
    """
    计算Pearson线性相关系数 (PLCC)
    
    Args:
        predictions: 预测值
        targets: 真实值
    
    Returns:
        plcc: PLCC值
    """
    plcc, _ = stats.pearsonr(predictions, targets)
    return plcc


def calculate_srcc(predictions, targets):
    """
    计算Spearman秩相关系数 (SRCC)
    
    Args:
        predictions: 预测值
        targets: 真实值
    
    Returns:
        srcc: SRCC值
    """
    srcc, _ = stats.spearmanr(predictions, targets)
    return srcc


def calculate_krcc(predictions, targets):
    """
    计算Kendall秩相关系数 (KRCC)
    
    Args:
        predictions: 预测值
        targets: 真实值
    
    Returns:
        krcc: KRCC值
    """
    krcc, _ = stats.kendalltau(predictions, targets)
    return krcc


def calculate_rmse(predictions, targets):
    """
    计算均方根误差 (RMSE)
    
    Args:
        predictions: 预测值
        targets: 真实值
    
    Returns:
        rmse: RMSE值
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def evaluate_all_metrics(predictions, targets):
    """
    计算所有评估指标
    
    Args:
        predictions: 预测值数组
        targets: 真实值数组
    
    Returns:
        metrics: 包含所有指标的字典
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = {
        'PLCC': calculate_plcc(predictions, targets),
        'SRCC': calculate_srcc(predictions, targets),
        'KRCC': calculate_krcc(predictions, targets),
        'RMSE': calculate_rmse(predictions, targets)
    }
    
    return metrics


def print_metrics(metrics, prefix=''):
    """打印评估指标"""
    print(f"\n{prefix}评估指标:")
    print(f"  PLCC: {metrics['PLCC']:.4f}")
    print(f"  SRCC: {metrics['SRCC']:.4f}")
    print(f"  KRCC: {metrics['KRCC']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
