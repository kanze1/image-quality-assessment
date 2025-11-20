"""
数据预处理
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(score_file, all_dir):
    """
    加载数据 - 同时加载质量和一致性两个维度的分数
    
    数据结构说明：
    - RAW中有200张原始图像 (1.jpg - 200.jpg)
    - All中有1600张生成图像，每张原始图像对应8张生成图像
      例如: RAW/1.jpg → All/1.jpg, All/2.jpg, ..., All/8.jpg
           RAW/2.jpg → All/9.jpg, All/10.jpg, ..., All/16.jpg
    
    Args:
        score_file: 分数文件路径
        all_dir: 生成图像目录 (All/)
    
    Returns:
        image_paths: 图像路径列表
        quality_scores: 质量维度分数列表
        identity_scores: 人脸一致性维度分数列表
        raw_ids: 对应的原始图像ID列表
    """
    # 读取分数文件
    df = pd.read_excel(score_file)
    
    # 构建图像路径和分数列表
    image_paths = []
    quality_scores = []
    identity_scores = []
    raw_ids = []  # 记录对应的原始图像ID
    
    for idx, row in df.iterrows():
        img_id = int(row['图像ID'])  # 转换为整数
        quality_score = row['质量维度分数']
        identity_score = row['人脸一致性维度分数']
        
        # 计算对应的原始图像ID
        # All中的图像ID从1开始，每8张对应一张原始图像
        # img_id=1-8 → raw_id=1, img_id=9-16 → raw_id=2
        raw_id = (img_id - 1) // 8 + 1
        
        # 构建生成图像路径
        img_path = os.path.join(all_dir, f"{img_id}.jpg")
        
        # 检查文件是否存在
        if os.path.exists(img_path):
            image_paths.append(img_path)
            quality_scores.append(quality_score)
            identity_scores.append(identity_score)
            raw_ids.append(raw_id)
        else:
            # 尝试其他常见扩展名
            found = False
            for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                img_path_alt = os.path.join(all_dir, f"{img_id}{ext}")
                if os.path.exists(img_path_alt):
                    image_paths.append(img_path_alt)
                    quality_scores.append(quality_score)
                    identity_scores.append(identity_score)
                    raw_ids.append(raw_id)
                    found = True
                    break
            
            if not found:
                print(f"警告: 未找到图像 {img_id}")
    
    print(f"\n数据加载完成:")
    print(f"  加载了 {len(image_paths)} 张生成图像")
    print(f"  对应 {len(set(raw_ids))} 张原始图像")
    print(f"  质量分数范围: [{min(quality_scores):.3f}, {max(quality_scores):.3f}]")
    print(f"  质量分数均值: {np.mean(quality_scores):.3f}, 标准差: {np.std(quality_scores):.3f}")
    print(f"  一致性分数范围: [{min(identity_scores):.3f}, {max(identity_scores):.3f}]")
    print(f"  一致性分数均值: {np.mean(identity_scores):.3f}, 标准差: {np.std(identity_scores):.3f}")
    
    return image_paths, quality_scores, identity_scores, raw_ids


def split_data(image_paths, quality_scores, identity_scores, raw_ids, 
               train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    划分数据集 - 多任务版本
    
    重要: 按原始图像ID进行划分，确保同一张原始图像的8张生成图像在同一个集合中
    这样可以避免数据泄露，更准确地评估模型的泛化能力
    
    Args:
        image_paths: 图像路径列表
        quality_scores: 质量分数列表
        identity_scores: 一致性分数列表
        raw_ids: 原始图像ID列表
        train_ratio, val_ratio, test_ratio: 划分比例
        random_seed: 随机种子
    
    Returns:
        train_paths, train_quality, train_identity, train_raw_ids,
        val_paths, val_quality, val_identity, val_raw_ids,
        test_paths, test_quality, test_identity, test_raw_ids
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # 获取唯一的原始图像ID
    unique_raw_ids = sorted(set(raw_ids))
    np.random.seed(random_seed)
    np.random.shuffle(unique_raw_ids)
    
    # 按原始图像ID划分
    n_total = len(unique_raw_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_raw_ids_set = set(unique_raw_ids[:n_train])
    val_raw_ids_set = set(unique_raw_ids[n_train:n_train + n_val])
    test_raw_ids_set = set(unique_raw_ids[n_train + n_val:])
    
    # 根据原始图像ID分配生成图像
    train_paths, train_quality, train_identity, train_raw_ids = [], [], [], []
    val_paths, val_quality, val_identity, val_raw_ids = [], [], [], []
    test_paths, test_quality, test_identity, test_raw_ids = [], [], [], []
    
    for img_path, q_score, i_score, raw_id in zip(image_paths, quality_scores, identity_scores, raw_ids):
        if raw_id in train_raw_ids_set:
            train_paths.append(img_path)
            train_quality.append(q_score)
            train_identity.append(i_score)
            train_raw_ids.append(raw_id)
        elif raw_id in val_raw_ids_set:
            val_paths.append(img_path)
            val_quality.append(q_score)
            val_identity.append(i_score)
            val_raw_ids.append(raw_id)
        elif raw_id in test_raw_ids_set:
            test_paths.append(img_path)
            test_quality.append(q_score)
            test_identity.append(i_score)
            test_raw_ids.append(raw_id)
    
    print(f"\n数据划分 (按原始图像ID):")
    print(f"  训练集: {len(train_raw_ids_set)} 张原始图像 → {len(train_paths)} 张生成图像")
    print(f"  验证集: {len(val_raw_ids_set)} 张原始图像 → {len(val_paths)} 张生成图像")
    print(f"  测试集: {len(test_raw_ids_set)} 张原始图像 → {len(test_paths)} 张生成图像")
    
    return (train_paths, train_quality, train_identity, train_raw_ids,
            val_paths, val_quality, val_identity, val_raw_ids,
            test_paths, test_quality, test_identity, test_raw_ids)
