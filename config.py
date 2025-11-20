"""
配置文件
"""
import os

class Config:
    # 数据路径
    DATA_ROOT = "."
    RAW_DIR = os.path.join(DATA_ROOT, "RAW")
    ALL_DIR = os.path.join(DATA_ROOT, "All")
    SCORE_FILE = os.path.join(DATA_ROOT, "BT-Scores.xlsx")
    
    # 模型配置
    MODEL_NAME = "vit_small_patch16_224"  # 改用ViT Small（22M参数，更适合小数据集）
    PRETRAINED = True  # ✓ 使用ImageNet预训练权重（关键！）
    FACE_PRETRAINED_PATH = None  # 人脸预训练权重路径（可选）
    FREEZE_BACKBONE = True  # 冻结backbone，只训练任务头（小数据集关键策略）
    
    NUM_CLASSES = 1  # 回归任务
    IMG_SIZE = 224
    
    # 训练配置
    BATCH_SIZE = 16  # 减小batch size，增加更新频率
    NUM_EPOCHS = 100  # 增加epoch，让模型充分学习
    LEARNING_RATE = 5e-5  # 降低学习率，使用预训练权重时需要小心微调
    WEIGHT_DECAY = 1e-3  # 增加正则化，防止过拟合
    WARMUP_EPOCHS = 10  # 增加warmup
    
    # 数据划分
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # 数据增强（轻微，避免破坏质量感知）
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.3  # 降低翻转概率
    
    # 损失函数
    LOSS_TYPE = "multitask"  # 多任务学习
    # L_total = λ_mse * (L_mse_quality + L_mse_identity) 
    #          + λ_rank * (L_rank_quality + L_rank_identity)
    #          + λ_contrast * L_contrastive
    LAMBDA_MSE = 1.0  # MSE loss权重（提高，回归任务的基础）
    LAMBDA_RANK = 0.3  # Ranking loss权重（降低，小数据集上噪声大）
    LAMBDA_CONTRAST = 0.1  # Contrastive loss权重（大幅降低，小数据集效果差）
    
    # 对比学习类型
    CONTRASTIVE_TYPE = 'raw_id'  # 'raw_id': 基于RAW ID的InfoNCE (推荐)
                                  # 'hybrid': 混合RAW ID和分数相似度
    
    # 多任务学习
    USE_MULTITASK = True  # 是否使用多任务学习
    EMBEDDING_DIM = 256  # Embedding维度（用于对比学习）
    
    # 参考引导架构
    # 使用共享backbone + 差异建模，高效且语义清晰
    
    # 优化器
    OPTIMIZER = "adamw"
    SCHEDULER = "cosine"  # 可选: "cosine", "step", "plateau"
    
    # 保存路径
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULT_DIR = "results"
    
    # 设备
    DEVICE = "cuda"  # 自动检测
    NUM_WORKERS = 4
