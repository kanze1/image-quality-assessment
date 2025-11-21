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
    MODEL_NAME = "hf-hub:timm/vit_base_patch16_224.augreg2_in21k_ft_in1k"
    PRETRAINED = True  # ✓ 使用ImageNet预训练权重（关键！）
    FACE_PRETRAINED_PATH = None  # 人脸预训练权重路径（可选）
    FREEZE_BACKBONE = False  # 冻结backbone，只训练任务头（小数据集关键策略）
    
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
    
    # 损失函数配置
    # 回归权重
    W_Q = 1.0  # 质量回归权重
    W_ID = 1.0  # ID一致性回归权重
    # PLCC 权重（趋势相关性）
    LAMBDA_PLCC = 0.1
    # SupCon 权重（同 raw_id）
    LAMBDA_SUP = 0.05
    # ranking 权重（ID 排序）
    LAMBDA_RANK = 0.05
    # SupCon 温度
    SUPCON_TEMPERATURE = 0.1
    # ranking margin
    RANK_MARGIN = 0.1
    
    # Embedding维度（用于对比学习）
    EMBEDDING_DIM = 256
    
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
