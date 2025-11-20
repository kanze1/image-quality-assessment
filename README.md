# 人脸图像质量评估 - 双分支架构

基于Vision Transformer的人脸图像质量评估系统，通过引入原始图像先验提升评估性能。

## 核心创新

### 双分支架构 + 原始图像先验

1. **生成图像分支**: 提取生成图像特征
2. **原始图像分支**: 提取原始参考图像特征  
3. **跨注意力机制**: 让生成图像特征关注原始图像特征
4. **多任务学习**: 同时预测质量维度和人脸一致性维度

### 损失函数设计

```
L_total = L_mse_quality + L_mse_identity 
          + λ_rank * (L_rank_quality + L_rank_identity)
          + λ_contrast * L_contrastive
```

- **MSE Loss**: 基础回归损失
- **Ranking Loss**: 保持分数相对顺序
- **Contrastive Loss**: 对比学习，增强特征表达

## 项目结构

```
.
├── config.py              # 配置文件
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── data/
│   ├── dataset.py         # 数据集定义
│   └── preprocess.py      # 数据预处理
├── models/
│   ├── vit_regressor.py   # 模型定义
│   └── losses.py          # 损失函数
└── utils/
    ├── metrics.py         # 评估指标
    └── visualization.py   # 可视化工具
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision timm pandas openpyxl scikit-learn scipy matplotlib tqdm
```

### 2. 数据准备

确保数据按以下结构组织：

```
.
├── RAW/                   # 200张原始图像
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── All/                   # 1600张生成图像
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── BT-Scores.xlsx         # 标注文件
```

### 3. 训练模型

```bash
python train.py
```

训练过程会自动：
- 按原始图像ID划分数据集（避免数据泄露）
- 保存最佳模型到 `checkpoints/best_model.pth`
- 生成训练曲线到 `results/training_curves.png`

### 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

评估结果包括：
- PLCC, SRCC, KRCC, RMSE 四项指标
- 预测值vs真实值散点图
- 误差分布图
- 结果文本文件

## 配置说明

主要配置项（`config.py`）：

```python
# 模型配置
MODEL_NAME = "vit_base_patch16_224"
EMBEDDING_DIM = 256

# 训练配置
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# 损失函数权重
LAMBDA_RANK = 0.5        # Ranking loss权重
LAMBDA_CONTRAST = 0.15   # Contrastive loss权重

# 人脸预训练权重（可选）
FACE_PRETRAINED_PATH = None  # 设置路径以使用预训练权重
```

## 评估指标

- **PLCC** (Pearson Linear Correlation Coefficient): 线性相关性，越接近1越好
- **SRCC** (Spearman Rank Correlation Coefficient): 秩相关性，越接近1越好
- **KRCC** (Kendall Rank Correlation Coefficient): 秩相关性，越接近1越好
- **RMSE** (Root Mean Square Error): 均方根误差，越接近0越好

## 数据划分策略

**重要**: 按原始图像ID划分，而非随机划分生成图像

- 训练集: 140张原始图像 → 1120张生成图像 (70%)
- 验证集: 30张原始图像 → 240张生成图像 (15%)
- 测试集: 30张原始图像 → 240张生成图像 (15%)

这样可以避免同一原始图像的不同生成版本出现在训练集和测试集中，更准确评估泛化能力。

## 使用人脸预训练权重（推荐）

使用在人脸数据集上预训练的权重可以显著提升性能：

1. 下载预训练权重（如VGGFace2, ArcFace等）
2. 在 `config.py` 中设置路径：
   ```python
   FACE_PRETRAINED_PATH = "path/to/pretrained_weights.pth"
   ```
3. 重新训练模型

## 结果输出

训练和评估会生成以下文件：

```
checkpoints/
└── best_model.pth         # 最佳模型权重

results/
├── training_curves.png    # 训练曲线
├── quality_scatter.png    # 质量评估散点图
├── quality_error_dist.png # 质量评估误差分布
├── identity_scatter.png   # 一致性散点图
├── identity_error_dist.png # 一致性误差分布
└── test_results.txt       # 测试结果文本
```
