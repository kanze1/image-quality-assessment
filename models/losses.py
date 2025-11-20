"""
多任务损失函数 - MSE + Ranking + Contrastive
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingLoss(nn.Module):
    """
    Ranking Loss - 保持分数的相对顺序
    对于分数对 (s1, s2)，如果 s1 > s2，则预测也应该 p1 > p2
    """
    def __init__(self, margin=0.5):
        super(RankingLoss, self).__init__()
        self.margin = margin
    
    def forward(self, predictions, targets):
        # 创建所有可能的配对
        batch_size = predictions.size(0)
        
        # 扩展维度用于配对比较
        pred_i = predictions.unsqueeze(1)  # (batch, 1)
        pred_j = predictions.unsqueeze(0)  # (1, batch)
        target_i = targets.unsqueeze(1)
        target_j = targets.unsqueeze(0)
        
        # 计算分数差异
        pred_diff = pred_i - pred_j  # (batch, batch)
        target_diff = target_i - target_j
        
        # 只考虑目标分数有明显差异的配对（避免噪声）
        valid_pairs = torch.abs(target_diff) > 0.5
        
        # Ranking loss: 如果 target_i > target_j，则希望 pred_i > pred_j
        # 使用 hinge loss
        loss = torch.clamp(self.margin - torch.sign(target_diff) * pred_diff, min=0)
        
        # 只计算有效配对的损失
        loss = loss * valid_pairs.float()
        
        # 平均损失
        num_valid = valid_pairs.sum() + 1e-8
        return loss.sum() / num_valid


class RawIDContrastiveLoss(nn.Module):
    """
    基于RAW ID的对比学习损失 - 标准InfoNCE loss
    
    核心思想：
    - 同一个原始图像的8张生成图像 = 正样本（应该相似）
    - 不同原始图像的生成图像 = 负样本（应该不同）
    """
    def __init__(self, temperature=0.07):
        super(RawIDContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, raw_ids):
        """
        Args:
            embeddings: (batch_size, embedding_dim) - L2归一化的embedding
            raw_ids: (batch_size,) - 对应的原始图像ID
        
        Returns:
            InfoNCE loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 计算embedding之间的相似度矩阵
        # sim_matrix[i,j] = embeddings[i] · embeddings[j] / temperature
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature  # (batch, batch)
        
        # 构建正样本mask：raw_ids相同的为正样本
        raw_ids = raw_ids.unsqueeze(1)  # (batch, 1)
        positive_mask = (raw_ids == raw_ids.t()).float()  # (batch, batch)
        
        # 排除自己（对角线）
        self_mask = torch.eye(batch_size, device=device).bool()
        positive_mask = positive_mask.masked_fill(self_mask, 0)
        
        # 负样本mask：raw_ids不同的为负样本
        negative_mask = (raw_ids != raw_ids.t()).float()
        
        # 计算InfoNCE loss
        # 对于每个anchor，计算与所有正样本的相似度 vs 所有负样本的相似度
        
        # 数值稳定性：减去最大值
        sim_matrix_exp = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach())
        
        # 分子：正样本的相似度之和
        pos_sim = (sim_matrix_exp * positive_mask).sum(dim=1)  # (batch,)
        
        # 分母：所有样本（正样本+负样本）的相似度之和
        all_sim = (sim_matrix_exp * (positive_mask + negative_mask)).sum(dim=1)  # (batch,)
        
        # InfoNCE loss: -log(正样本相似度 / 所有相似度)
        # 只计算有正样本的anchor
        has_positive = (positive_mask.sum(dim=1) > 0)
        
        if has_positive.sum() > 0:
            loss = -torch.log(pos_sim[has_positive] / (all_sim[has_positive] + 1e-8))
            return loss.mean()
        else:
            # 如果batch中没有正样本对，返回0
            return torch.tensor(0.0, device=device)


class HybridContrastiveLoss(nn.Module):
    """
    混合对比学习损失：结合RAW ID和分数相似度
    
    - RAW ID定义硬正负样本（同一原始图像 vs 不同原始图像）
    - 分数相似度定义软权重（分数接近的样本更重要）
    """
    def __init__(self, temperature=0.07, score_weight=0.3):
        super(HybridContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.score_weight = score_weight  # 分数相似度的权重
    
    def forward(self, embeddings, raw_ids, scores):
        """
        Args:
            embeddings: (batch_size, embedding_dim) - L2归一化的embedding
            raw_ids: (batch_size,) - 对应的原始图像ID
            scores: (batch_size,) - 质量或一致性分数
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # 相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # RAW ID正样本mask
        raw_ids = raw_ids.unsqueeze(1)
        positive_mask = (raw_ids == raw_ids.t()).float()
        self_mask = torch.eye(batch_size, device=device).bool()
        positive_mask = positive_mask.masked_fill(self_mask, 0)
        
        # 分数相似度权重
        score_i = scores.unsqueeze(1)
        score_j = scores.unsqueeze(0)
        score_diff = torch.abs(score_i - score_j)
        score_sim = torch.exp(-score_diff ** 2 / 2.0)  # 高斯核
        
        # 结合RAW ID和分数相似度
        # 正样本：必须是同一RAW ID，权重由分数相似度调整
        weighted_positive_mask = positive_mask * (1.0 + self.score_weight * score_sim)
        
        # 负样本：不同RAW ID
        negative_mask = (raw_ids != raw_ids.t()).float()
        
        # InfoNCE loss
        sim_matrix_exp = torch.exp(sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach())
        
        pos_sim = (sim_matrix_exp * weighted_positive_mask).sum(dim=1)
        all_sim = (sim_matrix_exp * (weighted_positive_mask + negative_mask)).sum(dim=1)
        
        has_positive = (positive_mask.sum(dim=1) > 0)
        
        if has_positive.sum() > 0:
            loss = -torch.log(pos_sim[has_positive] / (all_sim[has_positive] + 1e-8))
            return loss.mean()
        else:
            return torch.tensor(0.0, device=device)


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    L_total = λ_mse * (L_mse_quality + L_mse_identity) 
              + λ_rank * (L_rank_quality + L_rank_identity)
              + λ_contrast * L_contrastive
    """
    def __init__(self, lambda_mse=0.3, lambda_rank=1.0, lambda_contrast=0.5, 
                 contrastive_type='raw_id'):
        """
        Args:
            lambda_mse: MSE loss权重
            lambda_rank: Ranking loss权重
            lambda_contrast: Contrastive loss权重
            contrastive_type: 对比学习类型
                - 'raw_id': 基于RAW ID的标准InfoNCE (推荐)
                - 'hybrid': 混合RAW ID和分数相似度
        """
        super(MultiTaskLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = RankingLoss(margin=0.5)
        
        # 选择对比学习损失类型
        if contrastive_type == 'raw_id':
            self.contrastive_loss = RawIDContrastiveLoss(temperature=0.07)
        elif contrastive_type == 'hybrid':
            self.contrastive_loss = HybridContrastiveLoss(temperature=0.07, score_weight=0.3)
        else:
            raise ValueError(f"Unknown contrastive_type: {contrastive_type}")
        
        self.contrastive_type = contrastive_type
        self.lambda_mse = lambda_mse
        self.lambda_rank = lambda_rank
        self.lambda_contrast = lambda_contrast
    
    def forward(self, quality_pred, identity_pred, quality_target, identity_target, 
                embeddings, raw_ids):
        """
        Args:
            quality_pred: 质量预测 (batch_size,)
            identity_pred: 一致性预测 (batch_size,)
            quality_target: 质量真实值 (batch_size,)
            identity_target: 一致性真实值 (batch_size,)
            embeddings: embedding向量 (batch_size, embedding_dim)
            raw_ids: 原始图像ID (batch_size,)
        """
        # MSE损失
        loss_mse_quality = self.mse_loss(quality_pred, quality_target)
        loss_mse_identity = self.mse_loss(identity_pred, identity_target)
        
        # Ranking损失
        loss_rank_quality = self.ranking_loss(quality_pred, quality_target)
        loss_rank_identity = self.ranking_loss(identity_pred, identity_target)
        
        # 对比学习损失（基于RAW ID）
        if self.contrastive_type == 'raw_id':
            loss_contrast = self.contrastive_loss(embeddings, raw_ids)
        elif self.contrastive_type == 'hybrid':
            # 使用一致性分数作为软权重
            loss_contrast = self.contrastive_loss(embeddings, raw_ids, identity_target)
        
        # 总损失
        total_loss = (self.lambda_mse * (loss_mse_quality + loss_mse_identity) + 
                     self.lambda_rank * (loss_rank_quality + loss_rank_identity) +
                     self.lambda_contrast * loss_contrast)
        
        # 返回总损失和各项损失（用于监控）
        loss_dict = {
            'total': total_loss,
            'mse_quality': loss_mse_quality,
            'mse_identity': loss_mse_identity,
            'rank_quality': loss_rank_quality,
            'rank_identity': loss_rank_identity,
            'contrastive': loss_contrast
        }
        
        return total_loss, loss_dict


def get_loss_function(loss_type='multitask', lambda_mse=0.3, lambda_rank=1.0, 
                     lambda_contrast=0.5, contrastive_type='raw_id'):
    """获取损失函数"""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'multitask':
        return MultiTaskLoss(
            lambda_mse=lambda_mse, 
            lambda_rank=lambda_rank, 
            lambda_contrast=lambda_contrast,
            contrastive_type=contrastive_type
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
