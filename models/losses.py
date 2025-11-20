"""
端到端多任务损失函数 - 质量 + ID 一致性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class LossConfig:
    # 回归权重
    w_q: float = 1.0
    w_id: float = 1.0
    # PLCC 权重（趋势相关性）
    lambda_plcc: float = 0.1
    # SupCon 权重（同 raw_id）
    lambda_sup: float = 0.05
    # ranking 权重（ID 排序）
    lambda_rank: float = 0.05
    # SupCon 温度
    supcon_temperature: float = 0.1
    # ranking margin
    rank_margin: float = 0.1


class IQALoss(nn.Module):
    """
    适用于：一次输出两个分数（质量 + ID 一致性）的端到端模型。
    
    输入：
        q_pred:  [B] 或 [B, 1]
        id_pred: [B] 或 [B, 1]
        q_gt:    [B]
        id_gt:   [B]
        feats:   [B, D] 中间特征
        raw_ids: [B] int，同一 raw_id 表示来自同一原始人脸
    """
    def __init__(self, config: LossConfig = LossConfig()):
        super().__init__()
        self.cfg = config
        self.huber = nn.SmoothL1Loss(reduction='mean')  # Huber loss
    
    def forward(self,
                q_pred: torch.Tensor,
                id_pred: torch.Tensor,
                q_gt: torch.Tensor,
                id_gt: torch.Tensor,
                feats: torch.Tensor,
                raw_ids: torch.Tensor):
        # 保证形状一致：[B]
        q_pred = q_pred.view(-1)
        id_pred = id_pred.view(-1)
        q_gt = q_gt.view(-1)
        id_gt = id_gt.view(-1)
        raw_ids = raw_ids.view(-1)
        
        # ===== 1. 回归损失（质量 + ID） =====
        loss_q = self.huber(q_pred, q_gt)
        loss_id = self.huber(id_pred, id_gt)
        loss_reg = self.cfg.w_q * loss_q + self.cfg.w_id * loss_id
        
        # ===== 2. PLCC 损失（趋势相关性） =====
        loss_plcc_q = self.plcc_loss(q_pred, q_gt)
        loss_plcc_id = self.plcc_loss(id_pred, id_gt)
        loss_plcc = loss_plcc_q + loss_plcc_id
        loss_plcc = self.cfg.lambda_plcc * loss_plcc
        
        # ===== 3. SupCon 损失（同 raw_id 为正样本） =====
        loss_sup = self.supcon_loss(feats, raw_ids)
        loss_sup = self.cfg.lambda_sup * loss_sup
        
        # ===== 4. ID 排序损失（同 raw 内高分 > 低分） =====
        loss_rank_id = self.id_ranking_loss(id_pred, id_gt, raw_ids)
        loss_rank_id = self.cfg.lambda_rank * loss_rank_id
        
        # ===== 5. 总损失 =====
        loss_total = loss_reg + loss_plcc + loss_sup + loss_rank_id
        
        # 方便 log：返回一个 dict
        loss_dict = {
            'total': loss_total,
            'reg': loss_reg,
            'q_reg': loss_q,
            'id_reg': loss_id,
            'plcc': loss_plcc,
            'plcc_q': loss_plcc_q,
            'plcc_id': loss_plcc_id,
            'supcon': loss_sup,
            'rank_id': loss_rank_id
        }
        
        return loss_dict
    
    # ---------- 工具函数们 ----------
    @staticmethod
    def plcc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """batch 内 Pearson 相关系数的损失: L = 1 - ρ"""
        x = pred
        y = target
        x = x - x.mean()
        y = y - y.mean()
        vx = x.pow(2).mean()
        vy = y.pow(2).mean()
        corr = (x * y).mean() / (vx.sqrt() * vy.sqrt() + eps)
        return 1.0 - corr
    
    def supcon_loss(self, feats: torch.Tensor, raw_ids: torch.Tensor) -> torch.Tensor:
        """
        Supervised Contrastive Loss：同一个 raw_id 作为正样本，不同 raw_id 作为负样本。
        """
        device = feats.device
        B = feats.size(0)
        
        # 归一化特征
        feats = F.normalize(feats, dim=1)
        
        # 构造 [B, B] 的相似度矩阵
        sim_matrix = torch.matmul(feats, feats.t())  # cosine sim 因为已 normalize
        
        # 构造 mask: 同 raw_id 且 i!=j 为正样本
        raw_ids = raw_ids.view(-1, 1)
        mask = torch.eq(raw_ids, raw_ids.t()).float().to(device)
        # 去掉对角线（与自己不算正样本）
        mask = mask - torch.eye(B, device=device)
        
        # logits / temperature
        logits = sim_matrix / self.cfg.supcon_temperature
        
        # 对每个 i，计算：
        # -log( sum_{p in P(i)} exp(sim(i,p)/T) / sum_{k!=i} exp(sim(i,k)/T) )
        # denominator: sum over all k != i
        # 先构造一个 mask 去掉自己
        logits_mask = 1 - torch.eye(B, device=device)
        exp_logits = torch.exp(logits) * logits_mask  # [B, B]
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # 只对正样本位置求平均
        # 对每个 i，正样本数量为 mask[i].sum()
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 整个 batch 平均
        loss = -mean_log_prob_pos.mean()
        return loss
    
    def id_ranking_loss(self,
                       id_pred: torch.Tensor,
                       id_gt: torch.Tensor,
                       raw_ids: torch.Tensor) -> torch.Tensor:
        """
        同一个 raw_id 内做 margin ranking：
        如果 id_gt_i > id_gt_j，期望 id_pred_i > id_pred_j + margin
        """
        device = id_pred.device
        margin = self.cfg.rank_margin
        
        total_loss = 0.0
        count = 0
        
        # 遍历每一个 raw_id
        unique_raw_ids = raw_ids.unique()
        for rid in unique_raw_ids:
            idx = (raw_ids == rid).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue  # 只有一张没法排
            
            # 当前 raw 的 gt & pred
            gt = id_gt[idx]
            pred = id_pred[idx]
            
            # 构造所有 pair (i, j) 使得 gt_i > gt_j
            # 简单做法：两层循环，数据量不大（每个 raw 8 张）
            for i in range(len(idx)):
                for j in range(len(idx)):
                    if gt[i] > gt[j] + 1e-6:  # 避免相等
                        # 希望 pred[i] >= pred[j] + margin
                        diff = pred[i] - pred[j]
                        loss_ij = F.relu(margin - diff)
                        total_loss += loss_ij
                        count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / count
