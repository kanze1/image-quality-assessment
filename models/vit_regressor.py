"""
Vision Transformer based regressor with reference-guided architecture.
核心思想：直接建模生成图像相对于原始图像的差异
"""
import torch
import torch.nn as nn
import timm


class ReferenceGuidedViT(nn.Module):
    """
    参考引导的ViT架构
    
    核心设计理念：
    1. 共享backbone提取特征（避免参数冗余）
    2. 直接建模特征差异（gen - raw）
    3. 结合绝对特征和相对差异进行预测
    
    优势：
    - 参数高效：只有一个ViT backbone
    - 计算高效：可以batch处理两类图像
    - 语义清晰：显式建模"相对于原始图像的质量差异"
    """
    
    def __init__(
        self,
        model_name='vit_base_patch16_224',
        pretrained=False,
        embedding_dim=256,
        face_pretrained_path=None,
        freeze_backbone=False,
    ):
        super().__init__()
        
        # 共享backbone
        # 处理 hf-hub: 前缀
        if model_name.startswith('hf-hub:'):
            actual_model_name = model_name.replace('hf-hub:', '')
        else:
            actual_model_name = model_name
            
        self.backbone = timm.create_model(
            actual_model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        # 冻结backbone（小数据集关键策略）
        if freeze_backbone:
            print("🔒 冻结backbone参数，只训练任务头")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 加载人脸预训练权重
        if face_pretrained_path and face_pretrained_path.lower() != 'none':
            print(f"加载人脸预训练权重: {face_pretrained_path}")
            try:
                state_dict = torch.load(face_pretrained_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                self.backbone.load_state_dict(state_dict, strict=False)
                print("✓ 权重加载成功")
            except Exception as e:
                print(f"⚠ 警告: 权重加载失败: {e}")
        
        self.feature_dim = self.backbone.num_features
        
        # 差异建模层：学习如何利用特征差异
        self.diff_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 特征融合：结合绝对特征和相对差异
        # 输入：[gen_feat, raw_feat, diff_feat] -> 3 * feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim + 512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # 任务头
        self.quality_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        
        self.identity_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
    
    def forward(self, gen_images, raw_images, return_embedding=False):
        """
        Args:
            gen_images: Generated images (batch, 3, H, W)
            raw_images: Raw reference images (batch, 3, H, W)
            return_embedding: Whether to return embedding
        
        Returns:
            quality_pred, identity_pred, [embeddings]
        """
        batch_size = gen_images.size(0)
        
        # 高效batch处理：将生成图像和原始图像拼接在一起
        # [gen_1, gen_2, ..., gen_n, raw_1, raw_2, ..., raw_n]
        combined_images = torch.cat([gen_images, raw_images], dim=0)  # (2*batch, 3, H, W)
        
        # 一次forward提取所有特征
        combined_features = self.backbone(combined_images)  # (2*batch, feature_dim)
        
        # 分离生成图像和原始图像的特征
        gen_feat = combined_features[:batch_size]  # (batch, feature_dim)
        raw_feat = combined_features[batch_size:]  # (batch, feature_dim)
        
        # 计算特征差异（核心：显式建模相对差异）
        diff_feat = gen_feat - raw_feat  # (batch, feature_dim)
        
        # 编码差异特征
        diff_encoded = self.diff_encoder(diff_feat)  # (batch, 512)
        
        # 融合：生成图像特征 + 差异特征
        # 原始图像特征作为"锚点"已经隐含在差异中，不需要显式使用
        fused_feat = torch.cat([gen_feat, diff_encoded], dim=1)  # (batch, feature_dim + 512)
        
        # 获取最终embedding
        embedding = self.fusion(fused_feat)  # (batch, embedding_dim)
        embedding_norm = nn.functional.normalize(embedding, p=2, dim=1)
        
        # 预测分数
        quality_pred = self.quality_head(embedding).squeeze(-1)
        identity_pred = self.identity_head(embedding).squeeze(-1)
        
        if return_embedding:
            return quality_pred, identity_pred, embedding_norm
        else:
            return quality_pred, identity_pred


class SingleBranchViT(nn.Module):
    """Single-branch ViT (baseline, only uses generated images)."""
    
    def __init__(
        self,
        model_name='vit_base_patch16_224',
        pretrained=False,
        embedding_dim=256,
        face_pretrained_path=None,
    ):
        super().__init__()
        
        # 处理 hf-hub: 前缀
        if model_name.startswith('hf-hub:'):
            actual_model_name = model_name.replace('hf-hub:', '')
        else:
            actual_model_name = model_name
            
        self.backbone = timm.create_model(
            actual_model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        
        if face_pretrained_path and face_pretrained_path.lower() != 'none':
            print(f"加载人脸预训练权重: {face_pretrained_path}")
            try:
                state_dict = torch.load(face_pretrained_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                self.backbone.load_state_dict(state_dict, strict=False)
                print("✓ 权重加载成功")
            except Exception as e:
                print(f"⚠ 警告: 权重加载失败: {e}")
        
        self.feature_dim = self.backbone.num_features
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        
        self.identity_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
    
    def forward(self, gen_images, return_embedding=False):
        feat = self.backbone(gen_images)
        embedding = self.embedding_layer(feat)
        embedding_norm = nn.functional.normalize(embedding, p=2, dim=1)
        
        quality_pred = self.quality_head(embedding).squeeze(-1)
        identity_pred = self.identity_head(embedding).squeeze(-1)
        
        if return_embedding:
            return quality_pred, identity_pred, embedding_norm
        else:
            return quality_pred, identity_pred
