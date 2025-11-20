"""
训练脚本 - 双分支架构（引入原始图像先验）
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from data.dataset import MultiTaskDataset, get_transforms
from data.preprocess import load_data, split_data
from models.vit_regressor import ReferenceGuidedViT
from models.losses import MultiTaskLoss
from utils.metrics import evaluate_all_metrics, print_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=2):
    model.train()
    total_loss = 0.0
    loss_components = {
        "mse_quality": 0.0,
        "mse_identity": 0.0,
        "rank_quality": 0.0,
        "rank_identity": 0.0,
        "contrastive": 0.0,
    }

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="训练")
    
    for batch_idx, batch in enumerate(pbar):
        gen_images, raw_images, quality_scores, identity_scores, raw_ids = batch
        gen_images = gen_images.to(device)
        raw_images = raw_images.to(device)
        quality_scores = quality_scores.to(device)
        identity_scores = identity_scores.to(device)
        raw_ids = raw_ids.to(device)

        quality_pred, identity_pred, embeddings = model(
            gen_images, raw_images, return_embedding=True
        )

        loss, loss_dict = criterion(
            quality_pred,
            identity_pred,
            quality_scores,
            identity_scores,
            embeddings,
            raw_ids,
        )
        
        # 梯度累积：模拟更大的batch size
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        for key in loss_components:
            loss_components[key] += loss_dict[key].item()

        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)

    return avg_loss, loss_components


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_quality_pred = []
    all_quality_target = []
    all_identity_pred = []
    all_identity_target = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证"):
            gen_images, raw_images, quality_scores, identity_scores, raw_ids = batch
            gen_images = gen_images.to(device)
            raw_images = raw_images.to(device)
            quality_scores = quality_scores.to(device)
            identity_scores = identity_scores.to(device)
            raw_ids = raw_ids.to(device)

            quality_pred, identity_pred, embeddings = model(
                gen_images, raw_images, return_embedding=True
            )

            loss, _ = criterion(
                quality_pred,
                identity_pred,
                quality_scores,
                identity_scores,
                embeddings,
                raw_ids,
            )

            total_loss += loss.item()
            all_quality_pred.extend(quality_pred.cpu().numpy())
            all_quality_target.extend(quality_scores.cpu().numpy())
            all_identity_pred.extend(identity_pred.cpu().numpy())
            all_identity_target.extend(identity_scores.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    quality_metrics = evaluate_all_metrics(all_quality_pred, all_quality_target)
    identity_metrics = evaluate_all_metrics(all_identity_pred, all_identity_target)

    return avg_loss, quality_metrics, identity_metrics


def main():
    cfg = Config()

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("\n加载数据...")
    image_paths, quality_scores, identity_scores, raw_ids = load_data(
        cfg.SCORE_FILE, cfg.ALL_DIR
    )

    (
        train_paths,
        train_quality,
        train_identity,
        train_raw_ids,
        val_paths,
        val_quality,
        val_identity,
        val_raw_ids,
        test_paths,
        test_quality,
        test_identity,
        test_raw_ids,
    ) = split_data(
        image_paths,
        quality_scores,
        identity_scores,
        raw_ids,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        random_seed=cfg.RANDOM_SEED,
    )

    train_transform = get_transforms(
        cfg.IMG_SIZE, is_train=True, use_augmentation=cfg.USE_AUGMENTATION
    )
    val_transform = get_transforms(cfg.IMG_SIZE, is_train=False)

    train_dataset = MultiTaskDataset(
        train_paths,
        train_quality,
        train_identity,
        train_raw_ids,
        cfg.RAW_DIR,
        transform=train_transform,
        use_raw_branch=True,
    )
    val_dataset = MultiTaskDataset(
        val_paths,
        val_quality,
        val_identity,
        val_raw_ids,
        cfg.RAW_DIR,
        transform=val_transform,
        use_raw_branch=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"\n创建参考引导模型: {cfg.MODEL_NAME}")
    print("  架构: 共享backbone + 差异建模")
    print(f"  预训练: {cfg.PRETRAINED}")
    print(f"  冻结backbone: {cfg.FREEZE_BACKBONE}")
    
    model = ReferenceGuidedViT(
        model_name=cfg.MODEL_NAME,
        pretrained=cfg.PRETRAINED,
        embedding_dim=cfg.EMBEDDING_DIM,
        face_pretrained_path=cfg.FACE_PRETRAINED_PATH,
        freeze_backbone=cfg.FREEZE_BACKBONE,
    )
    model = model.to(device)
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    criterion = MultiTaskLoss(
        lambda_mse=cfg.LAMBDA_MSE,
        lambda_rank=cfg.LAMBDA_RANK,
        lambda_contrast=cfg.LAMBDA_CONTRAST,
        contrastive_type=cfg.CONTRASTIVE_TYPE
    )

    # 优化器：只优化需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    
    # 学习率调度：Warmup + Cosine Annealing
    from torch.optim.lr_scheduler import LambdaLR
    
    def warmup_cosine_schedule(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            # Warmup阶段：线性增长
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        else:
            # Cosine annealing
            progress = (epoch - cfg.WARMUP_EPOCHS) / (cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

    print(f"\n开始训练 (共 {cfg.NUM_EPOCHS} 个epoch)...")
    print(f"损失权重: λ_mse={cfg.LAMBDA_MSE}, λ_rank={cfg.LAMBDA_RANK}, λ_contrast={cfg.LAMBDA_CONTRAST}")
    print(f"对比学习类型: {cfg.CONTRASTIVE_TYPE}")

    best_avg_plcc = -1.0
    best_epoch = 0
    patience = 20  # 早停耐心值
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    quality_plccs = []
    identity_plccs = []

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.NUM_EPOCHS}")
        print(f"{'='*60}")

        train_loss, loss_components = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        print(f"\n训练损失: {train_loss:.4f}")
        print(f"  MSE (质量): {loss_components['mse_quality']:.4f}")
        print(f"  MSE (一致性): {loss_components['mse_identity']:.4f}")
        print(f"  Ranking (质量): {loss_components['rank_quality']:.4f}")
        print(f"  Ranking (一致性): {loss_components['rank_identity']:.4f}")
        print(f"  Contrastive: {loss_components['contrastive']:.4f}")

        val_loss, quality_metrics, identity_metrics = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"\n验证损失: {val_loss:.4f}")
        print_metrics(quality_metrics, prefix="质量评估")
        print_metrics(identity_metrics, prefix="人脸一致性")

        avg_plcc = (quality_metrics["PLCC"] + identity_metrics["PLCC"]) / 2

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        quality_plccs.append(quality_metrics["PLCC"])
        identity_plccs.append(identity_metrics["PLCC"])

        if avg_plcc > best_avg_plcc:
            best_avg_plcc = avg_plcc
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_avg_plcc": best_avg_plcc,
                    "quality_metrics": quality_metrics,
                    "identity_metrics": identity_metrics,
                },
                checkpoint_path,
            )
            print(f"\n✓ 保存最佳模型 (平均PLCC: {best_avg_plcc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发！{patience}个epoch无改善")
                print(f"最佳epoch: {best_epoch}, 最佳PLCC: {best_avg_plcc:.4f}")
                break

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"最佳平均PLCC: {best_avg_plcc:.4f}")
    print(f"{'='*60}")

    from utils.visualization import plot_training_curves
    
    curves_path = os.path.join(cfg.RESULT_DIR, "training_curves.png")
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    plot_training_curves(
        train_losses, val_losses, quality_plccs, identity_plccs, save_path=curves_path
    )


if __name__ == "__main__":
    main()
