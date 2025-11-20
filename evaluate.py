"""
评估脚本 - 双分支架构
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data.dataset import MultiTaskDataset, get_transforms
from data.preprocess import load_data, split_data
from models.vit_regressor import ReferenceGuidedViT
from utils.metrics import evaluate_all_metrics, print_metrics
from utils.visualization import plot_scatter, plot_error_distribution


def evaluate(model, dataloader, device):
    model.eval()

    all_quality_pred = []
    all_quality_target = []
    all_identity_pred = []
    all_identity_target = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估"):
            gen_images, raw_images, quality_scores, identity_scores = batch
            gen_images = gen_images.to(device)
            raw_images = raw_images.to(device)

            quality_pred, identity_pred = model(gen_images, raw_images)

            all_quality_pred.extend(quality_pred.cpu().numpy())
            all_quality_target.extend(quality_scores.numpy())
            all_identity_pred.extend(identity_pred.cpu().numpy())
            all_identity_target.extend(identity_scores.numpy())

    return (
        np.array(all_quality_pred),
        np.array(all_quality_target),
        np.array(all_identity_pred),
        np.array(all_identity_target),
    )


def main(args):
    cfg = Config()

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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"\n加载模型: {args.checkpoint}")

    test_transform = get_transforms(cfg.IMG_SIZE, is_train=False)
    test_dataset = MultiTaskDataset(
        test_paths,
        test_quality,
        test_identity,
        test_raw_ids,
        cfg.RAW_DIR,
        transform=test_transform,
        use_raw_branch=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    model = ReferenceGuidedViT(
        model_name=cfg.MODEL_NAME,
        pretrained=False,
        embedding_dim=cfg.EMBEDDING_DIM,
    )
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if "best_avg_plcc" in checkpoint:
        print(f"模型最佳平均PLCC: {checkpoint['best_avg_plcc']:.4f}")

    print("\n在测试集上评估...")
    quality_pred, quality_target, identity_pred, identity_target = evaluate(
        model, test_loader, device
    )

    quality_metrics = evaluate_all_metrics(quality_pred, quality_target)
    identity_metrics = evaluate_all_metrics(identity_pred, identity_target)

    print(f"\n{'='*60}")
    print("测试集结果")
    print(f"{'='*60}")
    print_metrics(quality_metrics, prefix="质量评估")
    print_metrics(identity_metrics, prefix="人脸一致性")

    avg_plcc = (quality_metrics["PLCC"] + identity_metrics["PLCC"]) / 2
    avg_srcc = (quality_metrics["SRCC"] + identity_metrics["SRCC"]) / 2
    print(f"\n综合指标:")
    print(f"  平均PLCC: {avg_plcc:.4f}")
    print(f"  平均SRCC: {avg_srcc:.4f}")

    os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    scatter_path = os.path.join(cfg.RESULT_DIR, "quality_scatter.png")
    plot_scatter(
        quality_pred,
        quality_target,
        save_path=scatter_path,
        title="质量评估 - 预测值 vs 真实值",
    )
    print(f"\n质量评估散点图: {scatter_path}")

    error_path = os.path.join(cfg.RESULT_DIR, "quality_error_dist.png")
    plot_error_distribution(quality_pred, quality_target, save_path=error_path)
    print(f"质量评估误差分布: {error_path}")

    scatter_path = os.path.join(cfg.RESULT_DIR, "identity_scatter.png")
    plot_scatter(
        identity_pred,
        identity_target,
        save_path=scatter_path,
        title="人脸一致性 - 预测值 vs 真实值",
    )
    print(f"\n人脸一致性散点图: {scatter_path}")

    error_path = os.path.join(cfg.RESULT_DIR, "identity_error_dist.png")
    plot_error_distribution(identity_pred, identity_target, save_path=error_path)
    print(f"人脸一致性误差分布: {error_path}")

    results_file = os.path.join(cfg.RESULT_DIR, "test_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("测试集结果\n")
        f.write("="*60 + "\n\n")
        
        f.write("质量评估指标:\n")
        for key, value in quality_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\n人脸一致性指标:\n")
        for key, value in identity_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write(f"\n综合指标:\n")
        f.write(f"  平均PLCC: {avg_plcc:.4f}\n")
        f.write(f"  平均SRCC: {avg_srcc:.4f}\n")
    
    print(f"\n结果已保存: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估人脸质量评估模型")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="模型checkpoint路径",
    )

    args = parser.parse_args()
    main(args)
