"""
Dataset definitions for multi-task regression.
"""
import os
from typing import List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiTaskDataset(Dataset):
    """Multi-task dataset supporting both single-branch and dual-branch modes."""

    def __init__(
        self,
        generated_paths: Sequence[str],
        quality_scores: Sequence[float],
        identity_scores: Sequence[float],
        raw_ids: Sequence[int],
        raw_dir: str,
        transform: Optional[transforms.Compose] = None,
        use_raw_branch: bool = False,
    ):
        """
        Args:
            generated_paths: List of generated image paths.
            quality_scores: List of quality scores.
            identity_scores: List of identity-consistency scores.
            raw_ids: Matching raw image IDs.
            raw_dir: Directory that stores raw images (RAW/).
            transform: Optional torchvision transforms to apply.
            use_raw_branch: If True, load raw images for dual-branch model.
        """
        self.generated_paths = list(generated_paths)
        self.quality_scores = list(quality_scores)
        self.identity_scores = list(identity_scores)
        self.raw_ids = list(raw_ids)
        self.raw_dir = raw_dir
        self.transform = transform
        self.use_raw_branch = use_raw_branch
        
        # Build raw image path mapping
        self.raw_paths = {}
        if use_raw_branch:
            for raw_id in set(raw_ids):
                # Try different extensions
                for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                    raw_path = os.path.join(raw_dir, f"{raw_id}{ext}")
                    if os.path.exists(raw_path):
                        self.raw_paths[raw_id] = raw_path
                        break
                
                if raw_id not in self.raw_paths:
                    print(f"Warning: Raw image {raw_id} not found in {raw_dir}")

    def __len__(self) -> int:
        return len(self.generated_paths)

    def __getitem__(self, idx: int):
        gen_path = self.generated_paths[idx]
        gen_image = Image.open(gen_path).convert("RGB")

        if self.transform:
            gen_image = self.transform(gen_image)

        quality_score = torch.tensor(self.quality_scores[idx], dtype=torch.float32)
        identity_score = torch.tensor(self.identity_scores[idx], dtype=torch.float32)

        raw_id = torch.tensor(self.raw_ids[idx], dtype=torch.long)
        
        if self.use_raw_branch:
            # Load corresponding raw image
            raw_id_val = self.raw_ids[idx]
            raw_path = self.raw_paths.get(raw_id_val)
            
            if raw_path:
                raw_image = Image.open(raw_path).convert("RGB")
                if self.transform:
                    raw_image = self.transform(raw_image)
            else:
                # Fallback: use generated image as raw (shouldn't happen)
                raw_image = gen_image
            
            return gen_image, raw_image, quality_score, identity_score, raw_id
        else:
            return gen_image, quality_score, identity_score, raw_id


def get_transforms(img_size: int = 224, is_train: bool = True, use_augmentation: bool = True):
    """
    Build standard transforms.

    Note: keep augmentations mild for perceptual quality tasks.
    """
    if is_train and use_augmentation:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.3),  # 轻微翻转
                transforms.ColorJitter(
                    brightness=0.05, contrast=0.05, saturation=0.05  # 减小颜色扰动
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return transform
