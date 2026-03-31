"""
Phase A1: Data Preparation & Organization
Extracts TB Chest X-ray from raw structure and organizes into train/val/test splits with resizing.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_image_paths(class_dir: Path) -> List[Path]:
    """Get all PNG images from a class directory."""
    return sorted([f for f in class_dir.glob("*.png") if f.is_file()])


def split_indices(
    total_count: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split indices into train/val/test with given ratios.
    
    Args:
        total_count: Total number of samples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    indices = np.arange(total_count)
    np.random.shuffle(indices)
    
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    
    train_indices = indices[:train_count].tolist()
    val_indices = indices[train_count : train_count + val_count].tolist()
    test_indices = indices[train_count + val_count :].tolist()
    
    return train_indices, val_indices, test_indices


def resize_and_save_image(
    src_path: Path, dst_path: Path, size: Tuple[int, int] = (224, 224)
) -> bool:
    """
    Load, resize to 224x224, and save image.
    Grayscale images are converted to 3-channel RGB for compatibility with ResNet/VGG.
    
    Args:
        src_path: Source image path
        dst_path: Destination image path
        size: Target size (H, W)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(src_path)
        
        # Convert grayscale to RGB (TB X-rays are grayscale)
        if img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Save
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)
        return True
    except Exception as e:
        logger.warning(f"Failed to process {src_path}: {e}")
        return False


def prepare_tb_dataset(
    raw_dir: Path = config.TB_DATA_ROOT,
    output_dir: Path = config.TB_ORGANIZED_ROOT,
    img_size: int = config.IMG_SIZE,
    train_ratio: float = config.TRAIN_SPLIT,
    val_ratio: float = config.VAL_SPLIT,
    test_ratio: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """
    Main function to prepare TB dataset:
    1. Load images from raw structure (Normal/, Tuberculosis/)
    2. Split into train/val/test (70/15/15)
    3. Resize all images to 224x224
    4. Save organized structure
    
    Args:
        raw_dir: Path to raw TB_Chest_Radiography_Database folder
        output_dir: Path to output organized folder
        img_size: Target image size
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Testing split ratio
        seed: Random seed
    
    Returns:
        Dictionary with statistics (counts, distribution, etc.)
    """
    
    logger.info(f"Starting TB dataset preparation...")
    logger.info(f"Raw data: {raw_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Check raw data exists
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dir}")
    
    class_dirs = {
        "Normal": raw_dir / "Normal",
        "Tuberculosis": raw_dir / "Tuberculosis",
    }
    
    # Verify class directories
    for class_name, class_dir in class_dirs.items():
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        logger.info(f"Found '{class_name}' directory: {class_dir}")
    
    # Collect all images per class
    class_images = {}
    total_images = 0
    for class_name, class_dir in class_dirs.items():
        images = get_image_paths(class_dir)
        class_images[class_name] = images
        total_images += len(images)
        logger.info(f"  {class_name}: {len(images)} images")
    
    logger.info(f"Total images: {total_images}")
    
    # Split each class independently (stratified split)
    split_data = {}
    for class_name, images in class_images.items():
        train_idx, val_idx, test_idx = split_indices(
            len(images), train_ratio, val_ratio, test_ratio, seed
        )
        
        split_data[class_name] = {
            "train": [images[i] for i in train_idx],
            "val": [images[i] for i in val_idx],
            "test": [images[i] for i in test_idx],
        }
        logger.info(
            f"{class_name} split: train={len(split_data[class_name]['train'])}, "
            f"val={len(split_data[class_name]['val'])}, "
            f"test={len(split_data[class_name]['test'])}"
        )
    
    # Process and save images
    stats = {"processed": 0, "failed": 0, "splits": {}}
    
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        split_stats = {}
        
        for class_name in class_images.keys():
            class_split_dir = split_dir / class_name
            images_to_process = split_data[class_name][split_name]
            
            processed = 0
            for i, src_path in enumerate(images_to_process):
                dst_path = class_split_dir / src_path.name
                
                if resize_and_save_image(src_path, dst_path, (img_size, img_size)):
                    processed += 1
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1
                
                if (i + 1) % 500 == 0:
                    logger.info(
                        f"  {split_name}/{class_name}: {i + 1}/{len(images_to_process)} processed"
                    )
            
            split_stats[class_name] = processed
            logger.info(f"{split_name}/{class_name}: {processed} files processed")
        
        stats["splits"][split_name] = split_stats
    
    logger.info(f"\nPreparation complete!")
    logger.info(f"  Total processed: {stats['processed']}")
    logger.info(f"  Total failed: {stats['failed']}")
    logger.info(f"  Output structure: {output_dir}")
    
    return stats


def verify_prepared_dataset(output_dir: Path = config.TB_ORGANIZED_ROOT) -> dict:
    """
    Verify the prepared dataset structure and count files.
    
    Args:
        output_dir: Path to organized output folder
    
    Returns:
        Dictionary with verification details
    """
    logger.info(f"\nVerifying prepared dataset...")
    
    verification = {}
    
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        split_verification = {}
        
        for class_name in ["Normal", "Tuberculosis"]:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png")))
                split_verification[class_name] = count
            else:
                split_verification[class_name] = 0
        
        verification[split] = split_verification
        total = sum(split_verification.values())
        logger.info(f"{split}: Normal={split_verification['Normal']}, TB={split_verification['Tuberculosis']}, Total={total}")
    
    return verification


if __name__ == "__main__":
    # Run data preparation
    stats = prepare_tb_dataset()
    
    # Verify
    verification = verify_prepared_dataset()
