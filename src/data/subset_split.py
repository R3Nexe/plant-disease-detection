import os
import shutil
import argparse
import random
from pathlib import Path


# This piece of code splits the dataset into train val and test splits
# first 70%: train
# next 15% validation
# last 15%: test

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in input_dir.iterdir() if d.is_dir()]

    for cls in classes:
        files = list((input_dir / cls).glob("*.jpg")) + list((input_dir / cls).glob("*.JPG")) + list((input_dir / cls).glob("*.png"))
        random.shuffle(files)

        n_total = len(files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for split, split_files in splits.items():
            split_dir = output_dir / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, split_dir / f.name)

    print(f"✅ Dataset split into train/val/test at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to output dataset")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.input, args.output, args.train, args.val, args.seed)
