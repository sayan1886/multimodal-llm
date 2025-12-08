#!/usr/bin/env python3
"""Create a smaller Flickr8k subset by sampling images and their captions.

Usage:
  python scripts/downsample_flickr8k.py --src multimodal-dataset/flickr8k \
      --dst multimodal-dataset/flickr8k-subset --num 1000 --symlink

This script reads `captions.txt` (CSV with header `image,caption`),
samples up to `--num` unique images, creates `dst/images/` and
symlinks (or copies) the selected image files, and writes a new
`captions.txt` containing only caption lines for the sampled images.
"""

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Downsample Flickr8k captions and images")
    p.add_argument("--src", default="multimodal-dataset/flickr8k", help="Source Flickr8k folder")
    p.add_argument("--dst", default="multimodal-dataset/flickr8k-subset", help="Destination subset folder")
    p.add_argument("--num", type=int, default=1000, help="Number of unique images to sample")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying image files")
    p.add_argument("--force", action="store_true", help="Remove existing destination if present")
    return p.parse_args()


def read_captions(captions_path):
    items = []  # list of (image, caption)
    with open(captions_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # If header doesn't look like image,caption, treat it as data
        if header and len(header) >= 2 and header[0].strip().lower() == "image":
            pass
        else:
            # first row was data
            if header:
                items.append((header[0], header[1] if len(header) > 1 else ""))
        for row in reader:
            if not row:
                continue
            img = row[0]
            cap = row[1] if len(row) > 1 else ""
            items.append((img, cap))
    return items


def main():
    args = parse_args()
    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    captions_path = src / "captions.txt"
    images_dir = src / "images"

    if not captions_path.exists():
        print(f"Error: captions file not found at {captions_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Warning: images dir not found at {images_dir} (images may be elsewhere)")

    items = read_captions(captions_path)
    unique_images = []
    seen = set()
    for img, _ in items:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)

    total_images = len(unique_images)
    k = min(args.num, total_images)
    if k <= 0:
        print("No images to sample (num <= 0 or no images found). Exiting.")
        sys.exit(1)

    sampled = set(random.sample(unique_images, k))

    # Prepare destination
    if dst.exists():
        if args.force:
            shutil.rmtree(dst)
        else:
            print(f"Error: destination {dst} already exists. Use --force to overwrite.")
            sys.exit(1)
    (dst / "images").mkdir(parents=True, exist_ok=True)

    # Link or copy images
    found = 0
    missing_images = []
    for img in sampled:
        src_img = images_dir / img
        dst_img = dst / "images" / img
        if src_img.exists():
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            try:
                if args.symlink:
                    os.symlink(os.path.abspath(src_img), str(dst_img))
                else:
                    shutil.copy2(src_img, dst_img)
                found += 1
            except OSError:
                # Fall back to copy if symlink fails
                shutil.copy2(src_img, dst_img)
                found += 1
        else:
            missing_images.append(img)

    # Write filtered captions.txt preserving all caption lines for sampled images
    out_captions = dst / "captions.txt"
    with open(out_captions, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        written_lines = 0
        for img, cap in items:
            if img in sampled and img not in missing_images:
                writer.writerow([img, cap])
                written_lines += 1

    print(f"Selected {k} images out of {total_images} unique images")
    print(f"Images found and copied/symlinked: {found}")
    if missing_images:
        print(f"Images missing in source images dir: {len(missing_images)} (they were skipped)")
    print(f"Wrote {written_lines} caption lines to {out_captions}")
    print(f"Done. New dataset at {dst}")


if __name__ == "__main__":
    main()
