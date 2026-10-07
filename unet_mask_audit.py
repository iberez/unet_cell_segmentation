"""List, for every mask, whether its cells are outlined or marked with dots.

    python3 unet_mask_audit.py

Reads only the masks (no model, no GPU). Prints one line per section and the totals per split, and
saves the table as mask_audit.csv in the data directory. unet_training_dict.py and
unet_evaluation_dict.py must sit next to this file.

A mask counts as "dots" when (nearly) all of its objects have one and the same size, which is what
point markers stamped at cell positions look like; otherwise it counts as "outlines".
"""

import csv
import os

import numpy as np
from scipy import ndimage

import unet_training_dict as tr
from unet_evaluation_dict import STRUCTURE, annotation_type


def main():
    data_dir = tr.DATA_DIR
    images, segs = tr.list_pairs(data_dir)

    split_of = {}
    split_path = os.path.join(data_dir, tr.SPLIT_FILE)
    if os.path.exists(split_path):
        for split, members in tr.split_by_slide(images, segs, split_path).items():
            for item in members:
                split_of[item["label"]] = split

    rows = []
    print(f"\nchecking {len(segs)} masks ({tr.MASK_KEY})...")
    for image, seg in zip(images, segs):
        mask = tr.load_mask(seg)[0] > 0.5
        labels, n_cells = ndimage.label(mask, structure=STRUCTURE)
        areas = np.bincount(labels.ravel())[1:]
        sizes, counts = np.unique(areas, return_counts=True) if n_cells else (np.array([0]), np.array([0]))
        rows.append(
            {
                "section": tr.section_key(image),
                "slide": tr.slide_of(image),
                "split": split_of.get(seg, "?"),
                "annotation": annotation_type(areas),
                "cells": n_cells,
                "median_area_px": float(np.median(areas)) if n_cells else 0.0,
                "most_common_area_px": int(sizes[counts.argmax()]),
                "share_with_that_area": round(float(counts.max()) / n_cells, 2) if n_cells else 0.0,
                "size_trimmed": seg in tr.CROP_TO,
            }
        )
        row = rows[-1]
        print(
            f"  {row['section']:<26} {row['split']:<5} {row['annotation']:<8} {n_cells:>5} cells, median area "
            f"{row['median_area_px']:>6.0f} px, {100 * row['share_with_that_area']:>3.0f}% are exactly "
            f"{row['most_common_area_px']} px{'   (size-trimmed pair)' if row['size_trimmed'] else ''}"
        )

    out_path = os.path.join(data_dir, "mask_audit.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    kinds = ["outlines", "dots", "empty"]
    print("\nSECTIONS (cells) PER SPLIT")
    print(f"  {'split':<6}" + "".join(f"{kind:>18}" for kind in kinds))
    for split in sorted({row["split"] for row in rows}, key=["train", "val", "test", "?"].index):
        cells = []
        for kind in kinds:
            group = [row for row in rows if row["split"] == split and row["annotation"] == kind]
            cells.append(f"{len(group)} ({sum(row['cells'] for row in group)})")
        print(f"  {split:<6}" + "".join(f"{text:>18}" for text in cells))

    print("\nPER SLIDE (outlines / dots / empty)")
    for slide in sorted({row["slide"] for row in rows}):
        group = [row for row in rows if row["slide"] == slide]
        tally = " / ".join(str(sum(row["annotation"] == kind for row in group)) for kind in kinds)
        print(f"  {slide} [{group[0]['split']}]: {tally}")

    trimmed = [row for row in rows if row["size_trimmed"]]
    if trimmed:
        print(f"\nof the {len(trimmed)} size-trimmed pairs, {sum(row['annotation'] == 'dots' for row in trimmed)} are dot masks")
    print(f"\ntable saved to {out_path}")


if __name__ == "__main__":
    main()
