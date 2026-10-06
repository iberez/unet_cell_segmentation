"""Evaluate the trained U-Net on held-out sections.

    python3 unet_evaluation_dict.py                # validation slides: use this while developing
    python3 unet_evaluation_dict.py --split test   # test slides: once, at the very end

Everything that has to match training (file pairing, size check, loaders, normalisation, network,
patch size, the slide split) is imported from unet_training_dict.py, which must sit next to this file.

Results go to <data directory>/evaluation/<split>_<date>_<time>/ :
    summary.txt            the numbers printed at the end
    sections.csv           the same counts, one row per section
    gallery.pdf            close-ups of missed cells, false alarms, and found cells
    sections_overview.pdf  one page per section showing where the hits, misses and false alarms are
    predicted_masks/       the predicted mask of every section as a PNG (0 = background, 255 = cell)
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from monai.inferers import sliding_window_inference

import unet_training_dict as tr  # also sets matplotlib's file-only backend and the PDF font settings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

THRESHOLD = 0.5       # predicted probability above which a pixel counts as cell
SW_BATCH_SIZE = 16    # windows per forward pass (affects speed only)

# An annotated cell and a predicted object are compared by IoU = overlap / union of the two shapes.
FOUND_IOU = 0.1       # the cell counts as "found" from this IoU up ...
CLOSE_IOU = 0.5       # ... and as "found with a closely matching outline" from this IoU up

CROP = 128            # side of the gallery close-ups, in pixels
EXAMPLES_PER_ROW = 12
CATEGORIES = ["missed", "false alarm", "found, outline differs", "found, outline matches"]


def predict_section(model, image, device):
    """Predicted cell mask (Y, X, bool) for one normalised section (1, Y, X), tiled exactly as in training."""
    with torch.no_grad():
        inputs = torch.from_numpy(image)[None].to(device)  # (1, 1, Y, X)
        logits = sliding_window_inference(inputs, tr.PATCH_SIZE, SW_BATCH_SIZE, model)
        return (torch.sigmoid(logits)[0, 0] > THRESHOLD).cpu().numpy()


def match_objects(truth, predicted):
    """Compare annotated cells with predicted objects in one section.

    Both masks are split into connected objects. Every overlapping (cell, object) pair gets an IoU,
    and pairs are accepted best-first, each cell and each object being used at most once. So one
    large predicted blob covering three cells is credited with one of them, not all three.
    """
    structure = np.ones((3, 3), dtype=bool)  # pixels touching at a corner belong to the same object
    true_labels, n_true = ndimage.label(truth, structure=structure)
    pred_labels, n_pred = ndimage.label(predicted, structure=structure)
    true_area = np.bincount(true_labels.ravel(), minlength=n_true + 1)
    pred_area = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)

    # overlap of every (cell, object) pair that shares at least one pixel
    both = (true_labels > 0) & (pred_labels > 0)
    codes, intersection = np.unique(
        true_labels[both].astype(np.int64) * (n_pred + 1) + pred_labels[both], return_counts=True
    )
    cell, obj = codes // (n_pred + 1), codes % (n_pred + 1)
    iou = intersection / (true_area[cell] + pred_area[obj] - intersection)

    cell_iou = np.zeros(n_true + 1)             # IoU of each cell with the object matched to it (0 = none)
    cell_object = np.zeros(n_true + 1, dtype=int)
    object_taken = np.zeros(n_pred + 1, dtype=bool)
    for k in np.argsort(-iou):
        if iou[k] < FOUND_IOU:
            break
        if cell_object[cell[k]] == 0 and not object_taken[obj[k]]:
            cell_object[cell[k]], cell_iou[cell[k]], object_taken[obj[k]] = obj[k], iou[k], True

    def centres(labels):
        return np.array([[(s.start + s.stop) // 2 for s in box] for box in ndimage.find_objects(labels)]).reshape(-1, 2)

    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "cell_iou": cell_iou[1:],                       # per annotated cell
        "cell_area": true_area[1:],
        "matched_object_area": np.where(cell_object[1:] > 0, pred_area[cell_object[1:]], 0),  # per cell (0 if missed)
        "object_matched": object_taken[1:],             # per predicted object
        "cell_centres": centres(true_labels),           # (row, column)
        "object_centres": centres(pred_labels),
    }


def collect_examples(name, image, truth, predicted, match, rng, per_category=3):
    """A few close-ups per category from one section, for the gallery."""
    found = match["cell_iou"] >= FOUND_IOU
    close = match["cell_iou"] >= CLOSE_IOU
    candidates = {
        "missed": [(c, "") for c in match["cell_centres"][~found]],
        "false alarm": [(c, "") for c in match["object_centres"][~match["object_matched"]]],
        "found, outline differs": [
            (c, f"IoU {i:.2f}") for c, i in zip(match["cell_centres"][found & ~close], match["cell_iou"][found & ~close])
        ],
        "found, outline matches": [
            (c, f"IoU {i:.2f}") for c, i in zip(match["cell_centres"][close], match["cell_iou"][close])
        ],
    }
    examples = {category: [] for category in CATEGORIES}
    for category, items in candidates.items():
        for k in rng.permutation(len(items))[:per_category]:
            (row, col), note = items[k]
            r0 = int(np.clip(row - CROP // 2, 0, max(truth.shape[0] - CROP, 0)))
            c0 = int(np.clip(col - CROP // 2, 0, max(truth.shape[1] - CROP, 0)))
            window = (slice(r0, r0 + CROP), slice(c0, c0 + CROP))
            examples[category].append(
                {
                    "image": image[window].copy(),
                    "truth": truth[window].copy(),
                    "predicted": predicted[window].copy(),
                    "title": f"{name}\n{note}".strip(),
                }
            )
    return examples


def save_gallery(examples, path, rng):
    """One row of close-ups per category: annotated outline in green, predicted outline in dashed magenta."""
    fig, axes = plt.subplots(
        len(CATEGORIES), EXAMPLES_PER_ROW, figsize=(2 * EXAMPLES_PER_ROW, 2.3 * len(CATEGORIES)), squeeze=False
    )
    for row, category in enumerate(CATEGORIES):
        chosen = [examples[category][k] for k in rng.permutation(len(examples[category]))[:EXAMPLES_PER_ROW]]
        for col in range(EXAMPLES_PER_ROW):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{category}\n({len(examples[category])} collected)", fontsize=9)
            if col >= len(chosen):
                ax.set_frame_on(False)
                continue
            example = chosen[col]
            low, high = np.percentile(example["image"], [1, 99.8])
            ax.imshow(example["image"], cmap="gray", vmin=low, vmax=max(high, low + 1e-6), interpolation="nearest")
            if example["truth"].any():
                ax.contour(example["truth"], levels=[0.5], colors="lime", linewidths=1.2)
            if example["predicted"].any():
                ax.contour(example["predicted"], levels=[0.5], colors="magenta", linewidths=1.2, linestyles="--")
            ax.set_title(example["title"], fontsize=6)
    fig.suptitle(
        f"{CROP} x {CROP} pixel close-ups (contrast stretched per panel).  "
        "Solid green: annotated cell.  Dashed magenta: predicted cell.",
        fontsize=11,
    )
    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)


def add_overview_page(pdf, name, image, match, dice):
    """One page: the whole section, shrunk, with every annotated cell and predicted object marked."""
    step = max(1, max(image.shape) // 1200)
    found = match["cell_iou"] >= FOUND_IOU
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image[::step, ::step], cmap="gray", vmin=0, vmax=max(np.percentile(image, 99.9), 1e-6))
    for centres, style, label in [
        (match["cell_centres"][found], {"marker": "o", "color": "lime", "s": 14}, "found"),
        (match["cell_centres"][~found], {"marker": "x", "color": "red", "s": 40}, "missed"),
        (
            match["object_centres"][~match["object_matched"]],
            {"marker": "s", "facecolors": "none", "edgecolors": "magenta", "s": 40},
            "false alarm",
        ),
    ]:
        ax.scatter(centres[:, 1] / step, centres[:, 0] / step, label=f"{label} ({len(centres)})", linewidths=1.2, **style)
    ax.set_xticks([])
    ax.set_yticks([])
    dice_text = "no annotated cells" if np.isnan(dice) else f"pixel Dice {dice:.2f}"
    ax.set_title(f"{name}  ({dice_text})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def summarise(rows, cell_iou, cell_area, matched_area, split, weights_path):
    """The text summary, as a list of lines."""
    total = {key: sum(row[key] for row in rows) for key in ["cells", "objects", "found", "close", "false_alarms"]}
    total_pixels = {key: sum(row[key] for row in rows) for key in ["true_pixels", "predicted_pixels", "overlap_pixels"]}
    with_cells = [row for row in rows if row["cells"] > 0]
    empty = [row for row in rows if row["cells"] == 0]
    found_iou = cell_iou[cell_iou >= FOUND_IOU]

    def share(part, whole):
        return f"{100 * part / whole:.1f}%" if whole else "n/a"

    lines = [
        f"split: {split}   weights: {weights_path}   threshold: {THRESHOLD}",
        f"sections: {len(rows)} ({len(with_cells)} with annotated cells, {len(empty)} with an empty mask)",
        "",
        "PIXELS",
        f"  mean Dice per section (sections with cells only; the number in the training log): "
        f"{np.mean([row['dice'] for row in with_cells]) if with_cells else float('nan'):.4f}",
        f"  Dice over all pixels of all sections pooled: "
        f"{2 * total_pixels['overlap_pixels'] / max(total_pixels['true_pixels'] + total_pixels['predicted_pixels'], 1):.4f}",
        "",
        f"CELLS  (found = a predicted object overlaps the annotated cell with IoU >= {FOUND_IOU})",
        f"  annotated cells: {total['cells']}    predicted objects: {total['objects']}",
        f"  found:        {total['found']}  ({share(total['found'], total['cells'])} of annotated cells)",
        f"    of these, outline closely matching (IoU >= {CLOSE_IOU}): {total['close']}  "
        f"({share(total['close'], total['cells'])} of annotated cells)",
        f"  missed:       {total['cells'] - total['found']}  ({share(total['cells'] - total['found'], total['cells'])} of annotated cells)",
        f"  false alarms: {total['false_alarms']}  ({share(total['false_alarms'], total['objects'])} of predicted objects)"
        f"; {sum(row['false_alarms'] for row in empty)} of them in sections with an empty mask",
    ]
    if len(found_iou):
        lines += [
            "",
            "OUTLINES OF THE CELLS THAT WERE FOUND",
            f"  mean IoU {found_iou.mean():.2f}, which corresponds to a Dice of {np.mean(2 * found_iou / (1 + found_iou)):.2f} per cell",
            f"  median area: annotated {np.median(cell_area[cell_iou >= FOUND_IOU]):.0f} px, "
            f"predicted {np.median(matched_area[cell_iou >= FOUND_IOU]):.0f} px",
        ]
    if len(cell_area):
        lines.append(f"  (median area of all annotated cells: {np.median(cell_area):.0f} px)")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained U-Net on held-out sections.")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val", help="which slides to evaluate")
    parser.add_argument("--weights", default=os.path.join(tr.DATA_DIR, tr.MODEL_FILE), help="trained weights (.pth)")
    args = parser.parse_args()

    data_dir = tr.DATA_DIR
    split_path = os.path.join(data_dir, tr.SPLIT_FILE)
    for needed in (split_path, args.weights):
        if not os.path.exists(needed):
            raise SystemExit(f"{needed} not found: run unet_training_dict.py first")
    if args.split == "test":
        print("NOTE: this evaluates the TEST slides. Do it once, at the end; tune on the validation slides.")

    # the same pairing, size check and slide split as in training
    images, segs = tr.list_pairs(data_dir)
    files = tr.split_by_slide(images, segs, split_path)[args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tr.build_model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    out_dir = os.path.join(data_dir, "evaluation", f"{args.split}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.join(out_dir, "predicted_masks"))
    rng = np.random.default_rng(0)
    examples = {category: [] for category in CATEGORIES}
    rows, all_iou, all_cell_area, all_matched_area = [], [], [], []

    print(f"evaluating {len(files)} sections of the {args.split} slides on {device}...")
    with PdfPages(os.path.join(out_dir, "sections_overview.pdf")) as overview:
        for number, item in enumerate(files, start=1):
            name = tr.section_key(item["image"])
            image = tr.load_and_normalize_section(item["image"])  # (1, Y, X), as the network saw it in training
            truth = tr.load_mask(item["label"])[0] > 0.5           # (Y, X) bool
            predicted = predict_section(model, image, device)     # (Y, X) bool
            image = image[0]

            match = match_objects(truth, predicted)
            found = int((match["cell_iou"] >= FOUND_IOU).sum())
            overlap = int((truth & predicted).sum())
            true_pixels, predicted_pixels = int(truth.sum()), int(predicted.sum())
            dice = 2 * overlap / (true_pixels + predicted_pixels) if true_pixels else float("nan")
            rows.append(
                {
                    "section": name,
                    "slide": tr.slide_of(item["image"]),
                    "cells": match["n_true"],
                    "objects": match["n_pred"],
                    "found": found,
                    "close": int((match["cell_iou"] >= CLOSE_IOU).sum()),
                    "missed": match["n_true"] - found,
                    "false_alarms": int((~match["object_matched"]).sum()),
                    "dice": dice,
                    "true_pixels": true_pixels,
                    "predicted_pixels": predicted_pixels,
                    "overlap_pixels": overlap,
                }
            )
            all_iou.append(match["cell_iou"])
            all_cell_area.append(match["cell_area"])
            all_matched_area.append(match["matched_object_area"])

            for category, new in collect_examples(name, image, truth, predicted, match, rng).items():
                examples[category] += new
            add_overview_page(overview, name, image, match, dice)
            Image.fromarray(predicted.astype(np.uint8) * 255).save(os.path.join(out_dir, "predicted_masks", f"{name}.png"))
            print(
                f"  {number}/{len(files)} {name}: {match['n_true']} cells, {found} found, "
                f"{rows[-1]['missed']} missed, {rows[-1]['false_alarms']} false alarms"
            )

    with open(os.path.join(out_dir, "sections.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_gallery(examples, os.path.join(out_dir, "gallery.pdf"), rng)

    lines = summarise(
        rows, np.concatenate(all_iou), np.concatenate(all_cell_area), np.concatenate(all_matched_area), args.split, args.weights
    )
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nresults saved in {out_dir}")


if __name__ == "__main__":
    main()
