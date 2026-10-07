"""Evaluate the trained U-Net on held-out sections.

    python3 unet_evaluation_dict.py                # validation slides: use this while developing
    python3 unet_evaluation_dict.py --split test   # test slides: once, at the very end

Everything that has to match training (file pairing, size check, loaders, normalisation, network,
patch size, the slide split) is imported from unet_training_dict.py, which must sit next to this file.

The masks come in two kinds, and each section is scored according to its kind:
    outlines   every cell is drawn as a shape      -> pixel Dice, and cells matched by overlap (IoU)
    dots       every cell is a small fixed marker  -> a cell is found if the marker's centre lies inside
                                                      a predicted object; Dice and outlines mean nothing

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

# Predicted objects that come within this many pixels of the black surround (or of the image border)
# are discarded before scoring. The surround was cut out with a coarse mask, so that edge is artificial.
# Set to 0 to keep everything (and to reproduce the mean Dice of the training log exactly).
EDGE_MARGIN = 8

# Outlined cells: an annotated cell and a predicted object are compared by IoU = overlap / union.
FOUND_IOU = 0.1       # the cell counts as "found" from this IoU up ...
CLOSE_IOU = 0.5       # ... and as "found with a closely matching outline" from this IoU up

CROP = 128            # side of the gallery close-ups, in pixels
EXAMPLES_PER_ROW = 12
CATEGORIES = ["missed", "false alarm", "found, outline differs", "found, outline matches", "found (dot marker)"]
STRUCTURE = np.ones((3, 3), dtype=bool)  # pixels touching at a corner belong to the same object


def annotation_type(cell_areas):
    """'dots' if (nearly) all annotated objects have one and the same size, i.e. are point markers.

    Masks with fewer than three objects cannot be told apart this way and are treated as outlines.
    """
    if len(cell_areas) == 0:
        return "empty"
    _, counts = np.unique(cell_areas, return_counts=True)
    return "dots" if len(cell_areas) >= 3 and counts.max() >= 0.8 * len(cell_areas) else "outlines"


def predict_section(model, image, device):
    """Predicted cell mask (Y, X, bool) for one normalised section (1, Y, X), tiled exactly as in training."""
    with torch.no_grad():
        inputs = torch.from_numpy(image)[None].to(device)  # (1, 1, Y, X)
        logits = sliding_window_inference(inputs, tr.PATCH_SIZE, SW_BATCH_SIZE, model)
        return (torch.sigmoid(logits)[0, 0] > THRESHOLD).cpu().numpy()


def remove_edge_objects(predicted, truth, image):
    """Drop predicted objects that come within EDGE_MARGIN pixels of the black surround or the image border.

    Returns the cleaned mask and three counts: objects removed, how many of those overlapped an
    annotated cell, and how many annotated cells lie in that edge band themselves.
    """
    if EDGE_MARGIN <= 0:
        return predicted, 0, 0, 0
    tissue = (image > 0).view(np.uint8)
    interior = ndimage.minimum_filter(tissue, size=2 * EDGE_MARGIN + 1, mode="constant", cval=0).astype(bool)
    labels, _ = ndimage.label(predicted, structure=STRUCTURE)
    at_edge = np.unique(labels[predicted & ~interior])
    removed = np.isin(labels, at_edge) & predicted
    on_cells = len(np.unique(labels[removed & truth]))
    true_labels, _ = ndimage.label(truth, structure=STRUCTURE)
    cells_at_edge = len(np.unique(true_labels[truth & ~interior]))
    return predicted & ~removed, len(at_edge), on_cells, cells_at_edge


def match_objects(truth, predicted):
    """Compare annotated cells with predicted objects in one section.

    Both masks are split into connected objects. Two comparisons are made:
    - by overlap: every overlapping (cell, object) pair gets an IoU, and pairs are accepted best-first,
      each cell and each object used at most once (one blob over three cells is credited with one);
    - by centre: which predicted object, if any, contains the centre of each annotated cell.
    """
    true_labels, n_true = ndimage.label(truth, structure=STRUCTURE)
    pred_labels, n_pred = ndimage.label(predicted, structure=STRUCTURE)
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
        boxes = ndimage.find_objects(labels)
        return np.array([[(s.start + s.stop) // 2 for s in box] for box in boxes], dtype=int).reshape(-1, 2)

    cell_centres, object_centres = centres(true_labels), centres(pred_labels)
    centre_object = pred_labels[cell_centres[:, 0], cell_centres[:, 1]]   # object under each cell centre (0 = none)
    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "cell_area": true_area[1:],                     # per annotated cell
        "cell_iou": cell_iou[1:],
        "matched_object_area": np.where(cell_object[1:] > 0, pred_area[cell_object[1:]], 0),  # 0 if missed
        "cell_centre_covered": centre_object > 0,
        "object_area": pred_area[1:],                   # per predicted object
        "object_matched": object_taken[1:],
        "object_cell_centres": np.bincount(centre_object, minlength=n_pred + 1)[1:],  # cell centres inside it
        "cell_centres": cell_centres,                   # (row, column)
        "object_centres": object_centres,
    }


def collect_examples(name, image, truth, predicted, match, verdict, rng, per_category=3):
    """A few close-ups per category from one section, for the gallery."""
    found, close, false_alarm = verdict["found"], verdict["close"], verdict["false_alarm"]
    tag = " [dots]" if verdict["kind"] == "dots" else ""
    with_iou = lambda keep: [(c, f"IoU {i:.2f}") for c, i in zip(match["cell_centres"][keep], match["cell_iou"][keep])]
    candidates = {
        "missed": [(c, "") for c in match["cell_centres"][~found]],
        "false alarm": [(c, "") for c in match["object_centres"][false_alarm]],
    }
    if verdict["kind"] == "dots":
        candidates["found (dot marker)"] = [(c, "") for c in match["cell_centres"][found]]
    else:
        candidates["found, outline differs"] = with_iou(found & ~close)
        candidates["found, outline matches"] = with_iou(close)

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
                    "title": f"{name}{tag}\n{note}".strip(),
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


def add_overview_page(pdf, name, image, match, verdict, dice):
    """One page: the whole section, shrunk, with every annotated cell and predicted object marked."""
    step = max(1, max(image.shape) // 1200)
    found = verdict["found"]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image[::step, ::step], cmap="gray", vmin=0, vmax=max(np.percentile(image, 99.9), 1e-6))
    for centres, style, label in [
        (match["cell_centres"][found], {"marker": "o", "color": "lime", "s": 14}, "found"),
        (match["cell_centres"][~found], {"marker": "x", "color": "red", "s": 40}, "missed"),
        (
            match["object_centres"][verdict["false_alarm"]],
            {"marker": "s", "facecolors": "none", "edgecolors": "magenta", "s": 40},
            "false alarm",
        ),
    ]:
        ax.scatter(centres[:, 1] / step, centres[:, 0] / step, label=f"{label} ({len(centres)})", linewidths=1.2, **style)
    ax.set_xticks([])
    ax.set_yticks([])
    detail = {"dots": "dot markers", "empty": "empty mask"}.get(verdict["kind"], f"outlined cells, pixel Dice {dice:.2f}")
    ax.set_title(f"{name}  ({detail})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def share(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def summarise(rows, pooled, split, weights_path):
    """The text summary, as a list of lines. `pooled` holds per-cell and per-object arrays of the outlined sections."""
    total = lambda group, key: sum(row[key] for row in group)
    by_kind = {kind: [row for row in rows if row["annotation"] == kind] for kind in ("outlines", "dots", "empty")}
    with_cells = by_kind["outlines"] + by_kind["dots"]

    lines = [
        f"split: {split}   weights: {weights_path}   threshold: {THRESHOLD}   edge margin: {EDGE_MARGIN} px",
        f"sections: {len(rows)}  ({len(by_kind['outlines'])} with outlined cells, {len(by_kind['dots'])} with dot markers, "
        f"{len(by_kind['empty'])} with an empty mask)",
        f"mean Dice per section over all sections with cells: "
        f"{np.mean([row['dice'] for row in with_cells]) if with_cells else float('nan'):.4f}"
        "   (the training-log number when the edge margin is 0; it mixes outlines and dots)",
    ]
    if EDGE_MARGIN > 0:
        lines.append(
            f"edge filter: {total(rows, 'edge_objects_removed')} predicted objects within {EDGE_MARGIN} px of the black "
            f"surround were discarded; {total(rows, 'edge_objects_on_cells')} of them overlapped an annotated cell; "
            f"{total(rows, 'cells_at_edge')} annotated cells lie in that band"
        )

    group = by_kind["outlines"]
    if group:
        cells, objects, found = total(group, "cells"), total(group, "objects"), total(group, "found")
        overlap, both_sizes = total(group, "overlap_pixels"), total(group, "true_pixels") + total(group, "predicted_pixels")
        found_iou = pooled["cell_iou"][pooled["cell_iou"] >= FOUND_IOU]
        lines += [
            "",
            f"SECTIONS WITH OUTLINED CELLS ({len(group)})",
            f"  pixels: mean Dice per section {np.mean([row['dice'] for row in group]):.4f}, "
            f"Dice over all their pixels pooled {2 * overlap / max(both_sizes, 1):.4f}",
            f"  annotated cells: {cells}    predicted objects: {objects}",
            f"  found (IoU >= {FOUND_IOU}):  {found}  ({share(found, cells)} of annotated cells)",
            f"    of these, outline closely matching (IoU >= {CLOSE_IOU}): {total(group, 'close')}  "
            f"({share(total(group, 'close'), cells)} of annotated cells)",
            f"  missed:        {cells - found}  ({share(cells - found, cells)} of annotated cells)",
            f"  false alarms:  {total(group, 'false_alarms')}  ({share(total(group, 'false_alarms'), objects)} of predicted objects)",
        ]
        if len(found_iou):
            lines.append(
                f"  outlines of the found cells: mean IoU {found_iou.mean():.2f} (a Dice of "
                f"{np.mean(2 * found_iou / (1 + found_iou)):.2f} per cell); median area annotated "
                f"{np.median(pooled['cell_area'][pooled['cell_iou'] >= FOUND_IOU]):.0f} px, predicted "
                f"{np.median(pooled['matched_object_area'][pooled['cell_iou'] >= FOUND_IOU]):.0f} px"
            )
        if len(pooled["false_alarm_area"]):
            lines.append(f"  median area of a false alarm: {np.median(pooled['false_alarm_area']):.0f} px")

    group = by_kind["dots"]
    if group:
        cells, objects, found = total(group, "cells"), total(group, "objects"), total(group, "found")
        lines += [
            "",
            f"SECTIONS WITH DOT MARKERS ({len(group)})   (found = the marker's centre lies inside a predicted object)",
            f"  annotated cells: {cells}    predicted objects: {objects}",
            f"  found:         {found}  ({share(found, cells)} of annotated cells)",
            f"  missed:        {cells - found}  ({share(cells - found, cells)} of annotated cells)",
            f"  false alarms:  {total(group, 'false_alarms')}  ({share(total(group, 'false_alarms'), objects)} of predicted "
            "objects; these contain no marker)",
            f"  merged:        {total(group, 'merged_objects')} predicted objects contain more than one marker "
            f"({total(group, 'cells_in_merged_objects')} markers in all)",
        ]

    group = by_kind["empty"]
    if group:
        lines += ["", f"SECTIONS WITH AN EMPTY MASK ({len(group)}): {total(group, 'objects')} predicted objects, all false alarms"]
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
    pooled = {"cell_iou": [], "cell_area": [], "matched_object_area": [], "false_alarm_area": []}  # outlined sections
    rows = []

    print(f"evaluating {len(files)} sections of the {args.split} slides on {device}...")
    with PdfPages(os.path.join(out_dir, "sections_overview.pdf")) as overview:
        for number, item in enumerate(files, start=1):
            name = tr.section_key(item["image"])
            image = tr.load_and_normalize_section(item["image"])  # (1, Y, X), as the network saw it in training
            truth = tr.load_mask(item["label"])[0] > 0.5           # (Y, X) bool
            predicted = predict_section(model, image, device)     # (Y, X) bool
            image = image[0]
            predicted, edge_removed, edge_on_cells, cells_at_edge = remove_edge_objects(predicted, truth, image)

            match = match_objects(truth, predicted)
            kind = annotation_type(match["cell_area"])
            if kind == "dots":  # point markers: judge by position only
                found = match["cell_centre_covered"]
                close = np.zeros(match["n_true"], dtype=bool)
                false_alarm = match["object_cell_centres"] == 0
            else:               # outlined cells (or an empty mask): judge by overlap
                found = match["cell_iou"] >= FOUND_IOU
                close = match["cell_iou"] >= CLOSE_IOU
                false_alarm = ~match["object_matched"]
            verdict = {"kind": kind, "found": found, "close": close, "false_alarm": false_alarm}

            overlap = int((truth & predicted).sum())
            true_pixels, predicted_pixels = int(truth.sum()), int(predicted.sum())
            dice = 2 * overlap / (true_pixels + predicted_pixels) if true_pixels else float("nan")
            merged = match["object_cell_centres"] > 1
            rows.append(
                {
                    "section": name,
                    "slide": tr.slide_of(item["image"]),
                    "annotation": kind,
                    "cells": match["n_true"],
                    "objects": match["n_pred"],
                    "found": int(found.sum()),
                    "close": int(close.sum()),
                    "missed": int((~found).sum()),
                    "false_alarms": int(false_alarm.sum()),
                    "dice": dice,
                    "true_pixels": true_pixels,
                    "predicted_pixels": predicted_pixels,
                    "overlap_pixels": overlap,
                    "merged_objects": int(merged.sum()),
                    "cells_in_merged_objects": int(match["object_cell_centres"][merged].sum()),
                    "edge_objects_removed": edge_removed,
                    "edge_objects_on_cells": edge_on_cells,
                    "cells_at_edge": cells_at_edge,
                }
            )
            if kind == "outlines":
                for key in ("cell_iou", "cell_area", "matched_object_area"):
                    pooled[key].append(match[key])
                pooled["false_alarm_area"].append(match["object_area"][false_alarm])

            for category, new in collect_examples(name, image, truth, predicted, match, verdict, rng).items():
                examples[category] += new
            add_overview_page(overview, name, image, match, verdict, dice)
            Image.fromarray(predicted.astype(np.uint8) * 255).save(os.path.join(out_dir, "predicted_masks", f"{name}.png"))
            print(
                f"  {number}/{len(files)} {name} [{kind}]: {match['n_true']} cells, {rows[-1]['found']} found, "
                f"{rows[-1]['missed']} missed, {rows[-1]['false_alarms']} false alarms, {edge_removed} edge objects discarded"
            )

    with open(os.path.join(out_dir, "sections.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_gallery(examples, os.path.join(out_dir, "gallery.pdf"), rng)

    pooled = {key: np.concatenate(parts) if parts else np.array([]) for key, parts in pooled.items()}
    lines = summarise(rows, pooled, args.split, args.weights)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nresults saved in {out_dir}")


if __name__ == "__main__":
    main()
