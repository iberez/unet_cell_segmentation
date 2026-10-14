"""Evaluate trained U-Nets on sections they were not trained on.

    python3 unet_evaluation_dict.py --cv           # every fold model on its own validation slides (slide split)
    python3 unet_evaluation_dict.py --split val    # one model on its own validation set
    python3 unet_evaluation_dict.py --split test   # one model on its own test set: once, at the very end
    python3 unet_evaluation_dict.py --split train  # one model on its own training set (slide split: writes hard negatives)

--weights picks the model for --split. By default that is models/final.pth when SPLIT_MODE in
unet_training_dict.py is "slide", and models/section.pth when it is "section". Which sections a model
trained, validated and is tested on is read from the .json record saved next to its weights; weights
without one were trained on the original slide split in split_by_slide.json.

Everything that has to match training (file pairing, size check, loaders, normalisation, network,
patch size, the slides) is imported from unet_training_dict.py, which must sit next to this file.

The masks come in two kinds, and each section is scored according to its kind:
    outlines   every cell is drawn as a shape      -> pixel Dice, and cells matched by overlap (IoU)
    dots       every cell is a small fixed marker  -> a cell is found if a predicted object touches the
                                                      marker; Dice and outlines mean nothing

Results go to <data directory>/evaluation/<cv or split>_<date>_<time>/ :
    summary.txt            the numbers printed at the end
    sections.csv           the same counts, one row per section
    false_alarms.csv       every false alarm: where it is, how big, how far from the nearest annotated cell
    gallery.pdf            close-ups of missed cells, false alarms, and found cells
    sections_overview.pdf  one page per section showing where the hits, misses and false alarms are
    predicted_masks/       the predicted mask of every section as a PNG (0 = background, 255 = cell)

Run on the training slides (--split train), it also writes hard_negatives.csv into the data directory:
the false alarms of the current model on the sections it was trained on. unet_training_dict.py reads
that file and draws extra background patches at those places.
"""

import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

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

# Predicted objects smaller than this many pixels are discarded before scoring (0 = keep all). The summary
# prints a table of what larger values would do, so choose it from a validation run. 20 px was chosen from
# the validation run of 8 October: it removed half of the false alarms and 5 of 1210 found cells.
MIN_OBJECT_AREA = 20

# A false alarm closer than this to an annotated cell (centre to centre) is counted as "near a cell".
NEAR_CELL_DISTANCE = 40

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


def remove_small_objects(predicted, truth):
    """Drop predicted objects smaller than MIN_OBJECT_AREA pixels.

    Returns the cleaned mask, the number of objects removed, and how many of those overlapped an annotated cell.
    """
    if MIN_OBJECT_AREA <= 0:
        return predicted, 0, 0
    labels, _ = ndimage.label(predicted, structure=STRUCTURE)
    small = np.flatnonzero(np.bincount(labels.ravel())[1:] < MIN_OBJECT_AREA) + 1
    removed = np.isin(labels, small) & predicted
    return predicted & ~removed, len(small), len(np.unique(labels[removed & truth]))


def match_objects(truth, predicted):
    """Compare annotated cells with predicted objects in one section.

    Both masks are split into connected objects. Two comparisons are made:
    - by IoU: every overlapping (cell, object) pair gets an IoU, and pairs are accepted best-first,
      each cell and each object used at most once (one blob over three cells is credited with one);
    - by touch: whether a cell shares any pixel with a predicted object, and how many cells each
      object touches (this is what counts for dot markers, where shapes mean nothing).
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

    cell_touched = np.zeros(n_true + 1, dtype=bool)
    cell_touched[cell] = True
    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "cell_area": true_area[1:],                     # per annotated cell
        "cell_iou": cell_iou[1:],
        "matched_object_area": np.where(cell_object[1:] > 0, pred_area[cell_object[1:]], 0),  # 0 if missed
        "cell_touched": cell_touched[1:],               # shares at least one pixel with a predicted object
        "object_area": pred_area[1:],                   # per predicted object
        "object_matched": object_taken[1:],
        "object_cells_touched": np.bincount(obj, minlength=n_pred + 1)[1:],  # annotated cells it overlaps
        "cell_centres": centres(true_labels),           # (row, column)
        "object_centres": centres(pred_labels),
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


def add_overview_page(pdf, name, image, match, verdict, dice, model_name):
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
    ax.set_title(f"{name} [{model_name}]  ({detail})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def share(part, whole):
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def per_group_lines(rows, key, title):
    """A small table with one line per model (key="model") or per slide (key="slide")."""
    lines = ["", title, f"  {key:<7} {'sections':>15}  {'cells':>6}  {'found':>7}  {'false alarms':>12}  {'Dice':>5}"]
    for value in sorted({row[key] for row in rows}):
        group = [row for row in rows if row[key] == value]
        cells, found = sum(row["cells"] for row in group), sum(row["found"] for row in group)
        n_dots = sum(row["annotation"] == "dots" for row in group)
        dice = [row["dice"] for row in group if row["annotation"] == "outlines"]
        sections = f"{len(group)}" + (f" ({n_dots} dots)" if n_dots else "")
        lines.append(
            f"  {value:<7} {sections:>15}  {cells:>6}  {share(found, cells):>7}  "
            f"{sum(row['false_alarms'] for row in group):>12}  " + (f"{np.mean(dice):>5.2f}" if dice else f"{'-':>5}")
        )
    return lines


def summarise(rows, pooled, split, weights_path):
    """The text summary, as a list of lines. `pooled` holds per-cell and per-object arrays of the outlined sections."""
    total = lambda group, key: sum(row[key] for row in group)
    by_kind = {kind: [row for row in rows if row["annotation"] == kind] for kind in ("outlines", "dots", "empty")}
    with_cells = by_kind["outlines"] + by_kind["dots"]

    lines = [
        f"evaluated: {split}   weights: {weights_path}   threshold: {THRESHOLD}   edge margin: {EDGE_MARGIN} px"
        f"   minimum object area: {MIN_OBJECT_AREA} px",
        f"sections: {len(rows)}  ({len(by_kind['outlines'])} with outlined cells, {len(by_kind['dots'])} with dot markers, "
        f"{len(by_kind['empty'])} with an empty mask)",
        f"mean Dice per section over all sections with cells: "
        f"{np.mean([row['dice'] for row in with_cells]) if with_cells else float('nan'):.4f}"
        "   (mixes outlines and dots; equals the training-log number only with both filters off)",
    ]
    if EDGE_MARGIN > 0:
        lines.append(
            f"edge filter: {total(rows, 'edge_objects_removed')} predicted objects within {EDGE_MARGIN} px of the black "
            f"surround were discarded; {total(rows, 'edge_objects_on_cells')} of them overlapped an annotated cell; "
            f"{total(rows, 'cells_at_edge')} annotated cells lie in that band"
        )
    if MIN_OBJECT_AREA > 0:
        lines.append(
            f"size filter: {total(rows, 'small_objects_removed')} predicted objects smaller than {MIN_OBJECT_AREA} px "
            f"were discarded; {total(rows, 'small_objects_on_cells')} of them overlapped an annotated cell"
        )

    group = by_kind["outlines"]
    if group:
        cells, objects, found = total(group, "cells"), total(group, "objects"), total(group, "found")
        false_alarms_n = total(group, "false_alarms")
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
            near = int((pooled["false_alarm_distance"] < NEAR_CELL_DISTANCE).sum())
            lines.append(
                f"  median area of a false alarm: {np.median(pooled['false_alarm_area']):.0f} px; "
                f"{near} of them lie within {NEAR_CELL_DISTANCE} px of an annotated cell, {false_alarms_n - near} farther away"
            )
        # what a (larger) minimum object area would do, judged from the sizes of today's objects
        found_object_area = pooled["matched_object_area"][pooled["cell_iou"] >= FOUND_IOU]
        false_alarms = total(group, "false_alarms")
        lines += [
            f"  what a larger minimum object area would do (objects under {MIN_OBJECT_AREA} px are already gone):",
            "      minimum area    false alarms left    found cells kept",
        ]
        for area in (0, 10, 20, 30, 40, 50, 75, 100):
            left, kept = int((pooled["false_alarm_area"] >= area).sum()), int((found_object_area >= area).sum())
            lines.append(
                f"      {area:>8} px    {left:>6} ({share(left, false_alarms):>6})    {kept:>6} ({share(kept, found):>6})"
            )

    group = by_kind["dots"]
    if group:
        cells, objects, found = total(group, "cells"), total(group, "objects"), total(group, "found")
        lines += [
            "",
            f"SECTIONS WITH DOT MARKERS ({len(group)})   (found = a predicted object touches the marker)",
            f"  annotated cells: {cells}    predicted objects: {objects}",
            f"  found:         {found}  ({share(found, cells)} of annotated cells)",
            f"  missed:        {cells - found}  ({share(cells - found, cells)} of annotated cells)",
            f"  false alarms:  {total(group, 'false_alarms')}  ({share(total(group, 'false_alarms'), objects)} of predicted "
            "objects; these touch no marker)",
            f"  merged:        {total(group, 'merged_objects')} predicted objects touch more than one marker "
            f"({total(group, 'cells_in_merged_objects')} markers in all)",
        ]

    group = by_kind["empty"]
    if group:
        lines += ["", f"SECTIONS WITH AN EMPTY MASK ({len(group)}): {total(group, 'objects')} predicted objects, all false alarms"]

    # how the models and the slides compare ("found" counts outlined cells by overlap and dot markers by touch)
    models = sorted({row["model"] for row in rows})
    if len(models) > 1:
        lines += per_group_lines(rows, "model", "PER MODEL (each on its own validation slides; Dice: mean over outlined sections)")
        model_dice = [
            np.mean([row["dice"] for row in rows if row["model"] == m and row["annotation"] == "outlines"])
            for m in models
            if any(row["model"] == m and row["annotation"] == "outlines" for row in rows)
        ]
        if len(model_dice) > 1:
            lines.append(
                f"  across models: Dice {np.mean(model_dice):.3f} +/- {np.std(model_dice, ddof=1):.3f} (mean +/- SD)"
            )
    lines += per_group_lines(rows, "slide", "PER SLIDE (anterior to posterior; Dice: mean over outlined sections)")
    return lines


def sets_of_model(weights_path, images, segs, data_dir):
    """The training, validation and test sections of a model, from the record saved next to its weights.

    Returns ({"train": [...], "val": [...], "test": [...]}, split mode). Records of slide-split runs list
    slides, records of section-split runs list sections; weights without a record were trained on the
    original slide split in SPLIT_FILE.
    """
    info = {}
    info_path = tr.model_info_path(weights_path)
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)

    if "train_sections" in info:
        sets = {name: tr.files_for_sections(images, segs, info[f"{name}_sections"]) for name in ("train", "val", "test")}
        for name, files in sets.items():
            if len(files) != len(info[f"{name}_sections"]):
                print(f"WARNING: {len(info[name + '_sections']) - len(files)} {name} sections of this model were not found")
        return sets, "section"

    if "train_slides" in info:
        slides = {name: info[f"{name}_slides"] for name in ("train", "val", "test")}
    else:
        assignment = tr.slide_assignment(images, os.path.join(data_dir, tr.SPLIT_FILE))
        slides = {name: sorted(s for s, part in assignment.items() if part == name) for name in ("train", "val", "test")}
    return {name: tr.files_for_slides(images, segs, slides[name]) for name in slides}, "slide"


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained U-Nets on sections they were not trained on.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--cv", action="store_true", help="every fold model on its own validation slides")
    which.add_argument("--split", choices=["val", "test", "train"], help="one model (--weights) on these slides of its own")
    parser.add_argument(
        "--weights",
        help=f"weights for --split (default: {tr.MODEL_FOLDER}/final.pth, or {tr.MODEL_FOLDER}/section.pth when SPLIT_MODE is section)",
    )
    args = parser.parse_args()
    if args.cv and args.weights:
        parser.error("--weights goes with --split; --cv always uses the fold models")

    data_dir = tr.DATA_DIR
    # the same pairing and size check as in training
    images, segs = tr.list_pairs(data_dir)

    # (model name, weights, sections to evaluate with those weights)
    jobs, mode = [], "slide"
    if args.cv:
        _, folds = tr.load_folds(images, data_dir)
        for k in range(len(folds)):
            weights = tr.model_path(data_dir, f"fold{k}")
            if not os.path.exists(weights):
                raise SystemExit(f"{weights} not found: run python3 unet_training_dict.py --fold {k} first")
            jobs.append((f"fold{k}", weights, sets_of_model(weights, images, segs, data_dir)[0]["val"]))
        label = "cross-validation, each fold model on its own validation slides"
        weights_label = os.path.join(data_dir, tr.MODEL_FOLDER, f"fold0 ... fold{len(folds) - 1}")
    else:
        default_run = "section" if tr.SPLIT_MODE == "section" else "final"
        weights = args.weights or tr.model_path(data_dir, default_run)
        if not os.path.exists(weights):
            how = "python3 unet_training_dict.py" + ("" if default_run == "section" else " --final")
            raise SystemExit(f"{weights} not found: train it first ({how})")
        sets, mode = sets_of_model(weights, images, segs, data_dir)
        files = sets[args.split]
        if not files:
            raise SystemExit(f"{weights} has no {args.split} sections (the final model trains on every non-test slide)")
        if args.split == "test":
            print("NOTE: this evaluates the TEST set. Do it once, at the end.")
        model_name = os.path.splitext(os.path.basename(weights))[0]
        jobs.append((model_name, weights, files))
        label, weights_label = f"{args.split} set of the {mode} split", weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_label = "cv" if args.cv else f"{args.split}_{model_name}"
    out_dir = os.path.join(data_dir, "evaluation", f"{run_label}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.join(out_dir, "predicted_masks"))
    rng = np.random.default_rng(0)
    examples = {category: [] for category in CATEGORIES}
    pooled = {"cell_iou": [], "cell_area": [], "matched_object_area": [], "false_alarm_area": [], "false_alarm_distance": []}
    rows, false_alarm_rows = [], []  # `pooled` covers the outlined sections only

    n_sections = sum(len(files) for _, _, files in jobs)
    print(f"evaluating {n_sections} sections on {device}...")
    number = 0
    with PdfPages(os.path.join(out_dir, "sections_overview.pdf")) as overview:
        for model_name, weights, files in jobs:
            model = tr.build_model().to(device)
            model.load_state_dict(torch.load(weights, map_location=device))
            model.eval()
            for item in files:
                number += 1
                name = tr.section_key(item["image"])
                image = tr.load_and_normalize_section(item["image"])  # (1, Y, X), as the network saw it in training
                truth = tr.load_mask(item["label"])[0] > 0.5           # (Y, X) bool
                predicted = predict_section(model, image, device)     # (Y, X) bool
                image = image[0]
                predicted, edge_removed, edge_on_cells, cells_at_edge = remove_edge_objects(predicted, truth, image)
                predicted, small_removed, small_on_cells = remove_small_objects(predicted, truth)

                match = match_objects(truth, predicted)
                kind = annotation_type(match["cell_area"])
                if kind == "dots":  # point markers: judge by position only
                    found = match["cell_touched"]
                    close = np.zeros(match["n_true"], dtype=bool)
                    false_alarm = match["object_cells_touched"] == 0
                else:               # outlined cells (or an empty mask): judge by overlap
                    found = match["cell_iou"] >= FOUND_IOU
                    close = match["cell_iou"] >= CLOSE_IOU
                    false_alarm = ~match["object_matched"]
                verdict = {"kind": kind, "found": found, "close": close, "false_alarm": false_alarm}

                overlap = int((truth & predicted).sum())
                true_pixels, predicted_pixels = int(truth.sum()), int(predicted.sum())
                dice = 2 * overlap / (true_pixels + predicted_pixels) if true_pixels else float("nan")
                merged = match["object_cells_touched"] > 1
                rows.append(
                    {
                        "section": name,
                        "slide": tr.slide_of(item["image"]),
                        "model": model_name,
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
                        "cells_in_merged_objects": int(match["object_cells_touched"][merged].sum()),
                        "edge_objects_removed": edge_removed,
                        "edge_objects_on_cells": edge_on_cells,
                        "cells_at_edge": cells_at_edge,
                        "small_objects_removed": small_removed,
                        "small_objects_on_cells": small_on_cells,
                    }
                )
                # every false alarm: position, size, and distance to the nearest annotated cell (centre to centre)
                alarm_centres, alarm_areas = match["object_centres"][false_alarm], match["object_area"][false_alarm]
                if match["n_true"] and len(alarm_centres):
                    alarm_distances = cKDTree(match["cell_centres"]).query(alarm_centres)[0]
                else:
                    alarm_distances = np.full(len(alarm_centres), np.inf)
                for (row, col), area, distance in zip(alarm_centres, alarm_areas, alarm_distances):
                    false_alarm_rows.append(
                        {
                            "section": name,
                            "slide": tr.slide_of(item["image"]),
                            "model": model_name,
                            "annotation": kind,
                            "row": int(row),
                            "col": int(col),
                            "area_px": int(area),
                            "distance_to_cell_px": round(float(distance), 1),
                        }
                    )

                if kind == "outlines":
                    for key in ("cell_iou", "cell_area", "matched_object_area"):
                        pooled[key].append(match[key])
                    pooled["false_alarm_area"].append(alarm_areas)
                    pooled["false_alarm_distance"].append(alarm_distances)

                for category, new in collect_examples(name, image, truth, predicted, match, verdict, rng).items():
                    examples[category] += new
                add_overview_page(overview, name, image, match, verdict, dice, model_name)
                Image.fromarray(predicted.astype(np.uint8) * 255).save(os.path.join(out_dir, "predicted_masks", f"{name}.png"))
                print(
                    f"  {number}/{n_sections} {name} [{kind}, {model_name}]: {match['n_true']} cells, {rows[-1]['found']} found, "
                    f"{rows[-1]['missed']} missed, {rows[-1]['false_alarms']} false alarms, {edge_removed} edge objects discarded"
                )

    with open(os.path.join(out_dir, "sections.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_gallery(examples, os.path.join(out_dir, "gallery.pdf"), rng)

    alarm_files = [os.path.join(out_dir, "false_alarms.csv")]
    # false alarms on a slide-split model's own training slides are the hard negatives for the next training
    # run; a section-split model trained on parts of every slide, so its false alarms are kept out of that file
    write_hard_negatives = args.split == "train" and mode == "slide"
    if write_hard_negatives:
        alarm_files.append(os.path.join(data_dir, tr.HARD_NEGATIVE_FILE))
    for path in alarm_files:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["section", "slide", "model", "annotation", "row", "col", "area_px", "distance_to_cell_px"]
            )
            writer.writeheader()
            writer.writerows(false_alarm_rows)

    pooled = {key: np.concatenate(parts) if parts else np.array([]) for key, parts in pooled.items()}
    lines = summarise(rows, pooled, label, weights_label)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nresults saved in {out_dir}")
    if write_hard_negatives:
        print(
            f"{len(false_alarm_rows)} false alarms on the training slides written to {alarm_files[-1]}; "
            "unet_training_dict.py will use them as hard negatives"
        )
    elif args.split == "train":
        print(f"{tr.HARD_NEGATIVE_FILE} left unchanged: hard negatives are only taken from slide-split models")


if __name__ == "__main__":
    main()
