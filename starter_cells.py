"""Count starter cells per section: cells that are labelled in both the AAV channel and the GFP channel.

    python3 starter_cells.py --masks ./124_02_test/124_02_single_section_seg_masks/ \\
        --pred-aav ./124_02_test/evaluation/all_aav_section_<time>/ \\
        --pred-gfp ./124_02_test/evaluation/all_section_<time>/

Ground truth: each section's .mat file holds an AAV mask (BW_I1) and a GFP mask (BW_I2). Each mask is split into
cells (connected pixels, diagonals included). Where an AAV cell and a GFP cell overlap, the pair is a candidate
starter cell. It counts when the overlapping pixels make up at least MIN_OVERLAP of the smaller of the two cells
(OVERLAP_MEASURE; --min-overlap and --measure change both for one run). Each cell takes part in at most one starter
cell, best overlap first, so one blob lying over two cells of the other channel counts once. The starter mask is
the overlap (the AND) of the pairs that count.

Predicted: the same, with the masks that unet_evaluation_dict.py --all saved for the AAV model and for the GFP
model: the predicted_masks folder of each run, after its edge and size filters. --pred-aav and --pred-gfp take
either the results folder of a run or its predicted_masks folder. Without them, only the ground truth is counted.

Written next to the masks folder (or into --out):
    starter_cells_gt_mask/<section>.mat      ground-truth starter cells: one logical (boolean) mask, BW_starter
    starter_cells_pred_mask/<section>.mat    predicted starter cells, the same way
    <brain>_starter_cells.xlsx               one row per section: ground-truth and predicted starter cells, with
                                             totals; a sheet with the totals at other overlap thresholds; settings
    <brain>_starter_cells_per_section.pdf    both counts for every section, slide by slide

Needs numpy, scipy, Pillow, matplotlib and openpyxl (not torch or MONAI), so it also runs without a GPU.
"""

import argparse
import csv
import os
import re
from datetime import datetime
from glob import glob

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat, savemat

import matplotlib

matplotlib.use("Agg")  # no display on the server: draw straight to file
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# embed fonts as TrueType so the text in the PDF stays editable and renders the same everywhere
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

Image.MAX_IMAGE_PIXELS = None  # whole sections are large, and these are your own files

# --- masks ---
AAV_KEY = "BW_I1"   # AAV channel mask in the .mat files (raw channel 1)
GFP_KEY = "BW_I2"   # GFP channel mask in the .mat files (raw channel 0)

# --- what counts as a starter cell ---
MIN_OVERLAP = 0.5           # the overlapping pixels must make up at least this share of ...
OVERLAP_MEASURE = "smaller"  # ... the cell(s) named here; one of the keys of OVERLAP_MEASURES
OTHER_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # totals at these go to a second sheet

# --- output ---
MASK_VARIABLE = "BW_starter"  # name of the mask inside each saved .mat file
GT_FOLDER = "starter_cells_gt_mask"
PRED_FOLDER = "starter_cells_pred_mask"

STRUCTURE = np.ones((3, 3), dtype=bool)  # pixels touching at a corner belong to the same cell

# overlap measure -> (what the share is taken of, as text; share of overlapping pixels from overlap, AAV and GFP areas)
OVERLAP_MEASURES = {
    "smaller": ("the smaller of the two cells", lambda overlap, aav, gfp: overlap / np.minimum(aav, gfp)),
    "larger": ("the larger of the two cells, so of each cell", lambda overlap, aav, gfp: overlap / np.maximum(aav, gfp)),
    "iou": ("the two cells together (intersection over union)", lambda overlap, aav, gfp: overlap / (aav + gfp - overlap)),
    "aav": ("the AAV cell", lambda overlap, aav, gfp: overlap / aav),
    "gfp": ("the GFP cell", lambda overlap, aav, gfp: overlap / gfp),
}

# chart colours (validated as a two-series categorical palette) and ink
GT_COLOR, PRED_COLOR = "#2a78d6", "#eb6834"
INK, SECONDARY_INK, MUTED_INK = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, AXIS_COLOR = "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------------------------------------------------- names

SECTION_PATTERN = re.compile(r"S(\d+)\.tif_section_(\d+)")


def section_of_mask(path):
    """'S001.tif_section_2_f' for BW_I1_I2_S001.tif_section_2_f.mat or ..._rev.mat (as in unet_training_dict.py)."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.removeprefix("BW_I1_I2_").removesuffix("_rev")


def section_of_prediction(path):
    """'S001.tif_section_2_f' for S001.tif_section_2_f.png, also when a brain name comes first (124_02_S001...)."""
    name = os.path.splitext(os.path.basename(path))[0]
    found = re.search(r"S\d+\.tif_section_.*$", name)
    return found.group(0) if found else name


def order_key(section):
    """Slide by slide, then section by section, in number order (section 10 after section 9)."""
    found = SECTION_PATTERN.search(section)
    return (int(found.group(1)), int(found.group(2)), section) if found else (float("inf"), float("inf"), section)


def slide_and_number(section):
    """('S001', 2) for 'S001.tif_section_2_f'; ('', None) if the name doesn't follow that pattern."""
    found = SECTION_PATTERN.search(section)
    return (f"S{found.group(1)}", int(found.group(2))) if found else ("", None)


def brain_of(mask_folder):
    """'124_02' for .../124_02_single_section_seg_masks."""
    folder = os.path.basename(os.path.abspath(mask_folder))
    return folder.split("_seg_masks")[0].removesuffix("_single_section") or folder


# ----------------------------------------------------------------------------------------------------- starter cells

def crop_to_common(first, second):
    """Both masks cut to their common top-left region, if their sizes differ (they normally don't)."""
    rows, cols = min(first.shape[0], second.shape[0]), min(first.shape[1], second.shape[1])
    return first[:rows, :cols], second[:rows, :cols]


def mask_kind(areas):
    """'outlines', 'dots' or 'empty', by the rule unet_evaluation_dict.py uses: dots when at least 80% of
    at least three cells have one and the same area (stamped markers rather than drawn outlines)."""
    if len(areas) == 0:
        return "empty"
    _, counts = np.unique(areas, return_counts=True)
    return "dots" if len(areas) >= 3 and counts.max() >= 0.8 * len(areas) else "outlines"


def find_starter_cells(aav, gfp, min_overlap, measure):
    """Starter cells in one section, from its AAV and GFP masks (2D bool arrays of the same shape).

    Returns (starter mask, overlap shares of all one-to-one pairs, AAV cell areas, GFP cell areas). The shares
    of all one-to-one pairs give the count at any other threshold t: the pairs with share >= t. That holds
    because pairs are taken best first, so every pair at or above t is decided before any pair below it.
    """
    aav_labels, n_aav = ndimage.label(aav, structure=STRUCTURE)
    gfp_labels, n_gfp = ndimage.label(gfp, structure=STRUCTURE)
    aav_area = np.bincount(aav_labels.ravel(), minlength=n_aav + 1)
    gfp_area = np.bincount(gfp_labels.ravel(), minlength=n_gfp + 1)

    # every overlapping (AAV cell, GFP cell) pair and the number of pixels the two share
    both = np.flatnonzero(aav & gfp)
    codes = aav_labels.ravel()[both].astype(np.int64) * (n_gfp + 1) + gfp_labels.ravel()[both]
    pairs, overlap = np.unique(codes, return_counts=True)
    aav_cell, gfp_cell = pairs // (n_gfp + 1), pairs % (n_gfp + 1)
    share = OVERLAP_MEASURES[measure][1](overlap, aav_area[aav_cell], gfp_area[gfp_cell])

    # one-to-one: best share first (then most shared pixels); a cell already in a pair is not used again
    aav_used, gfp_used = np.zeros(n_aav + 1, dtype=bool), np.zeros(n_gfp + 1, dtype=bool)
    kept = []
    for k in np.lexsort((-overlap, -share)):
        if not aav_used[aav_cell[k]] and not gfp_used[gfp_cell[k]]:
            aav_used[aav_cell[k]] = gfp_used[gfp_cell[k]] = True
            kept.append(k)
    kept = np.asarray(kept, dtype=np.int64)
    kept_share = share[kept]

    # the starter mask: the shared pixels of the pairs that reach min_overlap
    starter = np.zeros(aav.shape, dtype=bool)
    starter.reshape(-1)[both[np.isin(codes, pairs[kept[kept_share >= min_overlap]])]] = True
    return starter, kept_share, aav_area[1:], gfp_area[1:]


def save_mask(mask, path):
    """Save a starter mask as a .mat file holding one boolean array, MASK_VARIABLE (a logical array in MATLAB).

    scipy's loadmat reads it back as uint8 0/1; use loadmat(path)[MASK_VARIABLE] > 0 for a boolean array again.
    """
    savemat(path, {MASK_VARIABLE: np.ascontiguousarray(mask, dtype=bool)}, do_compression=True)


# ------------------------------------------------------------------------------------------------------- predictions

def prediction_folder(path):
    """The folder of predicted-mask PNGs: path itself, or its predicted_masks subfolder (a results folder)."""
    if not os.path.isdir(path):
        raise SystemExit(f"{path} is not a folder")
    inner = os.path.join(path, "predicted_masks")
    return inner if os.path.isdir(inner) else path


def model_of_run(path, expected_key, option):
    """What the evaluation run in path says about its model, from its summary.txt; stops if it names the wrong mask.

    Runs evaluated before the channel was recorded have no channel line; they are taken as given.
    """
    summary = os.path.join(os.path.dirname(os.path.abspath(prediction_folder(path))), "summary.txt")
    if not os.path.exists(summary):
        return "no summary.txt found"
    with open(summary) as f:
        text = f.read()
    weights = re.search(r"weights: (\S+)", text)
    channel = re.search(r"^channel: .*?mask (\w+)", text, flags=re.MULTILINE)
    if channel and channel.group(1) != expected_key:
        raise SystemExit(
            f"{option} {path} holds predictions for mask {channel.group(1)}, not {expected_key}: "
            "were --pred-aav and --pred-gfp swapped?"
        )
    return (weights.group(1) if weights else "model not named") + (
        f" (mask {channel.group(1)})" if channel else " (channel not recorded in its summary)"
    )


def read_prediction(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


# ------------------------------------------------------------------------------------------------------------ output

def write_workbook(path, rows, thresholds, totals, settings, has_predictions, measure_text, min_overlap):
    """The table as an Excel file: one row per section and a totals row; thresholds and settings on two more sheets."""
    from openpyxl import Workbook
    from openpyxl.styles import Border, Font, Side

    regular, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    italic = Font(name="Arial", size=10, italic=True, color="52514E")
    rule = Side(style="thin", color="C3C2B7")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "starter cells"
    header = ["section", "slide", "section number", "ground-truth starter cells"]
    if has_predictions:
        header += ["predicted starter cells", "predicted minus ground truth"]
    header += [f"AAV mask ({AAV_KEY})", f"GFP mask ({GFP_KEY})"]
    sheet.append(header)
    for cell in sheet[1]:
        cell.font, cell.border = bold, Border(bottom=rule)

    for r, row in enumerate(rows, start=2):
        values = [row["section"], row["slide"], row["section_number"], row["gt"]]
        if has_predictions:
            values += [row["pred"], f'=IF(OR(D{r}="",E{r}=""),"",E{r}-D{r})']
        values += [row["aav_kind"], row["gfp_kind"]]
        sheet.append(values)
    last = len(rows) + 1
    total_row = ["total", "", "", f"=SUM(D2:D{last})"]
    if has_predictions:
        total_row += [f"=SUM(E2:E{last})", f"=SUM(F2:F{last})"]
    sheet.append(total_row)
    for row_cells in sheet.iter_rows(min_row=2, max_row=last + 1):
        for cell in row_cells:
            cell.font = bold if cell.row == last + 1 else regular
            if cell.row == last + 1:
                cell.border = Border(top=rule)
    if has_predictions:
        for row_cells in sheet.iter_rows(min_row=2, max_row=last + 1, min_col=6, max_col=6):
            row_cells[0].number_format = "+0;-0;0"
    for column, width in zip("ABCDEFGH", (26, 8, 15, 25, 23, 28, 17, 17)):
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"
    note_row = last + 3
    sheet.cell(note_row, 1, (
        f"A starter cell is an AAV cell and a GFP cell whose overlapping pixels make up at least "
        f"{min_overlap:.0%} of {measure_text}; each cell counts in at most one starter cell. "
        "Settings and sources are on the settings sheet."
    )).font = italic

    sheet = workbook.create_sheet("overlap thresholds")
    sheet.append([f"Total starter cells over all sections when the overlap must reach the given share of {measure_text}."])
    sheet.append([f"The row in bold is the threshold used on the first sheet ({min_overlap:.0%})."])
    sheet.append([])
    header = ["minimum overlap", "ground-truth starter cells"] + (["predicted starter cells"] if has_predictions else [])
    sheet.append(header)
    for cell in sheet[4]:
        cell.font, cell.border = bold, Border(bottom=rule)
    for t in thresholds:
        values = [t, totals["gt"][t]] + ([totals["pred"][t]] if has_predictions else [])
        sheet.append(values)
        for cell in sheet[sheet.max_row]:
            cell.font = bold if np.isclose(t, min_overlap) else regular
        sheet.cell(sheet.max_row, 1).number_format = "0%"
    for row_cells in sheet.iter_rows(min_row=1, max_row=2):
        row_cells[0].font = italic
    for column, width in zip("ABC", (18, 26, 24)):
        sheet.column_dimensions[column].width = width

    sheet = workbook.create_sheet("settings")
    for key, value in settings:
        sheet.append([key, value])
        sheet.cell(sheet.max_row, 1).font, sheet.cell(sheet.max_row, 2).font = bold, regular
    sheet.column_dimensions["A"].width = 36
    sheet.column_dimensions["B"].width = 110
    workbook.save(path)


def write_csv(path, rows, has_predictions):
    """Fallback when openpyxl is missing: the first sheet only, as CSV."""
    columns = ["section", "slide", "section_number", "gt"] + (["pred"] if has_predictions else []) + ["aav_kind", "gfp_kind"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_counts(path, rows, brain, has_predictions, rule_text):
    """Dotted lines of the ground-truth and predicted starter cells for every section, slide by slide."""
    n = len(rows)
    x = np.arange(n)
    gt = np.array([row["gt"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(min(max(8.0, 0.085 * n + 2.0), 40.0), 4.6))

    # predicted (larger squares) under ground truth (smaller circles), so a section where both agree shows both marks
    line_style = {"linestyle": ":", "linewidth": 1.6, "markeredgecolor": "white", "markeredgewidth": 0.8}
    handles = []
    if has_predictions:
        pred = np.array([np.nan if row["pred"] is None else row["pred"] for row in rows], dtype=float)
        handles += ax.plot(x, pred, color=PRED_COLOR, marker="s", markersize=6, zorder=2,
                           label=f"predicted (total {int(np.nansum(pred))})", **line_style)
    handles = ax.plot(x, gt, color=GT_COLOR, marker="o", markersize=4.2, zorder=3,
                      label=f"ground truth (total {int(np.nansum(gt))})", **line_style) + handles

    # section numbers along the axis, slide names under them, a hairline between slides
    slides = [row["slide"] or row["section"] for row in rows]
    ax.set_xticks(x)
    ax.set_xticklabels([row["section_number"] if row["section_number"] is not None else "" for row in rows],
                       fontsize=5.5, rotation=90)
    start = 0
    for i in range(1, n + 1):
        if i == n or slides[i] != slides[start]:
            ax.annotate(
                slides[start], xy=((start + i - 1) / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -19),
                textcoords="offset points", ha="center", va="top", fontsize=8, color=SECONDARY_INK,
            )
            if i < n:
                ax.axvline(i - 0.5, color=GRID_COLOR, linewidth=0.8, zorder=0)
            start = i

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_INK, labelcolor=SECONDARY_INK, length=2)
    ax.set_ylabel("starter cells per section", color=SECONDARY_INK)
    ax.set_xlabel("section, slide by slide", color=SECONDARY_INK, labelpad=18)
    ax.set_title(f"{brain}: starter cells per section", loc="left", color=INK, fontsize=12, pad=34)
    ax.annotate(rule_text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points",
                fontsize=8, color=SECONDARY_INK, va="bottom")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK,
              borderaxespad=0.2, handlelength=2.6)

    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- run

def main():
    parser = argparse.ArgumentParser(description="Count ground-truth and predicted starter cells per section.")
    parser.add_argument("--masks", required=True, help=f"folder of the sections' .mat masks (with {AAV_KEY} and {GFP_KEY})")
    parser.add_argument("--pred-aav", help="results folder (or its predicted_masks folder) of the AAV model's --all run")
    parser.add_argument("--pred-gfp", help="results folder (or its predicted_masks folder) of the GFP model's --all run")
    parser.add_argument("--out", help="where to write the outputs (default: the folder that holds --masks)")
    parser.add_argument("--min-overlap", type=float, default=MIN_OVERLAP, help=f"minimum overlap share (default {MIN_OVERLAP})")
    parser.add_argument("--measure", choices=sorted(OVERLAP_MEASURES), default=OVERLAP_MEASURE,
                        help=f"what the overlap share is taken of (default {OVERLAP_MEASURE}: the smaller of the two cells)")
    args = parser.parse_args()
    if bool(args.pred_aav) != bool(args.pred_gfp):
        parser.error("give both --pred-aav and --pred-gfp, or neither (then only the ground truth is counted)")
    if not 0 < args.min_overlap <= 1:
        parser.error("--min-overlap must lie between 0 and 1, e.g. 0.5")
    if not os.path.isdir(args.masks):
        raise SystemExit(f"{args.masks} is not a folder")

    has_predictions = bool(args.pred_aav)
    measure_text = OVERLAP_MEASURES[args.measure][0]
    brain = brain_of(args.masks)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.masks))
    gt_dir, pred_dir = os.path.join(out_dir, GT_FOLDER), os.path.join(out_dir, PRED_FOLDER)
    os.makedirs(gt_dir, exist_ok=True)

    mask_files = {}
    for path in sorted(glob(os.path.join(args.masks, "*.mat"))):
        name = section_of_mask(path)
        if name in mask_files:
            raise SystemExit(f"two mask files belong to section {name}: {mask_files[name]} and {path}")
        mask_files[name] = path
    if not mask_files:
        raise SystemExit(f"no .mat files in {args.masks}")
    sections = sorted(mask_files, key=order_key)
    print(f"{brain}: {len(sections)} sections with masks in {args.masks}")

    aav_files = gfp_files = {}
    aav_model = gfp_model = ""
    if has_predictions:
        aav_model = model_of_run(args.pred_aav, AAV_KEY, "--pred-aav")
        gfp_model = model_of_run(args.pred_gfp, GFP_KEY, "--pred-gfp")
        aav_files = {section_of_prediction(p): p for p in glob(os.path.join(prediction_folder(args.pred_aav), "*.png"))}
        gfp_files = {section_of_prediction(p): p for p in glob(os.path.join(prediction_folder(args.pred_gfp), "*.png"))}
        print(f"predicted AAV masks: {len(aav_files)} from {aav_model}")
        print(f"predicted GFP masks: {len(gfp_files)} from {gfp_model}")
        os.makedirs(pred_dir, exist_ok=True)

    thresholds = sorted({round(t, 4) for t in OTHER_THRESHOLDS} | {round(args.min_overlap, 4)})
    totals = {"gt": dict.fromkeys(thresholds, 0), "pred": dict.fromkeys(thresholds, 0)}
    rows, missing_predictions = [], []
    print(f"starter cell: overlap of at least {args.min_overlap:.0%} of {measure_text}")
    for number, name in enumerate(sections, start=1):
        data = loadmat(mask_files[name], variable_names=[AAV_KEY, GFP_KEY])
        absent = [key for key in (AAV_KEY, GFP_KEY) if key not in data]
        if absent:
            raise SystemExit(f"{mask_files[name]} has no {' or '.join(absent)}")
        aav, gfp = crop_to_common(data[AAV_KEY] > 0, data[GFP_KEY] > 0)
        starter, shares, aav_areas, gfp_areas = find_starter_cells(aav, gfp, args.min_overlap, args.measure)
        save_mask(starter, os.path.join(gt_dir, f"{name}.mat"))
        slide, section_number = slide_and_number(name)
        row = {
            "section": name,
            "slide": slide,
            "section_number": section_number,
            "gt": int((shares >= args.min_overlap).sum()),
            "pred": None,
            "aav_kind": mask_kind(aav_areas),
            "gfp_kind": mask_kind(gfp_areas),
        }
        for t in thresholds:
            totals["gt"][t] += int((shares >= t).sum())

        if has_predictions:
            if name in aav_files and name in gfp_files:
                predicted_aav, predicted_gfp = crop_to_common(read_prediction(aav_files[name]), read_prediction(gfp_files[name]))
                starter, shares, _, _ = find_starter_cells(predicted_aav, predicted_gfp, args.min_overlap, args.measure)
                save_mask(starter, os.path.join(pred_dir, f"{name}.mat"))
                row["pred"] = int((shares >= args.min_overlap).sum())
                for t in thresholds:
                    totals["pred"][t] += int((shares >= t).sum())
            else:
                missing_predictions.append(name)
        rows.append(row)
        print(
            f"  {number}/{len(sections)} {name}: ground truth {row['gt']}"
            + (f", predicted {row['pred'] if row['pred'] is not None else 'n/a (prediction missing)'}" if has_predictions else "")
            + (f"   [masks: AAV {row['aav_kind']}, GFP {row['gfp_kind']}]" if "dots" in (row["aav_kind"], row["gfp_kind"]) else "")
        )

    unmatched = sorted((set(aav_files) | set(gfp_files)) - set(mask_files)) if has_predictions else []
    settings = [
        ("brain", brain),
        ("masks", os.path.abspath(args.masks)),
        ("AAV mask / GFP mask", f"{AAV_KEY} / {GFP_KEY}"),
        ("predicted AAV masks", f"{os.path.abspath(prediction_folder(args.pred_aav))} - {aav_model}" if has_predictions else "not given"),
        ("predicted GFP masks", f"{os.path.abspath(prediction_folder(args.pred_gfp))} - {gfp_model}" if has_predictions else "not given"),
        ("minimum overlap", f"{args.min_overlap:.0%} of {measure_text} (measure '{args.measure}')"),
        ("pairing", "one to one, best overlap first; cells are connected pixels, diagonals included"),
        ("starter masks", f"{gt_dir} ; {pred_dir}" if has_predictions else gt_dir),
        ("sections without both predicted masks", ", ".join(missing_predictions) or "none"),
        ("created", f"{datetime.now():%Y-%m-%d %H:%M}"),
    ]
    table_path = os.path.join(out_dir, f"{brain}_starter_cells.xlsx")
    try:
        write_workbook(table_path, rows, thresholds, totals, settings, has_predictions, measure_text, args.min_overlap)
    except ImportError:
        table_path = os.path.join(out_dir, f"{brain}_starter_cells.csv")
        write_csv(table_path, rows, has_predictions)
        print("NOTE: openpyxl is not installed, so the table was saved as CSV; pip install openpyxl for the Excel file")
    rule_text = f"Starter cell: an AAV cell and a GFP cell overlapping by at least {args.min_overlap:.0%} of {measure_text}"
    plot_path = os.path.join(out_dir, f"{brain}_starter_cells_per_section.pdf")
    plot_counts(plot_path, rows, brain, has_predictions, rule_text)

    print(f"\nground-truth starter cells: {sum(row['gt'] for row in rows)} in {len(rows)} sections")
    if has_predictions:
        both = [row for row in rows if row["pred"] is not None]
        print(
            f"predicted starter cells:    {sum(row['pred'] for row in both)} in {len(both)} sections "
            f"(ground truth in those sections: {sum(row['gt'] for row in both)})"
        )
        if missing_predictions:
            print(f"sections without both predicted masks ({len(missing_predictions)}): {', '.join(missing_predictions[:10])}"
                  + (" ..." if len(missing_predictions) > 10 else ""))
        if unmatched:
            print(f"predicted masks without a .mat mask, not used ({len(unmatched)}): {', '.join(unmatched[:10])}")
    dotted = [row["section"] for row in rows if "dots" in (row["aav_kind"], row["gfp_kind"])]
    if dotted:
        print(f"{len(dotted)} sections have dot markers in at least one ground-truth mask; see the mask columns of the table")
    print(f"starter masks: {gt_dir}" + (f" and {pred_dir}" if has_predictions else ""))
    print(f"table: {table_path}")
    print(f"plot:  {plot_path}")


if __name__ == "__main__":
    main()
