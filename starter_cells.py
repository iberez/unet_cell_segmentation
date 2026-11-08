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

Injection-site gate (predictions only, on by default): real AAV cells sit in one compact cluster at the injection
site, in an unbroken run of sections, while the AAV model's false positives are scattered over the whole brain and
often sit on GFP cells, so they become false starter cells. The gate
    1. scores each section by the predicted AAV pixels in its densest CLUSTER_RADIUS (300 px) disc, in cells' worth;
    2. starts at the peak, the highest score that a neighbouring section also reaches (so one section with a bright
       artefact can't pass for the injection site), and walks out, slide by slide and section by section, while the
       score stays at or above max(RUN_MIN_CELLS, RUN_FRACTION x the peak); that run is the injection site;
    3. empties the AAV prediction of every section outside the run, so they get 0 predicted starter cells;
    4. inside the run, drops AAV objects more than SITE_RADIUS (1000 px) from the section's cluster centre. Set
       SITE_RADIUS_FILTER = False to switch this step off, or use --no-site-radius for one run.
--no-gate switches the whole gate off for one run (INJECTION_GATE = False for good). The ground truth is never gated.

Written next to the masks folder (or into --out):
    starter_cells_gt_mask/<section>.mat      ground-truth starter cells: one logical (boolean) mask, BW_starter
    starter_cells_pred_mask/<section>.mat    predicted starter cells (after the gate), the same way
    <brain>_starter_cells.xlsx               one row per section: ground-truth and predicted starter cells, with
                                             totals; a sheet with the totals at other overlap thresholds; an AAV
                                             cells sheet (ground-truth and predicted AAV cells per section); an
                                             injection site sheet (each section's cluster score, whether it is in
                                             the run, predicted counts before and after the gate); settings
    <brain>_starter_cells_per_section.pdf    both counts for every section, slide by slide; with the gate, a panel
                                             on top shows the cluster scores, the cutoff and the run
    <brain>_aav_cells_per_section.pdf        the same for AAV cells alone (compare with aav_10x_filter.py's plot)

Needs numpy, scipy, Pillow, matplotlib and openpyxl (not torch or MONAI), so it also runs without a GPU.
"""

import argparse
import csv
import os
import re
import textwrap
from datetime import datetime
from glob import glob

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.io import loadmat, savemat
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")  # no display on the server: draw straight to file
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

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

# --- injection-site gate (predictions only; the ground truth is never gated) ---
# Real AAV cells sit in one compact cluster around the injection site, in an unbroken run of sections. The gate finds
# that run from the predicted AAV masks and empties the AAV prediction of every other section, so those sections get
# 0 predicted starter cells. (False AAV detections elsewhere are mostly bright things that also show up in the GFP
# channel, so the AND with the GFP prediction can't remove them.) --no-gate switches the gate off for one run.
INJECTION_GATE = True
CLUSTER_RADIUS = 300    # px: a section's cluster score is the predicted AAV area inside its densest disc of this radius,
                        # in cells' worth (that area divided by the median area of the brain's predicted AAV objects)
RUN_MIN_CELLS = 5       # sections stay in the run while their score is at least this many cells' worth ...
RUN_FRACTION = 0.03     # ... and at least this share of the peak section's score
RUN_MAX_GAP = 1         # up to this many sections in a row may fall below that inside the run, if the next one passes
# Inside the run, also drop AAV objects far from the section's cluster. Set SITE_RADIUS_FILTER = False to switch this
# off for good; --no-site-radius (or --site-radius) overrides it for one run.
SITE_RADIUS_FILTER = True
SITE_RADIUS = 1000      # px from the centre of the section's cluster

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

AAV_RULE_TEXT = (f"AAV cell: a connected object in the AAV mask ({AAV_KEY} for the ground truth, the AAV model's mask for the "
                 "predictions, where touching cells count as one)")

# chart colours (validated as a two-series categorical palette) and ink
GT_COLOR, PRED_COLOR = "#2a78d6", "#eb6834"
INK, SECONDARY_INK, MUTED_INK = "#0b0b0b", "#52514e", "#898781"
GRID_COLOR, AXIS_COLOR = "#e1e0d9", "#c3c2b7"
RUN_BAND_COLOR = "#f1f0ea"  # background of the injection-site run


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


def count_objects(mask):
    """Number of cells (connected pixels, diagonals included) in a mask."""
    return int(ndimage.label(mask, structure=STRUCTURE)[1])


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


# ----------------------------------------------------------------------------------------------- injection-site gate

def object_centres(mask):
    """Labels, centres (x, y) and areas in pixels of the objects in a mask (connected pixels, diagonals included)."""
    labels, n = ndimage.label(mask, structure=STRUCTURE)
    if n == 0:
        return labels, np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    rows, cols = np.nonzero(labels)
    which = labels[rows, cols]
    area = np.bincount(which, minlength=n + 1)[1:]
    x = np.bincount(which, weights=cols, minlength=n + 1)[1:] / area
    y = np.bincount(which, weights=rows, minlength=n + 1)[1:] / area
    return labels, np.column_stack([x, y]), area


def densest_disc(centres, areas, radius, near=None, within=None):
    """The densest disc of the given radius centred on one of the objects.

    Returns (predicted pixels inside, area-weighted centre of the objects inside); an object is inside when its
    centre is. With near and within, only discs centred within that distance of the point near are tried.
    (0, None) when there is no object to try.
    """
    if len(areas) == 0:
        return 0.0, None
    tree = cKDTree(centres)
    inside = np.zeros(len(areas))
    for start in range(0, len(areas), 256):   # in chunks, so a section with a huge number of objects stays in memory
        for i, members in enumerate(tree.query_ball_point(centres[start:start + 256], radius), start=start):
            inside[i] = areas[members].sum()
    tried = np.arange(len(areas))
    if near is not None:
        tried = tried[np.hypot(centres[:, 0] - near[0], centres[:, 1] - near[1]) <= within]
        if len(tried) == 0:
            return 0.0, None
    best = tried[np.argmax(inside[tried])]
    members = tree.query_ball_point(centres[best], radius)
    weights = areas[members].astype(float)
    return float(inside[best]), (centres[members] * weights[:, None]).sum(axis=0) / weights.sum()


def walk_out(passes, start, step, max_gap):
    """Last index of the run that starts at start and goes in direction step (-1 or +1) while sections pass.

    Up to max_gap sections in a row may fail, as long as the section right after them passes again.
    """
    i = start
    while True:
        for k in range(1, max_gap + 2):
            j = i + k * step
            if not 0 <= j < len(passes):
                return i
            if passes[j]:
                i = j
                break
        else:
            return i


def pick_run(score, min_score, fraction, max_gap):
    """The injection-site run in a list of per-section scores (slide/section order). Also used by aav_10x_filter.py.

    The peak is the section with the highest score that one of its neighbours also reaches, so a single section
    with a strong artefact (a bright streak) can't be taken for the injection site. The run grows from the peak in
    both directions while the score stays at or above the cutoff, max(min_score, fraction x the peak's score),
    bridging up to max_gap sections in a row that fall below it. Returns a dict: peak, cutoff, backed (the score
    that the peak and a neighbour both reach), passes (each section's score >= cutoff), first and last (indices of
    the run; None when no two neighbouring sections both reach min_score).
    """
    score = np.asarray(score, dtype=float)
    # the level a section and at least one of its neighbours both reach: an isolated spike only gets its neighbours'
    neighbour = np.array([max(score[i - 1] if i > 0 else 0.0, score[i + 1] if i + 1 < len(score) else 0.0)
                          for i in range(len(score))])
    backed = np.minimum(score, neighbour)
    peak = max(range(len(score)), key=lambda i: (backed[i], score[i]))
    cutoff = max(min_score, fraction * score[peak])
    passes = score >= cutoff
    found = backed[peak] >= min_score
    return {"peak": peak, "cutoff": float(cutoff), "backed": float(backed[peak]), "passes": passes,
            "first": walk_out(passes, peak, -1, max_gap) if found else None,
            "last": walk_out(passes, peak, 1, max_gap) if found else None}


def find_injection_site(names, objects, site_radius_filter):
    """Find the injection-site run among the sections in names (slide/section order, each with a predicted AAV mask).

    objects maps each name to the (centres, areas) of its predicted AAV objects. Steps:
      1. score each section: the predicted AAV pixels in its densest CLUSTER_RADIUS disc, in cells' worth;
      2. find the peak: the section with the highest score that one of its neighbours also reaches, so a single
         section with a strong artefact (a bright streak) can't be taken for the injection site; walk out from it in
         both directions while the score stays at or above the cutoff, max(RUN_MIN_CELLS, RUN_FRACTION x the peak's
         score), bridging up to RUN_MAX_GAP sections in a row that fall below it;
      3. give every section of the run a cluster centre, for the in-section filter: the peak's densest disc, then,
         section by section outward, the densest disc within SITE_RADIUS of the neighbour's centre, so a denser
         artefact elsewhere in the section can't pull the centre away. A section with nothing that near takes its
         own densest disc if that reaches the cutoff (the section sits elsewhere in its image), else the neighbour's.
    """
    gate = {"applied": False, "names": list(names), "score": {}, "objects": {n: len(objects[n][1]) for n in names},
            "run": [], "centre": {}, "site_radius_filter": site_radius_filter}
    areas = [objects[name][1] for name in names]
    if not names or not sum(len(a) for a in areas):
        gate["status"] = "not applied: there are no predicted AAV objects in any section"
        return gate
    cell_area = float(np.median(np.concatenate(areas)))
    score = np.array([densest_disc(*objects[name], CLUSTER_RADIUS)[0] / cell_area for name in names])
    found = pick_run(score, RUN_MIN_CELLS, RUN_FRACTION, RUN_MAX_GAP)
    peak, cutoff, passes, first, last = (found[k] for k in ("peak", "cutoff", "passes", "first", "last"))
    gate.update(score=dict(zip(names, score)), cell_area=cell_area, peak=names[peak], peak_score=float(score[peak]),
                cutoff=cutoff)
    if first is None:
        gate["status"] = (f"not applied: no two neighbouring sections both reach {RUN_MIN_CELLS} cells' worth of "
                          f"predicted AAV in one {CLUSTER_RADIUS} px disc (closest: {found['backed']:.1f}, at {names[peak]}), "
                          "so no injection site was found and the predictions were left as they were")
        return gate

    centre = {names[peak]: densest_disc(*objects[names[peak]], CLUSTER_RADIUS)[1]}
    for step, end in ((-1, first), (1, last)):
        previous = centre[names[peak]]
        for i in range(peak + step, end + step, step):
            centres, object_areas = objects[names[i]]
            _, found = densest_disc(centres, object_areas, CLUSTER_RADIUS, near=previous, within=SITE_RADIUS)
            if found is None:
                pixels, own = densest_disc(centres, object_areas, CLUSTER_RADIUS)
                found = own if own is not None and pixels / cell_area >= cutoff else previous
            centre[names[i]] = previous = found
    run = names[first:last + 1]
    gate.update(applied=True, run=run, run_set=set(run), centre=centre,
                bridged=[names[i] for i in range(first, last + 1) if not passes[i]])
    gate["status"] = (f"on: kept the {len(run)} sections from {run[0]} to {run[-1]}; the AAV predictions of the other "
                      f"{len(names) - len(run)} sections were emptied")
    return gate


def gate_aav(mask, name, gate):
    """A section's predicted AAV mask after the gate, and how many of its objects the gate dropped."""
    if not gate["applied"]:
        return mask, 0
    if name not in gate["run_set"]:
        return np.zeros_like(mask), gate["objects"].get(name, 0)
    if not gate["site_radius_filter"]:
        return mask, 0
    labels, centres, _ = object_centres(mask)
    x, y = gate["centre"][name]
    far = np.hypot(centres[:, 0] - x, centres[:, 1] - y) > SITE_RADIUS
    if not far.any():
        return mask, 0
    return np.concatenate([[False], ~far])[labels], int(far.sum())


# ------------------------------------------------------------------------------------------------------------ output

def write_workbook(path, rows, thresholds, totals, settings, has_predictions, measure_text, min_overlap, gate):
    """The table as an Excel file: one row per section and a totals row; thresholds, the injection-site gate (when it
    scored the sections) and settings on further sheets."""
    from openpyxl import Workbook
    from openpyxl.styles import Border, Font, PatternFill, Side

    regular, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    italic = Font(name="Arial", size=10, italic=True, color="52514E")
    rule = Side(style="thin", color="C3C2B7")
    gated = gate["applied"]

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "starter cells"
    header = ["section", "slide", "section number", "ground-truth starter cells"]
    if has_predictions:
        header += ["predicted starter cells" + (" (after the gate)" if gated else ""), "predicted minus ground truth"]
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
    for column, width in zip("ABCDEFGH", (26, 8, 15, 25, 36 if gated else 23, 28, 17, 17)):
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"
    note_row = last + 3
    sheet.cell(note_row, 1, (
        f"A starter cell is an AAV cell and a GFP cell whose overlapping pixels make up at least "
        f"{min_overlap:.0%} of {measure_text}; each cell counts in at most one starter cell. "
        "Settings and sources are on the settings sheet."
    )).font = italic
    if gated:
        sheet.cell(note_row + 1, 1, (
            "Predicted starter cells are counted after the injection-site gate: AAV predictions outside the run of "
            "sections around the injection site were emptied. The injection site sheet has the counts before it."
        )).font = italic

    sheet = workbook.create_sheet("overlap thresholds")
    sheet.append([f"Total starter cells over all sections when the overlap must reach the given share of {measure_text}."])
    sheet.append([f"The row in bold is the threshold used on the first sheet ({min_overlap:.0%})."
                  + (" Predicted totals are after the injection-site gate." if gated else "")])
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

    write_aav_sheet(workbook.create_sheet("AAV cells"), rows, has_predictions, gated, regular, bold, italic, rule, Border)
    if gate.get("score"):
        write_gate_sheet(workbook.create_sheet("injection site"), rows, gate, regular, bold, italic, rule,
                         PatternFill("solid", fgColor=RUN_BAND_COLOR.lstrip("#").upper()), Border)

    sheet = workbook.create_sheet("settings")
    for key, value in settings:
        sheet.append([key, value])
        sheet.cell(sheet.max_row, 1).font, sheet.cell(sheet.max_row, 2).font = bold, regular
    sheet.column_dimensions["A"].width = 36
    sheet.column_dimensions["B"].width = 110
    workbook.save(path)


def write_aav_sheet(sheet, rows, has_predictions, gated, regular, bold, italic, rule, Border):
    """One row per section: ground-truth and predicted AAV cells (after the gate, and before it when it ran)."""
    sheet.append([AAV_RULE_TEXT + "."])
    if not has_predictions:
        sheet.append(["No predicted masks were given, so only the ground truth is counted."])
    elif gated:
        sheet.append(["Predicted AAV cells are counted after the injection-site gate; the last count column has them "
                      "before it."])
    else:
        sheet.append(["Predicted AAV cells are counted from the AAV model's masks as they are (no gate)."])
    sheet.append([])
    header = ["section", "slide", "section number", "ground-truth AAV cells"]
    if has_predictions:
        header += ["predicted AAV cells" + (" (after the gate)" if gated else ""), "predicted minus ground truth"]
        if gated:
            header += ["predicted AAV cells before the gate"]
    header += [f"AAV mask ({AAV_KEY})"]
    sheet.append(header)
    header_row = sheet.max_row
    for cell in sheet[header_row]:
        cell.font, cell.border = bold, Border(bottom=rule)
    first = header_row + 1
    for r, row in enumerate(rows, start=first):
        values = [row["section"], row["slide"], row["section_number"], row["gt_aav"]]
        if has_predictions:
            values += [row["aav_pred"], f'=IF(OR(D{r}="",E{r}=""),"",E{r}-D{r})']
            if gated:
                values += [row["aav_pred_before_gate"]]
        values += [row["aav_kind"]]
        sheet.append(values)
        for cell in sheet[r]:
            cell.font = regular
        if has_predictions:
            sheet.cell(r, 6).number_format = "+0;-0;0"
    last = first + len(rows) - 1
    totals = ["total", "", "", f"=SUM(D{first}:D{last})"]
    if has_predictions:
        totals += [f"=SUM(E{first}:E{last})", f"=SUM(F{first}:F{last})"] + ([f"=SUM(G{first}:G{last})"] if gated else [])
    sheet.append(totals)
    for cell in sheet[sheet.max_row]:
        cell.font, cell.border = bold, Border(top=rule)
    if has_predictions:
        sheet.cell(sheet.max_row, 6).number_format = "+0;-0;0"
    for row_cells in sheet.iter_rows(min_row=1, max_row=2):
        row_cells[0].font = italic
    for column, width in zip("ABCDEFGH", (26, 8, 15, 22, 32 if gated else 20, 28, 34, 17)):
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = sheet.cell(first, 1)


def write_gate_sheet(sheet, rows, gate, regular, bold, italic, rule, run_fill, Border):
    """One row per section: its cluster score, whether it is in the run, and the predicted counts before and after."""
    sheet.append([f"Injection-site gate {gate['status']}."])
    sheet.append([
        f"Cluster score: predicted AAV pixels in the section's densest {CLUSTER_RADIUS} px disc, in cells' worth (one cell = "
        f"{gate['cell_area']:.0f} px, the median predicted AAV object). Peak: {gate['peak']}, {gate['peak_score']:.0f}. "
        f"Cutoff: {gate['cutoff']:.1f}. Shaded rows are the run; the centre is the cluster the in-section filter measured from."
    ])
    sheet.append([])
    header = ["section", "slide", "section number", "predicted AAV objects", "cluster score (cells' worth)", "in the run",
              "AAV objects dropped", "predicted starter cells before the gate", "predicted starter cells after the gate",
              "removed by the gate", "cluster centre x (px)", "cluster centre y (px)"]
    sheet.append(header)
    header_row = sheet.max_row
    for cell in sheet[header_row]:
        cell.font, cell.border = bold, Border(bottom=rule)
    first = header_row + 1
    for r, row in enumerate(rows, start=first):
        in_run = {True: "yes", False: "no", None: ""}[row["in_run"]]
        centre = row["centre"] if row["centre"] is not None else (None, None)
        sheet.append([
            row["section"], row["slide"], row["section_number"], row["aav_objects"],
            None if row["cluster_score"] is None else round(float(row["cluster_score"]), 2), in_run, row["aav_dropped"],
            row["pred_before_gate"], row["pred"], f'=IF(OR(H{r}="",I{r}=""),"",H{r}-I{r})',
            None if centre[0] is None else round(float(centre[0])), None if centre[1] is None else round(float(centre[1])),
        ])
        for cell in sheet[r]:
            cell.font = regular
            if row["in_run"]:
                cell.fill = run_fill
        sheet.cell(r, 5).number_format = "0.0"
    last = first + len(rows) - 1
    sheet.append(["total", "", "", f"=SUM(D{first}:D{last})", "", f'=COUNTIF(F{first}:F{last},"yes")',
                  f"=SUM(G{first}:G{last})", f"=SUM(H{first}:H{last})", f"=SUM(I{first}:I{last})", f"=SUM(J{first}:J{last})"])
    for cell in sheet[sheet.max_row]:
        cell.font, cell.border = bold, Border(top=rule)
    for row_cells in sheet.iter_rows(min_row=1, max_row=2):
        row_cells[0].font = italic
    for column, width in zip("ABCDEFGHIJKL", (26, 8, 15, 21, 26, 11, 20, 36, 35, 19, 20, 20)):
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = sheet.cell(first, 1)


def write_csv(path, rows, has_predictions, gate):
    """Fallback when openpyxl is missing: the first sheet only, as CSV (with the gate's columns when it ran)."""
    columns = ["section", "slide", "section_number", "gt"] + (["pred"] if has_predictions else [])
    if gate.get("score"):
        columns += ["pred_before_gate", "in_run", "cluster_score", "aav_objects", "aav_dropped"]
    columns += ["gt_aav"] + (["aav_pred"] if has_predictions else []) + (["aav_pred_before_gate"] if gate["applied"] else [])
    columns += ["aav_kind", "gfp_kind"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax):
    """Recessive grid and axes."""
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_INK, labelcolor=SECONDARY_INK, length=2)


def quietest_stretch(values, span):
    """Start index of the span-long stretch whose highest value is lowest (where a label on the cutoff line fits)."""
    filled = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    if len(filled) <= span:
        return 0
    highest = [filled[i:i + span].max() for i in range(len(filled) - span + 1)]
    return int(np.argmin(highest))


def kept_stretches(rows):
    """(first, last) row index of each stretch of kept sections (in_run True). Rows without a decision (in_run None,
    e.g. no prediction) between kept rows don't break a stretch."""
    stretches, start, last_kept = [], None, None
    for i, row in enumerate(rows):
        if row.get("in_run"):
            start = i if start is None else start
            last_kept = i
        elif row.get("in_run") is False and start is not None:
            stretches.append((start, last_kept))
            start = None
    if start is not None:
        stretches.append((start, last_kept))
    return stretches


def plot_score_panel(top, rows, panel, line_style):
    """The top panel: each section's score (log-like scale), the cutoff, and a label on the longest kept stretch."""
    n = len(rows)
    score = np.array([np.nan if row.get(panel["field"]) is None else row[panel["field"]] for row in rows], dtype=float)
    top.plot(np.arange(n), score, color=SECONDARY_INK, marker="o", markersize=4.2, zorder=3, **line_style)
    top.set_yscale("symlog", linthresh=1, linscale=0.5)
    highest = max(np.nanmax(score) if np.isfinite(score).any() else 0, panel["cutoff"], 10)
    top.set_ylim(-0.1, highest * 2.2)
    top.yaxis.set_major_locator(FixedLocator([t for t in (0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if t <= highest * 2.2]))
    top.yaxis.set_minor_locator(NullLocator())
    top.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    top.axhline(panel["cutoff"], color=SECONDARY_INK, linewidth=1.0, linestyle=(0, (5, 3)), zorder=2)
    # the label sits on the cutoff line where the scores stay lowest over the label's width
    label = top.annotate(f"cutoff {panel['cutoff']:.1f}", xy=(0, panel["cutoff"]), xytext=(0, 4), textcoords="offset points",
                         ha="left", va="bottom", fontsize=8, color=SECONDARY_INK)
    renderer = top.figure.canvas.get_renderer()
    sections_wide = label.get_window_extent(renderer).width / top.get_window_extent(renderer).width * (n + 0.4)
    label.xy = (quietest_stretch(score, int(np.ceil(sections_wide)) + 1), panel["cutoff"])
    stretches = kept_stretches(rows)
    if stretches:
        first, last = max(stretches, key=lambda s: s[1] - s[0])
        top.annotate(panel["band_label"], xy=((first + last) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, -3),
                     textcoords="offset points", ha="center", va="top", fontsize=8, color=SECONDARY_INK)
    top.set_ylabel(panel["ylabel"], color=SECONDARY_INK)


def label_sections(ax, rows, slides):
    """Section numbers along the x axis, slide names under them."""
    n = len(rows)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([row["section_number"] if row["section_number"] is not None else "" for row in rows],
                       fontsize=5.5, rotation=90)
    start = 0
    for i in range(1, n + 1):
        if i == n or slides[i] != slides[start]:
            ax.annotate(slides[start], xy=((start + i - 1) / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -19),
                        textcoords="offset points", ha="center", va="top", fontsize=8, color=SECONDARY_INK)
            start = i
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xlabel("section, slide by slide", color=SECONDARY_INK, labelpad=18)


def plot_sections(path, rows, title, rule_text, counts, panel=None):
    """Dotted lines of the ground-truth and predicted counts for every section, slide by slide.

    counts: what to draw below: "gt", "pred" and "before" name the row fields (pred and before may be None),
    "ylabel", "pred_label" and "before_label" the text. With a panel (field, ylabel, cutoff, band_label, text), a
    panel on top shows each section's score with the cutoff, the kept sections are shaded on both panels, and where
    the filter changed a section's predicted count, the count before it is shown as a grey square.
    Also used by aav_10x_filter.py, so both scripts draw the same plot.
    """
    n = len(rows)
    x = np.arange(n)
    width = min(max(8.0, 0.085 * n + 2.0), 40.0)
    if panel and counts:
        fig, (top, ax) = plt.subplots(2, 1, figsize=(width, 8.8), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.34})
    elif panel:   # the score panel alone (no counts to show)
        fig, top = plt.subplots(figsize=(width, 4.0))
        ax = None
    else:
        fig, ax = plt.subplots(figsize=(width, 4.6))
        top = None

    # the kept sections shaded behind both panels, a hairline between slides
    slides = [row["slide"] or row["section"] for row in rows]
    for axes in (a for a in (top, ax) if a is not None):
        for first, last in kept_stretches(rows):
            axes.axvspan(first - 0.5, last + 0.5, color=RUN_BAND_COLOR, linewidth=0, zorder=0)
        for i in range(1, n):
            if slides[i] != slides[i - 1]:
                axes.axvline(i - 0.5, color=GRID_COLOR, linewidth=0.8, zorder=0.5)
        style_axes(axes)
    line_style = {"linestyle": ":", "linewidth": 1.6, "markeredgecolor": "white", "markeredgewidth": 0.8}
    bottom = ax if ax is not None else top
    label_sections(bottom, rows, slides)
    if ax is None:
        plot_score_panel(top, rows, panel, line_style)
        characters = max(60, int(top.get_position().width * fig.get_figwidth() * 15))
        subtitle = top.annotate("\n".join(textwrap.wrap(rule_text, characters) + textwrap.wrap(panel["text"], characters)),
                                xy=(0, 1), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                                fontsize=8, color=SECONDARY_INK, va="bottom", linespacing=1.5)
        height = subtitle.get_window_extent(fig.canvas.get_renderer()).height * 72 / fig.dpi
        top.set_title(title, loc="left", color=INK, fontsize=12, pad=10 + height + 6)
        fig.savefig(path, transparent=True, bbox_inches="tight")
        plt.close(fig)
        return

    # predicted (larger squares) under ground truth (smaller circles), so a section where both agree shows both marks
    gt = np.array([row[counts["gt"]] for row in rows], dtype=float)
    handles, extra = [], []
    if counts.get("pred"):
        pred = np.array([np.nan if row[counts["pred"]] is None else row[counts["pred"]] for row in rows], dtype=float)
        handles += ax.plot(x, pred, color=PRED_COLOR, marker="s", markersize=6, zorder=2,
                           label=f"{counts['pred_label']} (total {int(np.nansum(pred))})", **line_style)
        if counts.get("before"):
            before = [row[counts["before"]] for row in rows]
            changed = np.array([np.nan if b is None or b == row[counts["pred"]] else b for b, row in zip(before, rows)],
                               dtype=float)
            extra = ax.plot(x, changed, color=MUTED_INK, marker="s", markersize=5, linestyle="none", markeredgecolor="white",
                            markeredgewidth=0.8, zorder=1.5,
                            label=f"{counts['before_label']}, where it differs (total {sum(b for b in before if b is not None)})")
    handles = ax.plot(x, gt, color=GT_COLOR, marker="o", markersize=4.2, zorder=3,
                      label=f"ground truth (total {int(np.nansum(gt))})", **line_style) + handles + extra

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(counts["ylabel"], color=SECONDARY_INK)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK,
              borderaxespad=0.2, handlelength=2.6)

    if panel:
        plot_score_panel(top, rows, panel, line_style)
        # wrap to the width of the panel (about 15 characters per inch at 8 pt), then lift the title above it
        characters = max(60, int(top.get_position().width * fig.get_figwidth() * 15))
        subtitle = top.annotate("\n".join(textwrap.wrap(rule_text, characters) + textwrap.wrap(panel["text"], characters)), xy=(0, 1),
                                xycoords="axes fraction", xytext=(0, 10), textcoords="offset points", fontsize=8,
                                color=SECONDARY_INK, va="bottom", linespacing=1.5)
        height = subtitle.get_window_extent(fig.canvas.get_renderer()).height * 72 / fig.dpi
        top.set_title(title, loc="left", color=INK, fontsize=12, pad=10 + height + 6)
    else:
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=34)
        ax.annotate(rule_text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points",
                    fontsize=8, color=SECONDARY_INK, va="bottom")

    fig.savefig(path, transparent=True, bbox_inches="tight")
    plt.close(fig)


def gate_panel(gate):
    """The score panel for the gate's plots, or None when the gate didn't score the sections."""
    if not gate.get("score"):
        return None
    text = (f"Top: the AAV cluster score, the predicted AAV in each section's densest {CLUSTER_RADIUS} px disc; dashed: the "
            f"cutoff, at least {RUN_MIN_CELLS} cells' worth and {RUN_FRACTION:.0%} of the peak. ")
    if gate["applied"]:
        text += ("Shaded: the injection-site run; AAV predictions outside it were dropped"
                 + (f", and inside it, AAV objects more than {SITE_RADIUS} px from the section's cluster."
                    if gate["site_radius_filter"] else "."))
    else:
        text += f"Gate {gate['status']}."
    return {"field": "cluster_score", "ylabel": "AAV cluster score\n(cells' worth)", "cutoff": gate["cutoff"],
            "band_label": "injection-site run (kept)", "text": text}


def plot_counts(path, rows, brain, has_predictions, rule_text, gate, kind="starter"):
    """The per-section plot of starter cells (kind "starter") or AAV cells (kind "aav"), with the gate's panel."""
    gt, pred, before, ylabel = {"starter": ("gt", "pred", "pred_before_gate", "starter cells per section"),
                                "aav": ("gt_aav", "aav_pred", "aav_pred_before_gate", "AAV cells per section")}[kind]
    counts = {"gt": gt, "pred": pred if has_predictions else None, "before": before if gate["applied"] else None,
              "ylabel": ylabel, "pred_label": "predicted after the gate" if gate["applied"] else "predicted",
              "before_label": "before the gate"}
    plot_sections(path, rows, f"{brain}: {ylabel}", rule_text, counts, gate_panel(gate))


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
    parser.add_argument("--gate", action=argparse.BooleanOptionalAction, default=INJECTION_GATE,
                        help="empty the AAV predictions outside the injection-site run (default: "
                             f"{'on' if INJECTION_GATE else 'off'}, INJECTION_GATE); --no-gate switches it off")
    parser.add_argument("--site-radius", action=argparse.BooleanOptionalAction, default=SITE_RADIUS_FILTER,
                        help=f"inside the run, also drop AAV objects more than {SITE_RADIUS} px from the section's cluster "
                             f"(default: {'on' if SITE_RADIUS_FILTER else 'off'}, SITE_RADIUS_FILTER); "
                             "--no-site-radius switches it off")
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

    gate = {"applied": False, "status": "off (--no-gate)" if has_predictions else "not used: no predictions given"}
    if has_predictions and args.gate:
        gate_names = [name for name in sections if name in aav_files]
        print(f"injection-site gate: scoring the predicted AAV masks of {len(gate_names)} sections")
        objects = {}
        for number, name in enumerate(gate_names, start=1):
            objects[name] = object_centres(read_prediction(aav_files[name]))[1:]
            if number % 25 == 0 or number == len(gate_names):
                print(f"  {number}/{len(gate_names)}")
        gate = find_injection_site(gate_names, objects, args.site_radius)
        print(f"injection-site gate {gate['status']}")
        if "peak" in gate:
            print(f"  peak {gate['peak']}: {gate['peak_score']:.0f} cells' worth (one cell = {gate['cell_area']:.0f} px); "
                  f"cutoff {gate['cutoff']:.1f} cells' worth")
        if gate["applied"]:
            if gate["bridged"]:
                print(f"  bridged, below the cutoff but inside the run: {', '.join(gate['bridged'])}")
            print(f"  in-section filter: " + (f"on, AAV objects more than {SITE_RADIUS} px from the section's cluster "
                                               "are dropped" if args.site_radius else "off (--no-site-radius)"))

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
            "pred_before_gate": None,
            "gt_aav": len(aav_areas),
            "aav_pred": None,
            "aav_pred_before_gate": None,
            "aav_kind": mask_kind(aav_areas),
            "gfp_kind": mask_kind(gfp_areas),
            "aav_objects": gate.get("objects", {}).get(name),
            "aav_dropped": None,
            "cluster_score": gate.get("score", {}).get(name),
            "in_run": (name in gate["run_set"]) if gate["applied"] and name in gate["score"] else None,
            "centre": gate.get("centre", {}).get(name),
        }
        for t in thresholds:
            totals["gt"][t] += int((shares >= t).sum())

        if has_predictions:
            if name in aav_files and name in gfp_files:
                predicted_aav, predicted_gfp = crop_to_common(read_prediction(aav_files[name]), read_prediction(gfp_files[name]))
                starter, shares, aav_cells, _ = find_starter_cells(predicted_aav, predicted_gfp, args.min_overlap, args.measure)
                row["pred_before_gate"] = int((shares >= args.min_overlap).sum())
                row["aav_pred_before_gate"] = row["aav_pred"] = len(aav_cells)
                gated_aav, row["aav_dropped"] = gate_aav(predicted_aav, name, gate)
                if row["aav_dropped"]:
                    if gated_aav.any():
                        starter, shares, aav_cells, _ = find_starter_cells(gated_aav, predicted_gfp, args.min_overlap,
                                                                           args.measure)
                    else:
                        starter, shares, aav_cells = np.zeros_like(predicted_aav), np.zeros(0), []
                    row["aav_pred"] = len(aav_cells)
                save_mask(starter, os.path.join(pred_dir, f"{name}.mat"))
                row["pred"] = int((shares >= args.min_overlap).sum())
                for t in thresholds:
                    totals["pred"][t] += int((shares >= t).sum())
            else:
                missing_predictions.append(name)
                if name in aav_files:   # no GFP prediction: no starter cells, but the AAV cells can still be counted
                    predicted_aav = read_prediction(aav_files[name])
                    gated_aav, row["aav_dropped"] = gate_aav(predicted_aav, name, gate)
                    row["aav_pred_before_gate"] = count_objects(predicted_aav)
                    row["aav_pred"] = count_objects(gated_aav) if row["aav_dropped"] else row["aav_pred_before_gate"]
        rows.append(row)
        if row["pred"] is None:
            predicted_text = ", predicted n/a (prediction missing)"
        elif row["pred"] != row["pred_before_gate"]:
            predicted_text = f", predicted {row['pred']} ({row['pred_before_gate']} before the gate)"
        else:
            predicted_text = f", predicted {row['pred']}"
        print(
            f"  {number}/{len(sections)} {name}: ground truth {row['gt']}"
            + (predicted_text if has_predictions else "")
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
        ("injection-site gate", gate["status"]),
    ]
    if "peak" in gate:
        settings += [
            ("cluster score", f"predicted AAV pixels in the section's densest disc of radius {CLUSTER_RADIUS} px (centred on "
                              f"an object), in cells' worth: divided by {gate['cell_area']:.0f} px, the median area of "
                              "the brain's predicted AAV objects"),
            ("run", f"from the peak, {gate['peak']} ({gate['peak_score']:.0f} cells' worth; the highest score a neighbouring "
                    f"section also reaches), outward while the score is at least {RUN_MIN_CELLS} cells' worth and "
                    f"{RUN_FRACTION:.0%} of the peak (cutoff {gate['cutoff']:.1f}); up to {RUN_MAX_GAP} section(s) in a "
                    "row below the cutoff are bridged when the next one passes"),
        ]
    if gate["applied"]:
        settings += [
            ("bridged sections", ", ".join(gate["bridged"]) or "none"),
            ("in-section filter", (f"on: inside the run, AAV objects more than {SITE_RADIUS} px from the section's cluster "
                                   f"centre were dropped ({sum(r['aav_dropped'] or 0 for r in rows if r['in_run'])} objects)")
             if args.site_radius else "off (--no-site-radius)"),
        ]
    settings.append(("created", f"{datetime.now():%Y-%m-%d %H:%M}"))
    table_path = os.path.join(out_dir, f"{brain}_starter_cells.xlsx")
    try:
        write_workbook(table_path, rows, thresholds, totals, settings, has_predictions, measure_text, args.min_overlap, gate)
    except ImportError:
        table_path = os.path.join(out_dir, f"{brain}_starter_cells.csv")
        write_csv(table_path, rows, has_predictions, gate)
        print("NOTE: openpyxl is not installed, so the table was saved as CSV; pip install openpyxl for the Excel file")
    rule_text = f"Starter cell: an AAV cell and a GFP cell overlapping by at least {args.min_overlap:.0%} of {measure_text}"
    plot_path = os.path.join(out_dir, f"{brain}_starter_cells_per_section.pdf")
    plot_counts(plot_path, rows, brain, has_predictions, rule_text, gate)
    aav_plot_path = os.path.join(out_dir, f"{brain}_aav_cells_per_section.pdf")
    plot_counts(aav_plot_path, rows, brain, has_predictions, AAV_RULE_TEXT, gate, kind="aav")

    print(f"\nground-truth starter cells: {sum(row['gt'] for row in rows)} in {len(rows)} sections")
    if has_predictions:
        both = [row for row in rows if row["pred"] is not None]
        print(
            f"predicted starter cells:    {sum(row['pred'] for row in both)} in {len(both)} sections "
            f"(ground truth in those sections: {sum(row['gt'] for row in both)})"
        )
        if gate["applied"]:
            print(f"  before the gate:          {sum(row['pred_before_gate'] for row in both)}; "
                  f"in the {len(gate['run'])}-section run: ground truth {sum(r['gt'] for r in both if r['in_run'])}, "
                  f"predicted {sum(r['pred'] for r in both if r['in_run'])}")
        if missing_predictions:
            print(f"sections without both predicted masks ({len(missing_predictions)}): {', '.join(missing_predictions[:10])}"
                  + (" ..." if len(missing_predictions) > 10 else ""))
        if unmatched:
            print(f"predicted masks without a .mat mask, not used ({len(unmatched)}): {', '.join(unmatched[:10])}")
    dotted = [row["section"] for row in rows if "dots" in (row["aav_kind"], row["gfp_kind"])]
    if dotted:
        print(f"{len(dotted)} sections have dot markers in at least one ground-truth mask; see the mask columns of the table")
    print(f"AAV cells: ground truth {sum(row['gt_aav'] for row in rows)}"
          + (f", predicted {sum(row['aav_pred'] or 0 for row in rows)}" if has_predictions else "")
          + (f" ({sum(row['aav_pred_before_gate'] or 0 for row in rows)} before the gate)" if gate["applied"] else ""))
    print(f"starter masks: {gt_dir}" + (f" and {pred_dir}" if has_predictions else ""))
    print(f"table: {table_path}")
    print(f"plots: {plot_path}\n       {aav_plot_path}")


if __name__ == "__main__":
    main()
