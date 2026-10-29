# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the U-Net on single-section images. What a run uses depends on SPLIT_MODE (see the settings below):

  SPLIT_MODE = "slide"     (whole slides; cross-validation by slide)
    python3 unet_training_dict.py --fold 0    # validate on fold 0's slides, train on the other non-test slides
    python3 unet_training_dict.py --final     # train on every non-test slide (no validation)

  SPLIT_MODE = "section"   (sections drawn at random within each slide)
    python3 unet_training_dict.py             # train on the training sections, validate on the validation sections

The test slides or sections are never used here. Each run keeps its last-epoch weights in
<data directory>/models/<run>.pth (fold0 ... fold3, final, or section), next to a <run>.json record of
what it trained and validated on, which unet_evaluation_dict.py reads. To train the four folds at once,
one per GPU:

    for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k nohup python3 unet_training_dict.py --fold $k > fold$k.log 2>&1 & done
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import warnings
import zlib
from datetime import datetime
from glob import glob

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import tifffile
import numpy as np
from scipy.io import loadmat, whosmat

import matplotlib
matplotlib.use("Agg")  # no display on the server: draw straight to file
import matplotlib.pyplot as plt

# embed fonts as TrueType so the text in the PDF stays editable and renders the same everywhere
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import monai
from monai.data import CacheDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Lambdad,
    LoadImage,
    RandCropByPosNegLabeld,
    RandRotated,
    Transform,
)
from monai.visualize import plot_2d_or_3d_image

# --- where things are (unet_evaluation_dict.py reads these too) ---
DATA_DIR = "/zjbd/zd1/isaac/rabies_cnn"
IMAGE_FOLDER = "112_02_single_section_raw_input"
MASK_FOLDER = "112_02_single_section_seg_masks"
MODEL_FOLDER = "models"   # weights of each run (fold0.pth, fold1.pth, ..., final.pth), each with a .json record

RAW_CHANNEL = 0       # which channel of the TIFF to train on
MASK_KEY = "BW_I2"    # which mask in the .mat goes with that channel

# Names of channel/mask pairs. A named pair puts its name in front of every file and folder its runs create
# (models/aav_section.pth, loss_vs_epochs_aav_section_<time>.pdf, evaluation/val_aav_section_<time>/,
# aav_mask_audit.csv, ...), so the outputs of different channels never mix or overwrite each other.
# The pair without a name (here channel 0 with BW_I2) keeps the plain names.
CHANNEL_NAMES = {(1, "BW_I1"): "aav"}

# --- masks that mark cells with fixed-size dots instead of outlines (listed by unet_mask_audit.py) ---
EXCLUDE_DOT_MASKS = True       # leave those sections out of training and validation
AUDIT_FILE = "mask_audit.csv"  # written into the data directory by unet_mask_audit.py

# --- image / mask size check ---
SHAPE_TOLERANCE = 2   # pairs differing by up to this many pixels per side are trimmed to their common size;
                      # set to 0 to drop every pair whose image and mask sizes differ

# file path -> (rows, columns) to trim to; filled by check_pair_shapes() and applied by the two loaders
CROP_TO = {}

# --- hard negatives: places where an earlier model raised false alarms on the training sections ---
HARD_NEGATIVE_FILE = "hard_negatives.csv"  # written into the data directory by: python3 unet_evaluation_dict.py --split train
HARD_NEGATIVE_SHARE = 0.5          # share of the background-centred patches drawn at those places (0 = don't use them)
HARD_NEGATIVE_MIN_DISTANCE = 40    # use only false alarms at least this many pixels from every annotated cell (0 = use all)
BACKGROUND_CENTRES = 50000         # ordinary background patch centres kept per section

# --- patch sampling ---
PATCH_SIZE = (256, 256)
NUM_SAMPLES = 8       # patches drawn from a section each time it comes up in a batch
POS, NEG = 3, 1       # POS / (POS + NEG) = share of patches centred on a cell pixel (the rest: tissue background)

# --- data split ---
# "slide":   whole slides. The test slides are fixed in SPLIT_FILE (written once, then reused, so the test set
#            never changes); the other slides are dealt, in anterior-to-posterior order, over N_FOLDS validation
#            groups, kept in FOLD_FILE. Train with --fold K or --final.
# "section": sections drawn at random within every slide: SECTION_SPLIT_SHARES of each slide's sections go to
#            validation and test, the rest to training, kept in SECTION_SPLIT_FILE. Every set then holds sections
#            from every slide, including the slide split's test slides. Train without --fold or --final.
SPLIT_MODE = "section"
SPLIT_FILE = "split_by_slide.json"
SPLIT_CYCLE = ("train", "train", "train", "val", "train", "train", "test")   # used only to create SPLIT_FILE
FOLD_FILE = "cv_folds.json"
N_FOLDS = 4
SECTION_SPLIT_FILE = "split_by_section.json"
SECTION_SPLIT_SHARES = {"val": 0.15, "test": 0.15}
SECTION_SPLIT_SEED = 0

# --- run length and learning rate ---
# One epoch = number of training sections / batch_size steps. The learning rate falls from LEARNING_RATE to
# MIN_LEARNING_RATE along a cosine over the run, and every run keeps its last-epoch weights: the final model
# has no validation slides to pick a best epoch with, so the folds are trained the same way.
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
VAL_INTERVAL = 5      # whole-section validation every this many epochs (patch validation runs every epoch)

# --- output ---
PLOT_FOLDER = "loss_vs_epochs_plots"   # created inside the data directory

def load_and_normalize_section(tiff_path, channel_axis=None):
    """Min-max normalize each channel of a single-section TIFF to [0, 1] (Falk et al., 2019).

    For every channel c: I_c <- (I_c - min(I_c)) / (max(I_c) - min(I_c)),
    with min and max taken over the whole section.

    Parameters
    ----------
    tiff_path : str or Path
        Section TIFF stored as (Y, X), (C, Y, X) or (Y, X, C).
    channel_axis : int, optional
        Axis holding the channels. If None, the shortest axis of a 3D image is used.

    Returns
    -------
    np.ndarray
        float32 array of shape (1, Y, X): channel RAW_CHANNEL, spanning [0, 1],
        with a leading channel axis because MONAI transforms expect channel-first input.
        A constant channel (max == min) comes back as all zeros.
    """
    raw_image = tifffile.imread(tiff_path)
    # Put channels first: (C, Y, X)
    if raw_image.ndim == 2:
        raw_image = raw_image[np.newaxis]
    elif raw_image.ndim == 3:
        if channel_axis is None:
            channel_axis = int(np.argmin(raw_image.shape))
        raw_image = np.moveaxis(raw_image, channel_axis, 0)
    else:
        raise ValueError(
            f"Expected one 2D section as (Y, X), (C, Y, X) or (Y, X, C); got shape {raw_image.shape}"
        )

    # float32 copy: fractional output at half the memory of float64, safe for in-place math
    image = np.ascontiguousarray(raw_image, dtype=np.float32)

    # Per-channel extrema over the full section (NaN-aware in case of float TIFFs)
    channel_min = np.nanmin(image, axis=(1, 2), keepdims=True)
    channel_max = np.nanmax(image, axis=(1, 2), keepdims=True)
    channel_range = channel_max - channel_min

    # Normalize in place; zero-range channels are already all 0 after the subtraction
    image -= channel_min
    np.divide(image, channel_range, out=image, where=channel_range > 0)
    return trim_to_agreed_shape(image[RAW_CHANNEL][np.newaxis], tiff_path)

def load_mask(mask_path):
    """Load the MASK_KEY mask from a BW_*.mat file as a float32 (1, Y, X) array of 0s and 1s."""
    mask = loadmat(mask_path, variable_names=[MASK_KEY])[MASK_KEY]
    return trim_to_agreed_shape((mask > 0).astype(np.float32)[np.newaxis], mask_path)


def trim_to_agreed_shape(array, path):
    """Trim a (1, Y, X) array to the size agreed for its image/mask pair, if check_pair_shapes() set one."""
    if path in CROP_TO:
        rows, cols = CROP_TO[path]
        array = np.ascontiguousarray(array[:, :rows, :cols])
    return array


def tiff_spatial_shape(tiff_path):
    """(Y, X) of a section TIFF, read from the file header without loading the pixels."""
    with tifffile.TiffFile(tiff_path) as tif:
        shape = list(tif.series[0].shape)
    if len(shape) == 3:  # drop the channel axis, taken to be the shortest one as in load_and_normalize_section
        shape.pop(int(np.argmin(shape)))
    return tuple(shape)


def check_pair_shapes(images, segs):
    """Compare the size of every image with its mask, reading file headers only.

    - same size: kept as is
    - one of the two larger by at most SHAPE_TOLERANCE pixels per side: kept, and both are trimmed
      to the common top-left region when loaded
    - anything else (larger difference, or image larger one way and mask the other, which is what
      a rotated or transposed mask looks like): dropped
    Returns the kept images and masks.
    """
    kept_images, kept_segs, trimmed, dropped = [], [], [], []
    for image, seg in zip(images, segs):
        image_shape = tiff_spatial_shape(image)
        mask_shape = next(tuple(shape) for name, shape, _ in whosmat(seg) if name == MASK_KEY)
        differences = [i - m for i, m in zip(image_shape, mask_shape)]
        report = f"{os.path.basename(image)}: image {image_shape}, mask {mask_shape}"
        if image_shape == mask_shape:
            pass
        elif (
            len(image_shape) == len(mask_shape)
            and max(abs(d) for d in differences) <= SHAPE_TOLERANCE
            and (min(differences) >= 0 or max(differences) <= 0)
        ):
            common = tuple(min(i, m) for i, m in zip(image_shape, mask_shape))
            CROP_TO[image] = CROP_TO[seg] = common
            trimmed.append(f"{report} -> both trimmed to {common}")
        else:
            dropped.append(report)
            continue
        kept_images.append(image)
        kept_segs.append(seg)

    print(f"size check: {len(images)} pairs, {len(trimmed)} trimmed to a common size, {len(dropped)} dropped")
    for line in trimmed:
        print(f"  trimmed  {line}")
    for line in dropped:
        print(f"  DROPPED  {line}")
    return kept_images, kept_segs


def section_key(path):
    """Name shared by a section's image and its mask.

    'S013.tif_section_4_f.tiff' and 'BW_I1_I2_S013.tif_section_4_f_rev.mat'
    both give 'S013.tif_section_4_f'.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    return name.removeprefix("BW_I1_I2_").removesuffix("_rev")


def pair_by_name(images, segs):
    """Keep only images that have a mask, matched by section name rather than list position."""
    mask_for = {section_key(p): p for p in segs}
    image_keys = [section_key(p) for p in images]
    if len(mask_for) != len(segs) or len(set(image_keys)) != len(images):
        raise ValueError("Two files map to the same section name; adjust section_key().")

    pairs = [(p, mask_for[k]) for p, k in zip(images, image_keys) if k in mask_for]
    images_without_mask = [os.path.basename(p) for p, k in zip(images, image_keys) if k not in mask_for]
    masks_without_image = [os.path.basename(mask_for[k]) for k in mask_for if k not in set(image_keys)]
    print(f"matched pairs: {len(pairs)}")
    print(f"images without a mask ({len(images_without_mask)}): {images_without_mask}")
    print(f"masks without an image ({len(masks_without_image)}): {masks_without_image}")
    if not pairs:
        raise ValueError("No image/mask pairs matched by name; adjust section_key().")
    return [p[0] for p in pairs], [p[1] for p in pairs]


def slide_of(path):
    """The scanned slide a section was cut from: 'S013' for 'S013.tif_section_4_f.tiff'."""
    return section_key(path).split(".tif_section_")[0]


def slide_assignment(images, split_path):
    """slide -> "train" / "val" / "test", from split_path; the file is created on first use (SPLIT_CYCLE order).

    All sections cut from one slide go to the same set. The file is read back on every later run, so the
    test slides stay the same.
    """
    slides = sorted({slide_of(p) for p in images})
    if len(slides) == len(images):
        print("WARNING: every section looks like its own slide; check slide_of() against your file names")

    if os.path.exists(split_path):
        with open(split_path) as f:
            slide_split = json.load(f)["slides"]
        unknown = [s for s in slides if s not in slide_split]
        if unknown:
            raise ValueError(
                f"slides {unknown} are not listed in {split_path}; delete that file to redo the split "
                "(note that this changes the test set)"
            )
    else:
        slide_split = {slide: SPLIT_CYCLE[i % len(SPLIT_CYCLE)] for i, slide in enumerate(slides)}
        with open(split_path, "w") as f:
            json.dump({"note": "slide -> split; delete this file to redo the split", "slides": slide_split}, f, indent=2)
        print(f"new split written to {split_path}")
    return {slide: slide_split[slide] for slide in slides}


def split_by_slide(images, segs, split_path):
    """train / val / test lists of {"image", "label"} dicts, by the slide split in split_path.

    This is the original single split; training now uses the folds (load_folds) instead.
    """
    slide_split = slide_assignment(images, split_path)
    files = {"train": [], "val": [], "test": []}
    for image, seg in zip(images, segs):
        files[slide_split[slide_of(image)]].append({"image": image, "label": seg})
    for name, members in files.items():
        in_split = [s for s, part in slide_split.items() if part == name]
        print(f"  {name}: {len(members)} sections from {len(in_split)} slides {in_split}")
    return files


def load_folds(images, data_dir):
    """(test slides, folds): the fixed test slides, and the other slides divided into validation groups.

    The test slides come from SPLIT_FILE. The folds come from FOLD_FILE, created on first use by dealing
    the other slides, in anterior-to-posterior order, round-robin over N_FOLDS groups, so that every group
    spans the whole axis. Both files are reused afterwards.
    """
    assignment = slide_assignment(images, os.path.join(data_dir, SPLIT_FILE))
    test_slides = sorted(s for s, part in assignment.items() if part == "test")
    other_slides = sorted(s for s, part in assignment.items() if part != "test")

    fold_path = os.path.join(data_dir, FOLD_FILE)
    if os.path.exists(fold_path):
        with open(fold_path) as f:
            folds = json.load(f)["folds"]
        if sorted(s for fold in folds for s in fold) != other_slides:
            raise ValueError(
                f"{fold_path} does not list exactly the non-test slides {other_slides}; delete it to redo the folds"
            )
    else:
        folds = [other_slides[k::N_FOLDS] for k in range(N_FOLDS)]
        with open(fold_path, "w") as f:
            json.dump(
                {
                    "note": "validation groups for cross-validation by slide; delete this file to redo them",
                    "test": test_slides,
                    "folds": folds,
                },
                f,
                indent=2,
            )
        print(f"cross-validation folds written to {fold_path}")
    return test_slides, folds


def run_slides(images, data_dir, run):
    """(training, validation, test) slides of a run: "fold0", "fold1", ... or "final"."""
    test_slides, folds = load_folds(images, data_dir)
    non_test = sorted(s for fold in folds for s in fold)
    if run == "final":
        return non_test, [], test_slides
    k = int(run.removeprefix("fold"))
    if not 0 <= k < len(folds):
        raise SystemExit(f"there are {len(folds)} folds, numbered 0 to {len(folds) - 1}; got --fold {k}")
    return [s for s in non_test if s not in folds[k]], list(folds[k]), test_slides


def files_for_slides(images, segs, slides):
    """{"image", "label"} dicts for every section on the given slides."""
    keep = set(slides)
    return [{"image": image, "label": seg} for image, seg in zip(images, segs) if slide_of(image) in keep]


def files_for_sections(images, segs, sections):
    """{"image", "label"} dicts for the given sections (section names as section_key() gives them)."""
    keep = set(sections)
    return [{"image": image, "label": seg} for image, seg in zip(images, segs) if section_key(image) in keep]


def split_by_section(images, segs, split_path):
    """Split the paired sections into train / val / test at random, separately within every slide.

    From each slide, SECTION_SPLIT_SHARES of its sections (rounded) go to validation and test and the rest
    to training, so every set holds sections from every slide. Sections of one slide share staining,
    mounting, imaging and here annotation style, so validation and test then measure performance on
    slides the model has seen. The assignment is written to split_path the first time and read back on
    every later run, so the test sections stay the same.
    """
    keys = [section_key(p) for p in images]
    if os.path.exists(split_path):
        with open(split_path) as f:
            section_split = json.load(f)["sections"]
        unknown = [k for k in keys if k not in section_split]
        if unknown:
            raise ValueError(
                f"{len(unknown)} sections (e.g. {unknown[0]}) are not listed in {split_path}; delete that file "
                "to redo the split (note that this changes the test set)"
            )
        print(f"using the existing split in {split_path}")
    else:
        rng = np.random.default_rng(SECTION_SPLIT_SEED)
        section_split = {}
        for slide in sorted({slide_of(p) for p in images}):
            members = sorted(k for k, p in zip(keys, images) if slide_of(p) == slide)
            n_val = int(len(members) * SECTION_SPLIT_SHARES["val"] + 0.5)
            n_test = int(len(members) * SECTION_SPLIT_SHARES["test"] + 0.5)
            for rank, i in enumerate(rng.permutation(len(members))):
                section_split[members[i]] = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"
        with open(split_path, "w") as f:
            json.dump(
                {
                    "note": "section -> split, drawn at random within each slide; delete this file to redo the split",
                    "seed": SECTION_SPLIT_SEED,
                    "sections": section_split,
                },
                f,
                indent=2,
            )
        print(f"new split written to {split_path}")

    files = {"train": [], "val": [], "test": []}
    for image, seg, key in zip(images, segs, keys):
        files[section_split[key]].append({"image": image, "label": seg})
    for name, members in files.items():
        slides = sorted({slide_of(item["image"]) for item in members})
        print(f"  {name}: {len(members)} sections from {len(slides)} slides")
    return files


def run_files(images, segs, data_dir, run):
    """Training, validation and test sections of a run, and the record of them to save with its weights.

    Runs "fold0", "fold1", ... and "final" use the slide split; run "section" uses the section split.
    """
    if run == "section":
        print(f"run {run}: sections drawn at random within each slide")
        files = split_by_section(images, segs, os.path.join(data_dir, SECTION_SPLIT_FILE))
        record = {"split_mode": "section"}
        for name in ("train", "val", "test"):
            record[f"{name}_sections"] = sorted(section_key(item["image"]) for item in files[name])
        return files["train"], files["val"], files["test"], record

    train_slides, val_slides, test_slides = run_slides(images, data_dir, run)
    print(f"run {run}:")
    print(f"  training slides ({len(train_slides)}): {', '.join(train_slides)}")
    print(f"  validation slides ({len(val_slides)}): {', '.join(val_slides) if val_slides else 'none (final model)'}")
    print(f"  test slides, not used here ({len(test_slides)}): {', '.join(test_slides)}")
    record = {"split_mode": "slide", "train_slides": train_slides, "val_slides": val_slides, "test_slides": test_slides}
    return (
        files_for_slides(images, segs, train_slides),
        files_for_slides(images, segs, val_slides),
        files_for_slides(images, segs, test_slides),
        record,
    )


def channel_name():
    """The name of the current RAW_CHANNEL / MASK_KEY pair in CHANNEL_NAMES, or "" if it has none."""
    return CHANNEL_NAMES.get((RAW_CHANNEL, MASK_KEY), "")


def tagged(name):
    """name with the channel name in front ("aav_section.pth"), or unchanged when the channel has no name."""
    return f"{channel_name()}_{name}" if channel_name() else name


def model_path(data_dir, run):
    """Where the weights of a run ("fold0", ..., "final", "section") are kept, for the current channel."""
    return os.path.join(data_dir, MODEL_FOLDER, tagged(f"{run}.pth"))


def model_info_path(weights_path):
    """The small JSON file saved next to a weights file, recording how that model was trained."""
    return os.path.splitext(weights_path)[0] + ".json"


def evaluate_patches(model, images, labels, dice_loss, bce_loss, device, batch_size=32):
    """Loss and Dice on a fixed set of patches, computed the same way as the training numbers."""
    model.eval()
    loss_sum = 0.0
    n_batches = true_positive = predicted_pixels = cell_pixels = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            inputs = images[start : start + batch_size].to(device)
            targets = labels[start : start + batch_size].to(device)
            outputs = model(inputs)
            loss_sum += (dice_loss(outputs, targets) + bce_loss(outputs, targets)).item()
            n_batches += 1
            predicted, truth = outputs > 0, targets > 0.5  # a logit above 0 is a probability above 0.5
            true_positive += (predicted & truth).sum().item()
            predicted_pixels += predicted.sum().item()
            cell_pixels += truth.sum().item()
    return loss_sum / n_batches, 2 * true_positive / max(predicted_pixels + cell_pixels, 1)


def record_epoch(history, **values):
    """Store one epoch's numbers, e.g. record_epoch(history, loss=0.31, val_loss=0.42)."""
    for name, value in values.items():
        history.setdefault(name, []).append(float(value))


def save_loss_plot(history, plot_dir, run):
    """Plot loss (left) and Dice (right) against epoch; save as a PDF in plot_dir and return the file path."""
    os.makedirs(plot_dir, exist_ok=True)
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # font embedding is chatty at INFO level
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(14, 5))

    # left: training and validation loss, both on patches drawn with the same sampling
    ax_loss.plot(epochs, history["loss"], color="black", linewidth=2, label="training (mean over the epoch)")
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], color="tab:red", linewidth=2, label="validation (fixed patches)")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (Dice + cross-entropy)")
    ax_loss.set_title("Loss vs. epoch")
    ax_loss.set_ylim(bottom=0)
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    # right: the same comparison as Dice, plus the whole-section validation score
    if "train_patch_dice" in history:
        ax_dice.plot(epochs, history["train_patch_dice"], color="black", linewidth=2, label="training patches")
    if "val_patch_dice" in history:
        ax_dice.plot(epochs, history["val_patch_dice"], color="tab:red", linewidth=2, label="validation patches")
    if "section_dice" in history:
        ax_dice.plot(
            history["section_dice_epoch"], history["section_dice"],
            color="tab:blue", marker="o", linestyle="--", label="whole validation sections (mean)",
        )
    ax_dice.set_xlabel("epoch")
    ax_dice.set_ylabel("Dice of thresholded predictions")
    ax_dice.set_title("Dice vs. epoch")
    ax_dice.set_ylim(0, 1)
    ax_dice.grid(alpha=0.3)
    ax_dice.legend(loc="lower right")

    fig.suptitle(f"{tagged(run)}: raw channel {RAW_CHANNEL}, mask {MASK_KEY}")
    plot_path = os.path.join(plot_dir, f"loss_vs_epochs_{tagged(run)}_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    fig.savefig(plot_path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def drop_dot_sections(files, data_dir, split_name):
    """Remove the sections whose mask marks cells with dots, according to the audit file."""
    audit_path = os.path.join(data_dir, tagged(AUDIT_FILE))
    if not os.path.exists(audit_path):
        raise SystemExit(
            f"{audit_path} not found: run unet_mask_audit.py first, with the same RAW_CHANNEL and MASK_KEY "
            "(it lists which masks are dots)"
        )
    with open(audit_path, newline="") as f:
        kind_of = {row["section"]: row["annotation"] for row in csv.DictReader(f)}
    unknown = [section_key(item["image"]) for item in files if section_key(item["image"]) not in kind_of]
    if unknown:
        raise SystemExit(f"{len(unknown)} sections are not in {audit_path} (e.g. {unknown[0]}): run unet_mask_audit.py again")
    kept = [item for item in files if kind_of[section_key(item["image"])] != "dots"]
    print(f"  {split_name}: {len(files) - len(kept)} dot-annotated sections left out, {len(kept)} kept")
    return kept


def load_hard_negatives(data_dir):
    """section name -> array of (row, column) false-alarm positions, or None if there is nothing to use.

    Reads the file written by `python3 unet_evaluation_dict.py --split train`, keeping the false alarms
    that lie at least HARD_NEGATIVE_MIN_DISTANCE pixels from every annotated cell.
    """
    path = os.path.join(data_dir, tagged(HARD_NEGATIVE_FILE))
    if HARD_NEGATIVE_SHARE <= 0 or not os.path.exists(path):
        return None
    positions = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["distance_to_cell_px"]) >= HARD_NEGATIVE_MIN_DISTANCE:
                positions.setdefault(row["section"], []).append((int(row["row"]), int(row["col"])))
    return {section: np.array(points, dtype=np.int64) for section, points in positions.items()}


def sampling_centres(label, image, hard_positions, seed):
    """Flat pixel indices from which patch centres are drawn: (cell pixels, background pixels).

    label, image: 2D arrays (Y, X). The background list holds up to BACKGROUND_CENTRES random tissue
    pixels without a cell. If hard_positions (N x 2 array of row, column) is given, those places are
    added to it, repeated often enough to make up HARD_NEGATIVE_SHARE of the list.
    """
    rng = np.random.default_rng(seed)
    cells = np.flatnonzero(label)
    background = np.flatnonzero((image > 0) & ~label)
    if len(background) > BACKGROUND_CENTRES:
        background = rng.choice(background, size=BACKGROUND_CENTRES, replace=False)

    if hard_positions is not None and len(hard_positions) and len(background):
        rows, cols = hard_positions[:, 0], hard_positions[:, 1]
        inside = (rows >= 0) & (rows < label.shape[0]) & (cols >= 0) & (cols < label.shape[1])
        hard = rows[inside] * label.shape[1] + cols[inside]
        hard = hard[~label.ravel()[hard]]  # a "negative" patch is never centred on an annotated cell
        if len(hard):
            share = min(HARD_NEGATIVE_SHARE, 0.95)
            n_hard = int(round(len(background) * share / (1 - share)))
            background = np.concatenate([background, rng.choice(hard, size=n_hard, replace=True)])
    return cells, background


class AddSamplingCentres(Transform):
    """Work out the patch-centre lists for one section and store them for RandCropByPosNegLabeld.

    Not random, so CacheDataset runs it once per section. Uses and removes the "hard_negatives" entry.
    """

    def __call__(self, data):
        d = dict(data)
        to_numpy = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        label, image = to_numpy(d["label"])[0] > 0.5, to_numpy(d["image"])[0]
        d["cell_centres"], d["background_centres"] = sampling_centres(
            label, image, d.pop("hard_negatives", None), seed=zlib.crc32(str(label.shape).encode()) + int(label.sum())
        )
        return d


def list_pairs(data_dir):
    """All usable image/mask pairs in data_dir: matched by name, then checked for size."""
    images = sorted(glob(os.path.join(data_dir, IMAGE_FOLDER, "*.tiff")))
    segs = sorted(glob(os.path.join(data_dir, MASK_FOLDER, "*.mat")))
    print (f"len images: {len(images)}")
    print (f"len segs: {len(segs)}")

    # pair each image with its own mask by name (the two folders differ in length)
    images, segs = pair_by_name(images, segs)

    # every image must be the same size as its mask (checked from the file headers, for all pairs)
    return check_pair_shapes(images, segs)


def build_model():
    """The network. unet_evaluation_dict.py builds the same one before loading the trained weights."""
    return monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def main(data_dir, run):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # sections whose mask is empty can only give background patches; MONAI warns about each one, every epoch
    warnings.filterwarnings("ignore", message=".*unable to generate class balanced samples.*")

    print(
        f"channel: raw channel {RAW_CHANNEL}, mask {MASK_KEY}"
        + (f' ("{channel_name()}": output names start with "{channel_name()}_")' if channel_name() else "")
    )
    print(f"loading images and segmentations from {data_dir} (this may take a while)")
    images, segs = list_pairs(data_dir)

    # sanity check on one pair: both must be (1, Y, X) with identical Y and X
    first_image, first_mask = load_and_normalize_section(images[0]), load_mask(segs[0])
    print(f"first pair: image {first_image.shape}, mask {first_mask.shape}, mask pixels = {int(first_mask.sum())}")
    assert first_image.shape == first_mask.shape, "image and mask shapes differ"

    # the sections this run trains and validates on (one dictionary per section, so the image and its mask
    # travel through the transforms together); the test sections are never used in this script
    train_files, val_files, _, record = run_files(images, segs, data_dir, run)
    has_validation = len(val_files) > 0
    if EXCLUDE_DOT_MASKS:
        train_files = drop_dot_sections(train_files, data_dir, "train")
        if val_files:
            val_files = drop_dot_sections(val_files, data_dir, "val")
    validate = len(val_files) > 0
    if has_validation and not validate:
        print("WARNING: no validation sections are left after leaving out the dot-annotated ones")

    # hard negatives: false alarms of an earlier model on these sections, if they have been collected
    hard_negatives = load_hard_negatives(data_dir)
    if hard_negatives is None:
        print(f"hard negatives: none in use (no {tagged(HARD_NEGATIVE_FILE)}, or HARD_NEGATIVE_SHARE is 0)")
        centre_step, centre_keys = [], {"image_key": "image", "image_threshold": 0}
    else:
        n_places = n_sections = 0
        for item in train_files:
            item["hard_negatives"] = hard_negatives.get(section_key(item["image"]), np.zeros((0, 2), dtype=np.int64))
            n_places += len(item["hard_negatives"])
            n_sections += len(item["hard_negatives"]) > 0
        print(
            f"hard negatives: {n_places} false alarms at least {HARD_NEGATIVE_MIN_DISTANCE} px from any annotated cell, "
            f"in {n_sections} of {len(train_files)} training sections; in those sections "
            f"{100 * HARD_NEGATIVE_SHARE:.0f}% of the background-centred patches are drawn there"
        )
        centre_step = [AddSamplingCentres()]
        centre_keys = {"fg_indices_key": "cell_centres", "bg_indices_key": "background_centres"}

    train_transforms = Compose(
        [
            # deterministic loading: CacheDataset runs these once per section and keeps the result in RAM
            Lambdad(keys="image", func=load_and_normalize_section),
            Lambdad(keys="label", func=load_mask),
            *centre_step,
            # foreground-aware sampling: NUM_SAMPLES patches per section, each centred either on a
            # cell pixel (label == 1) or on a tissue pixel without a cell (label == 0 and image > 0);
            # with hard negatives, the centres come from the lists prepared by AddSamplingCentres
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=POS,
                neg=NEG,
                num_samples=NUM_SAMPLES,
                **centre_keys,
            ),
            # same random angle (within +/- 0.4 rad) for image and mask; "nearest" keeps the mask 0/1
            RandRotated(keys=["image", "label"], prob=0.5, range_x=0.4, mode=("bilinear", "nearest")),
        ]
    )
    val_transforms = Compose(
        [
            Lambdad(keys="image", func=load_and_normalize_section),
            Lambdad(keys="label", func=load_mask),
        ]
    )

    # training data: each batch is batch_size sections x NUM_SAMPLES patches
    print(f"caching {len(train_files)} training and {len(val_files)} validation sections in memory...")
    train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0, num_workers=8, progress=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,  # keep the workers alive between epochs
        drop_last=True,  # no small leftover batch: Dice over the batch needs a decent mix of patches
    )
    if validate:
        # validation data: whole sections, one at a time
        val_ds = CacheDataset(val_files, val_transforms, cache_rate=1.0, num_workers=8, progress=False)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())

        # A fixed set of validation patches, drawn once with exactly the sampling used for training.
        # Loss and Dice on these are directly comparable with the training numbers; the whole-section
        # validation further down is a different, much harder measurement.
        val_sampler = RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=POS,
            neg=NEG,
            num_samples=NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
        )
        val_sampler.set_random_state(seed=0)
        val_patches = [patch for i in range(len(val_ds)) for patch in val_sampler(val_ds[i])]
        # shuffle once so that every batch mixes patches from many sections, as the training batches do
        order = torch.randperm(len(val_patches), generator=torch.Generator().manual_seed(0)).tolist()
        val_patch_images = torch.stack([val_patches[i]["image"].as_subclass(torch.Tensor) for i in order])
        val_patch_labels = torch.stack([val_patches[i]["label"].as_subclass(torch.Tensor) for i in order])
        print(f"validation patches: {len(val_patch_images)} fixed patches from {len(val_ds)} sections")

    # sampling check: how many training patches actually contain a cell?
    sections_with_cells = sum(bool(load_mask(f["label"]).any()) for f in train_files)
    print(f"sampling check: {sections_with_cells} of {len(train_files)} training sections have at least one cell pixel")
    n_patches = n_with_cell = 0
    n_cell_pixels = n_pixels = 0
    for check_batch in train_loader:
        has_cell = check_batch["label"].flatten(1).sum(dim=1) > 0
        n_patches += len(has_cell)
        n_with_cell += int(has_cell.sum())
        n_cell_pixels += check_batch["label"].sum().item()
        n_pixels += check_batch["label"].numel()
    cell_fraction = max(n_cell_pixels / n_pixels, 1e-6)
    print(
        f"sampling check: batches are image {tuple(check_batch['image'].shape)}, label {tuple(check_batch['label'].shape)}; "
        f"{n_with_cell} of {n_patches} patches in one epoch contain a cell"
    )
    print(f"sampling check: {100 * cell_fraction:.3f}% of the pixels in those patches are cell pixels")

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    print(f"CUDA available: {torch.cuda.is_available()}, GPUs visible: {torch.cuda.device_count()}")
    print(f"model is on: {next(model.parameters()).device}")

    # Start from the right prior. A fresh network outputs probability ~0.5 for every pixel, but well
    # under 1% of pixels are cells, so set the bias of the output convolution to the log-odds of the
    # cell-pixel share measured above (the "prior probability" initialisation of the focal-loss paper).
    output_conv = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    prior_bias = math.log(cell_fraction / (1 - cell_fraction))
    torch.nn.init.constant_(output_conv.bias, prior_bias)
    with torch.no_grad():
        start_probability = torch.sigmoid(model(check_batch["image"].to(device))).mean().item()
    print(
        f"output bias set to {prior_bias:.2f}: mean predicted cell probability is now {start_probability:.4f} "
        f"(cell-pixel share of the patches: {cell_fraction:.4f})"
    )
    if start_probability > 10 * cell_fraction:
        print("WARNING: the bias initialisation did not take effect as expected; check which layer output_conv is")

    # Dice over the whole batch (batch=True), so patches without a cell don't each count as a
    # total miss, plus cross-entropy, which gives every pixel a useful gradient from the start
    dice_loss = monai.losses.DiceLoss(sigmoid=True, batch=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    # cosine decay from LEARNING_RATE to MIN_LEARNING_RATE over the run, one step per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LEARNING_RATE)

    history = {}       # per-epoch numbers, filled by record_epoch()
    section_dice = []  # (epoch, mean whole-section validation Dice)
    writer = SummaryWriter(comment=f"_{tagged(run)}")  # TensorBoard folder: runs/<date>_<host>_<channel>_<run>
    for epoch in range(NUM_EPOCHS):
        learning_rate = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_loss = epoch_dice_term = epoch_bce_term = 0.0
        true_positive = predicted_pixels = cell_pixels = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            dice_term = dice_loss(outputs, labels)
            bce_term = bce_loss(outputs, labels)
            loss = dice_term + bce_term
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_dice_term += dice_term.item()
            epoch_bce_term += bce_term.item()
            with torch.no_grad():
                predicted = outputs > 0  # a logit above 0 is a probability above 0.5
                truth = labels > 0.5
                true_positive += (predicted & truth).sum().item()
                predicted_pixels += predicted.sum().item()
                cell_pixels += truth.sum().item()
            epoch_len = len(train_ds) // train_loader.batch_size
            global_step = epoch_len * epoch + step
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.add_scalar("train_dice_term", dice_term.item(), global_step)
            writer.add_scalar("train_bce_term", bce_term.item(), global_step)
        epoch_loss /= step
        # Dice of the thresholded predictions on this epoch's training patches
        train_dice = 2 * true_positive / max(predicted_pixels + cell_pixels, 1)
        line = (
            f"epoch {epoch + 1}/{NUM_EPOCHS} (lr {learning_rate:.1e}): train loss {epoch_loss:.4f} "
            f"(dice {epoch_dice_term / step:.4f} + bce {epoch_bce_term / step:.4f}), patch dice {train_dice:.4f}"
        )
        if validate:
            # the same two numbers on the fixed validation patches
            val_loss, val_patch_dice = evaluate_patches(model, val_patch_images, val_patch_labels, dice_loss, bce_loss, device)
            record_epoch(history, loss=epoch_loss, val_loss=val_loss, train_patch_dice=train_dice, val_patch_dice=val_patch_dice)
            writer.add_scalar("val_patch_loss", val_loss, epoch + 1)
            writer.add_scalar("val_patch_dice", val_patch_dice, epoch + 1)
            line += f" | val loss {val_loss:.4f}, patch dice {val_patch_dice:.4f}"
        else:
            record_epoch(history, loss=epoch_loss, train_patch_dice=train_dice)
        print(line)
        scheduler.step()

        if validate and ((epoch + 1) % VAL_INTERVAL == 0 or epoch + 1 == NUM_EPOCHS):
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = sliding_window_inference(val_images, PATCH_SIZE, 4, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
            section_dice.append((epoch + 1, metric))
            record_epoch(history, section_dice_epoch=epoch + 1, section_dice=metric)
            print(f"  whole-section validation at epoch {epoch + 1}: mean dice {metric:.4f}")
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            # the last validation section's image, label and output, for TensorBoard
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
    writer.close()

    # keep the last-epoch weights, with a record of the slides this run trained and validated on
    weights_path = model_path(data_dir, run)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    with open(model_info_path(weights_path), "w") as f:
        json.dump(
            {
                "run": run,
                "channel": channel_name(),
                "raw_channel": RAW_CHANNEL,
                "mask_key": MASK_KEY,
                **record,
                "epochs": NUM_EPOCHS,
                "exclude_dot_masks": EXCLUDE_DOT_MASKS,
                "hard_negative_share": HARD_NEGATIVE_SHARE if hard_negatives is not None else 0,
                "whole_section_val_dice": {str(e): round(float(m), 4) for e, m in section_dice},
                "saved": f"{datetime.now():%Y-%m-%d %H:%M}",
            },
            f,
            indent=2,
        )
    print(f"training finished; last-epoch weights saved to {weights_path}")
    if section_dice:
        best_epoch, best = max(section_dice, key=lambda pair: pair[1])
        print(
            f"whole-section validation dice at the end: {section_dice[-1][1]:.4f} "
            f"(highest during the run: {best:.4f}, at epoch {best_epoch})"
        )

    plot_path = save_loss_plot(history, os.path.join(data_dir, PLOT_FOLDER), run)
    print(f"loss plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the U-Net (see SPLIT_MODE at the top of this file).")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--fold", type=int, help='SPLIT_MODE "slide": validate on this fold\'s slides, train on the other non-test slides')
    which.add_argument("--final", action="store_true", help='SPLIT_MODE "slide": train on every non-test slide, without validation')
    args = parser.parse_args()
    if SPLIT_MODE == "section":
        if args.fold is not None or args.final:
            parser.error('SPLIT_MODE is "section": run without --fold or --final (those belong to SPLIT_MODE "slide")')
        run = "section"
    elif SPLIT_MODE == "slide":
        if args.fold is None and not args.final:
            parser.error('SPLIT_MODE is "slide": choose --fold K (0 to N_FOLDS - 1) or --final')
        run = "final" if args.final else f"fold{args.fold}"
    else:
        parser.error(f'SPLIT_MODE must be "slide" or "section", not {SPLIT_MODE!r}')
    main(DATA_DIR, run)
