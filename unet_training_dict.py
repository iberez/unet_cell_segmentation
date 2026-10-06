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

import json
import logging
import math
import os
import sys
import warnings
from datetime import datetime
from glob import glob

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import tifffile
import numpy as np
from scipy.io import loadmat

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
)
from monai.visualize import plot_2d_or_3d_image

RAW_CHANNEL = 0       # which channel of the TIFF to train on
MASK_KEY = "BW_I2"    # which mask in the .mat goes with that channel

# --- patch sampling ---
PATCH_SIZE = (256, 256)
NUM_SAMPLES = 8       # patches drawn from a section each time it comes up in a batch
POS, NEG = 3, 1       # POS / (POS + NEG) = share of patches centred on a cell pixel (the rest: tissue background)

# --- data split: whole slides go to train / val / test in this repeating order (5/7, 1/7, 1/7 of the slides) ---
SPLIT_FILE = "split_by_slide.json"   # written once into the data directory, then reused so the test set never changes
SPLIT_CYCLE = ("train", "train", "train", "val", "train", "train", "test")

# --- run length (one epoch = number of training sections / batch_size steps, about 30 with ~120 sections) ---
NUM_EPOCHS = 100
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
    return image[RAW_CHANNEL][np.newaxis]

def load_mask(mask_path):
    """Load the MASK_KEY mask from a BW_*.mat file as a float32 (1, Y, X) array of 0s and 1s."""
    mask = loadmat(mask_path, variable_names=[MASK_KEY])[MASK_KEY]
    return (mask > 0).astype(np.float32)[np.newaxis]


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


def split_by_slide(images, segs, split_path):
    """Split the paired sections into train / val / test by slide (about 70 / 15 / 15 %).

    All sections cut from one slide go to the same set. The assignment is written to split_path
    the first time and read back on every later run, so the test sections stay the same.
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
        print(f"using the existing split in {split_path}")
    else:
        slide_split = {slide: SPLIT_CYCLE[i % len(SPLIT_CYCLE)] for i, slide in enumerate(slides)}
        with open(split_path, "w") as f:
            json.dump({"note": "slide -> split; delete this file to redo the split", "slides": slide_split}, f, indent=2)
        print(f"new split written to {split_path}")

    files = {"train": [], "val": [], "test": []}
    for image, seg in zip(images, segs):
        files[slide_split[slide_of(image)]].append({"image": image, "label": seg})
    for name, members in files.items():
        in_split = [s for s in slides if slide_split[s] == name]
        print(f"  {name}: {len(members)} sections from {len(in_split)} slides {in_split}")
    return files


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


def save_loss_plot(history, plot_dir):
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

    fig.suptitle(f"raw channel {RAW_CHANNEL}, mask {MASK_KEY}")
    plot_path = os.path.join(plot_dir, f"loss_vs_epochs_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    fig.savefig(plot_path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main(data_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # sections whose mask is empty can only give background patches; MONAI warns about each one, every epoch
    warnings.filterwarnings("ignore", message=".*unable to generate class balanced samples.*")

    print(f"loading images and segmentations from {data_dir} (this may take a while)")
    images = sorted(glob(os.path.join(data_dir, "112_02_single_section_raw_input", "*.tiff")))
    segs = sorted(glob(os.path.join(data_dir, "112_02_single_section_seg_masks", "*.mat")))
    print (f"len images: {len(images)}")
    print (f"len segs: {len(segs)}")

    # pair each image with its own mask by name (the two folders differ in length)
    images, segs = pair_by_name(images, segs)

    # sanity check on one pair: both must be (1, Y, X) with identical Y and X
    first_image, first_mask = load_and_normalize_section(images[0]), load_mask(segs[0])
    print(f"first pair: image {first_image.shape}, mask {first_mask.shape}, mask pixels = {int(first_mask.sum())}")
    assert first_image.shape == first_mask.shape, "image and mask shapes differ"

    # one dictionary per section, so the image and its mask travel through the transforms together;
    # whole slides are assigned to train / val / test, and the assignment is kept in a file
    files = split_by_slide(images, segs, os.path.join(data_dir, SPLIT_FILE))
    train_files, val_files = files["train"], files["val"]  # files["test"] is deliberately not touched in this script

    train_transforms = Compose(
        [
            # deterministic loading: CacheDataset runs these once per section and keeps the result in RAM
            Lambdad(keys="image", func=load_and_normalize_section),
            Lambdad(keys="label", func=load_mask),
            # foreground-aware sampling: NUM_SAMPLES patches per section, each centred either on a
            # cell pixel (label == 1) or on a tissue pixel without a cell (label == 0 and image > 0)
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=POS,
                neg=NEG,
                num_samples=NUM_SAMPLES,
                image_key="image",
                image_threshold=0,
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
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
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

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    # start a typical PyTorch training
    val_interval = VAL_INTERVAL
    best_metric = -1
    best_metric_epoch = -1
    history = {}  # per-epoch training losses, filled by record_epoch()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(NUM_EPOCHS):
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
        # the same two numbers on the fixed validation patches
        val_loss, val_patch_dice = evaluate_patches(model, val_patch_images, val_patch_labels, dice_loss, bce_loss, device)
        record_epoch(history, loss=epoch_loss, val_loss=val_loss, train_patch_dice=train_dice, val_patch_dice=val_patch_dice)
        writer.add_scalar("val_patch_loss", val_loss, epoch + 1)
        writer.add_scalar("val_patch_dice", val_patch_dice, epoch + 1)
        print(
            f"epoch {epoch + 1}/{NUM_EPOCHS}: train loss {epoch_loss:.4f} (dice {epoch_dice_term / step:.4f} + bce {epoch_bce_term / step:.4f}),"
            f" patch dice {train_dice:.4f} | val loss {val_loss:.4f}, patch dice {val_patch_dice:.4f}"
        )

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    roi_size = PATCH_SIZE
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                record_epoch(history, section_dice_epoch=epoch + 1, section_dice=metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "  whole-section validation at epoch {}: mean dice {:.4f} (best {:.4f} at epoch {})".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

    plot_path = save_loss_plot(history, os.path.join(data_dir, PLOT_FOLDER))
    print(f"loss plot saved to {plot_path}")


if __name__ == "__main__":
    data_dir = "/zjbd/zd1/isaac/rabies_cnn"
    main(data_dir)
