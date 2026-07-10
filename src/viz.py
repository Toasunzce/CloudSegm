"""
Submodule for data visualization.
Usage example:

# single scene 
>>> show_images(imgs[0], dates="2024-07-01")
>>> show_masks(preds[0], dates="2024-07-01")

# batch
>>> show_images(imgs, dates=dates)
>>> show_masks(preds, dates=dates)

# axes inserting (single image only)
>>> fig, ax = plt.subplots(1, 2)
    show_images(imgs[0], ax=ax[0])
    show_masks(preds[0], ax=ax[1])
    plt.show()

# general analysis
>>> preds, probs = get_predictions(model, imgs, device)
>>> composite_img, source_map, mask_map = composite(imgs, model, device)

>>> show_predictions(imgs, preds)
>>> show_confidence(imgs, probs)
>>> show_composite_source(composite_img, source_map, mask_map, imgs:
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch


MASK_CMAP   = ListedColormap(["black", "white", "gray", "blue"])
CLASS_NAMES = ["clear", "thick cloud", "thin cloud", "shadow"]



def scene_to_rgb(img):
    """
    Translates (7, H, W) tensor -> (H, W, 3) numpy, clipped to [0, 1]
    """
    return np.clip(img[[2, 1, 0]].permute(1, 2, 0).numpy(), 0, 1)


def show_images(imgs, dates=None, ax=None):
    """
    Show scene(s) in RGB.

    imgs: (7, H, W) - single image
          (B, 7, H, W) - batch of images
    dates: str | list[str] | None
    ax: matplotlib Axes | None - if provided, draw into existing axes (single image only)
    """
    # single image
    if imgs.ndim == 3:
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.imshow(scene_to_rgb(imgs))
        if dates is not None:
            ax.set_title(dates, fontsize=8)
        ax.axis("off")
        if ax is None:
            plt.tight_layout()
            plt.show()
        return
    
    # batch
    N   = imgs.shape[0]
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 4), dpi=150)
    if N == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(scene_to_rgb(imgs[i]))
        if dates is not None:
            ax.set_title(dates[i], fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_masks(masks, dates=None, ax=None):
    """
    Show segmentation mask(s) in 4 classes.

    masks: (H, W) - single mask
           (N, H, W) - batch of masks
    dates: str | list[str] | None
    ax: matplotlib Axes | None - if provided, draw into existing axes (single mask only)
    """
    def _stats(mask):
        total = mask.numel()
        return "\n".join(
            f"{CLASS_NAMES[c]}: {(mask == c).sum().item() / total * 100:.1f}%"
            for c in range(4)
        )

    # single mask
    if masks.ndim == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.imshow(masks, cmap=MASK_CMAP, vmin=0, vmax=3)
        if dates is not None:
            ax.set_title(dates, fontsize=8)
        ax.set_xlabel(_stats(masks), fontsize=6, family="monospace")
        ax.axis("off")
        if ax is None:
            plt.tight_layout()
            plt.show()
        return

    # batch
    N   = masks.shape[0]
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 4), dpi=150)
    if N == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(masks[i], cmap=MASK_CMAP, vmin=0, vmax=3)
        if dates is not None:
            ax.set_title(dates[i], fontsize=8)
        ax.set_xlabel(_stats(masks[i]), fontsize=6, family="monospace")
        ax.axis("off")

    patches = [
        mpatches.Patch(color=MASK_CMAP(c / 3), label=CLASS_NAMES[c])
        for c in range(4)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=8)
    fig.tight_layout()
    plt.show()


def show_predictions(imgs, preds, dates=None):
    """
    Show RGB images and segmentation masks in 2 rows.

    imgs: (N, 7, H, W)
    preds: (N, H, W)
    dates: list[str] | None
    """
    N   = imgs.shape[0]
    fig, ax = plt.subplots(
        3, N,
        figsize=(3.5 * N, 9),
        dpi=150,
        gridspec_kw={"height_ratios": [4, 4, 1]},
    )
    if N == 1:
        ax = ax[:, np.newaxis]

    for i in range(N):
        # RGB
        ax[0, i].imshow(scene_to_rgb(imgs[i]))
        ax[0, i].set_title(dates[i] if dates else f"Scene {i+1}", fontsize=8, pad=4)
        ax[0, i].axis("off")

        # mask
        ax[1, i].imshow(preds[i], cmap=MASK_CMAP, vmin=0, vmax=3)
        ax[1, i].axis("off")

        # stats
        total = preds[i].numel()
        stats = "\n".join(
            f"{CLASS_NAMES[c]}: {(preds[i] == c).sum().item() / total * 100:.1f}%"
            for c in range(4)
        )
        ax[2, i].text(0.5, 0.5, stats, ha="center", va="center",
                      fontsize=7, family="monospace",
                      transform=ax[2, i].transAxes)
        ax[2, i].axis("off")

    ax[0, 0].set_ylabel("RGB", fontsize=10, labelpad=6)
    ax[1, 0].set_ylabel("Mask", fontsize=10, labelpad=6)

    patches = [
        mpatches.Patch(color=MASK_CMAP(c / 3), label=CLASS_NAMES[c])
        for c in range(4)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.01), fontsize=9, framealpha=0.9)
    fig.suptitle("Cloud Segmentation", fontsize=12, y=1.01)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    plt.show()


def show_confidence(imgs, probs, dates=None):
    """
    Show RGB images and per-pixel confidence heatmaps in 2 rows.

    imgs: (N, 7, H, W)
    probs: (N, 4, H, W) - class probabilities from inference.get_predictions method
    dates: list[str] | None
    """
    N   = imgs.shape[0]
    fig, ax = plt.subplots(2, N, figsize=(3.5 * N, 7), dpi=150)
    if N == 1:
        ax = ax[:, np.newaxis]

    im = None
    for i in range(N):
        # RGB
        ax[0, i].imshow(scene_to_rgb(imgs[i]))
        ax[0, i].set_title(dates[i] if dates else f"Scene {i+1}", fontsize=8)
        ax[0, i].axis("off")

        # confidence
        confidence = probs[i].max(dim=0).values.numpy()
        im = ax[1, i].imshow(confidence, cmap="RdYlGn", vmin=0.25, vmax=1.0)
        ax[1, i].axis("off")

    ax[0, 0].set_ylabel("RGB", fontsize=10)
    ax[1, 0].set_ylabel("Confidence", fontsize=10)

    fig.colorbar(im, ax=ax[1, :], shrink=0.6, label="max probability")  # ty:ignore[invalid-argument-type]
    fig.suptitle("Prediction Confidence", fontsize=12, y=1.01)
    fig.tight_layout()
    plt.show()


def show_composite_source(composite_img , source_map, mask_map, imgs, dates=None):
    """
    Show compositing result in 3 columns:
    - composite RGB
    - composite mask
    - source map (which scene each pixel came from)

    composite: (7, H, W)
    final_mask: (H, W)
    source_map: (H, W) - scene index per pixel
    imgs: (N, 7, H, W) - original scenes
    dates: list[str] | None
    """
    N          = imgs.shape[0]
    source_cmap = plt.cm.get_cmap("tab10", N)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # composite RGB
    ax[0].imshow(scene_to_rgb(composite_img))
    ax[0].set_title("Composite", fontsize=10, fontweight="bold")
    ax[0].axis("off")

    # composite mask
    ax[1].imshow(mask_map, cmap=MASK_CMAP, vmin=0, vmax=3)
    ax[1].set_title("Composite Mask", fontsize=10)
    ax[1].axis("off")

    total      = mask_map.numel()
    mask_stats = "\n".join(
        f"{CLASS_NAMES[c]}: {(mask_map == c).sum().item() / total * 100:.1f}%"
        for c in range(4)
    )
    ax[1].set_xlabel(mask_stats, fontsize=7, family="monospace")

    # source map
    im = ax[2].imshow(source_map, cmap=source_cmap, vmin=0, vmax=N - 1)
    ax[2].set_title("Pixel Source Map", fontsize=10)
    ax[2].axis("off")

    source_patches = [
        mpatches.Patch(
            color=source_cmap(i / N),
            label=dates[i] if dates else f"Scene {i+1}"
        )
        for i in range(N)
    ]
    ax[2].legend(handles=source_patches, loc="lower right",
                 fontsize=7, framealpha=0.9)

    # mask legend
    mask_patches = [
        mpatches.Patch(color=MASK_CMAP(c / 3), label=CLASS_NAMES[c])
        for c in range(4)
    ]
    fig.legend(handles=mask_patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=9, framealpha=0.9)

    fig.suptitle("Progressive Composite Result", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()