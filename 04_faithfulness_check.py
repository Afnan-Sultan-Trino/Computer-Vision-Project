"""
DAY 6 (the new small contribution): Quantify whether Grad-CAM/LIME
explanations actually point at the lesion, instead of just eyeballing
sample images. This is what makes your poster stand out with real rigor.

Approach for 8/9-day timeline (no time to get a clinician to draw exact
lesion masks for hundreds of images):
1. Take ~15-20 test images.
2. Draw a rough bounding box around the visible lesion yourself using
   the helper below (one click-drag per image) -- this acts as a cheap
   "pseudo ground-truth" region.
3. Compute overlap between that box and the Grad-CAM heatmap's high-
   activation region (IoU-style score).
4. Report the average faithfulness score on the poster as a first-pass
   quantitative check, and note it as a pilot for future clinician-
   validated evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ---------------- Step 1: quick manual box annotation tool ----------------
_boxes = {}


def annotate_box(img_array, img_id):
    """
    Opens a window; drag a rectangle around the lesion, then close the
    window. Box is stored in _boxes[img_id] as (x1, y1, x2, y2).
    Run this once per sample image (repeat for ~15-20 images).
    """
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    ax.set_title(f"Drag a box around the lesion: {img_id}")

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        _boxes[img_id] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        print(f"Saved box for {img_id}: {_boxes[img_id]}")

    selector = RectangleSelector(ax, onselect, useblit=True, button=[1])
    plt.show()
    return _boxes.get(img_id)


# ---------------- Step 2: overlap score between box and heatmap ----------------
def heatmap_to_binary_mask(heatmap, img_size, threshold=0.5):
    """Resize heatmap to image size and threshold to a binary 'attention region'."""
    import tensorflow as tf
    resized = tf.image.resize(heatmap[..., np.newaxis], img_size).numpy().squeeze()
    return (resized >= threshold).astype(np.uint8)


def box_to_mask(box, img_size):
    mask = np.zeros(img_size, dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1
    return mask


def iou_score(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def faithfulness_score(gradcam_heatmap, lesion_box, img_size=(224, 224), threshold=0.5):
    """
    Returns an IoU-style faithfulness score in [0, 1]:
    higher = Grad-CAM's high-activation region overlaps more with the
    (roughly annotated) lesion location.
    """
    heat_mask = heatmap_to_binary_mask(gradcam_heatmap, img_size, threshold)
    box_mask = box_to_mask(lesion_box, img_size)
    return iou_score(heat_mask, box_mask)


# ---------------- Step 3: run across your ~15-20 annotated samples ----------------
def summarize_faithfulness(scores: list[float]):
    scores = np.array(scores)
    print(f"n = {len(scores)}")
    print(f"Mean faithfulness (IoU): {scores.mean():.3f}")
    print(f"Median: {np.median(scores):.3f}  |  Min: {scores.min():.3f}  |  Max: {scores.max():.3f}")
    return {
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "n": len(scores),
    }


if __name__ == "__main__":
    # Example usage once you have heatmaps (from 02_explainability.py)
    # and boxes (from annotate_box) for each sample:
    #
    # heatmap = np.load("gradcam_heatmap.npy")
    # box = _boxes["sample_001"]
    # score = faithfulness_score(heatmap, box)
    #
    # Collect scores for all annotated samples into a list, then:
    # summarize_faithfulness(all_scores)
    print("See comments for usage — run interactively in Jupyter/Colab.")
