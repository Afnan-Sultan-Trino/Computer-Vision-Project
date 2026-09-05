"""
DAY 4: Grad-CAM + LIME explanations for the trained model.
Produces heatmaps you can save/show in the demo and poster.

pip install lime --break-system-packages   # if not already installed
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image as keras_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

MODEL_PATH = "arsenicnet_baseline.keras"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

# EfficientNetB0's last conv block name — adjust if you change the backbone
LAST_CONV_LAYER = "top_conv"


def load_img_array(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return arr


# ---------------- Grad-CAM ----------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array, axis=0))
        class_channel = predictions[:, 0]  # binary sigmoid output

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(predictions[0][0])


def overlay_gradcam(img_array, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], IMG_SIZE
    ).numpy().squeeze()
    jet = cm.get_cmap("jet")
    jet_heatmap = jet(heatmap_resized)[:, :, :3]
    overlay = jet_heatmap * alpha + img_array * (1 - alpha)
    return overlay


# ---------------- LIME ----------------
def predict_fn(images_batch):
    images_batch = images_batch.astype("float32") / 255.0 if images_batch.max() > 1 else images_batch
    preds = model.predict(images_batch, verbose=0)
    return np.hstack([1 - preds, preds])  # [P(healthy), P(arsenic)] for LIME


def run_lime(img_array):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        (img_array * 255).astype("uint8"),
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=500,   # lower for speed; raise to 1000+ if time allows
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=6, hide_rest=False
    )
    return mark_boundaries(temp / 255.0, mask), mask


# ---------------- Run on one example ----------------
if __name__ == "__main__":
    img_path = "data/test/arsenic/example.jpg"  # change to a real test image
    img_array = load_img_array(img_path)

    heatmap, pred_prob = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
    gradcam_overlay = overlay_gradcam(img_array, heatmap)
    lime_overlay, lime_mask = run_lime(img_array)

    label = "Arsenic-affected" if pred_prob > 0.5 else "Healthy"
    print(f"Prediction: {label} (confidence {pred_prob:.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(gradcam_overlay); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(lime_overlay); axes[2].set_title("LIME"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("explanation_example.png", dpi=150)
    plt.show()

    # heatmap and lime_mask are reused in 03_ontology_rules.py and
    # 04_faithfulness_check.py — save them so later scripts don't retrain
    np.save("gradcam_heatmap.npy", heatmap)
    np.save("lime_mask.npy", lime_mask)
