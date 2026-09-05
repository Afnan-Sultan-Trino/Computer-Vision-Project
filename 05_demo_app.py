"""
DEMO APP: Upload a skin image -> CNN prediction -> Grad-CAM heatmap ->
ontology-generated clinical report. Self-contained (no external imports
from other project files needed beyond the model + libraries).

Run in Colab with:
    !pip install streamlit pyngrok --quiet
    !streamlit run 05_demo_app.py &>/content/logs.txt &
    (then set up ngrok tunnel -- see chat instructions)

Or run locally with:
    streamlit run 05_demo_app.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from dataclasses import dataclass

# ==================================================================
# ---------------- Config ----------------
# ==================================================================
MODEL_PATH = "arsenicnet_baseline.keras"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "top_conv"

st.set_page_config(page_title="Arsenicosis Screening Demo", layout="centered")
st.title("Ontology-guided arsenicosis screening")
st.caption("Undergraduate research prototype (AIUB) -- not a certified diagnostic tool.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# ==================================================================
# ---------------- CNN + Grad-CAM ----------------
# ==================================================================
def preprocess_for_model(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    raw = np.array(img).astype("float32")
    display_arr = raw / 255.0
    model_input = preprocess_input(raw.copy())
    return display_arr, model_input


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array, axis=0))
        class_channel = predictions[:, 0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(predictions[0][0])


def overlay_gradcam(display_arr, heatmap, alpha=0.4):
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], IMG_SIZE).numpy().squeeze()
    jet = cm.get_cmap("jet")
    jet_heatmap = jet(heatmap_resized)[:, :, :3]
    overlay = jet_heatmap * alpha + display_arr * (1 - alpha)
    return np.clip(overlay, 0, 1)


# ==================================================================
# ---------------- Ontology rule engine ----------------
# ==================================================================
@dataclass
class OntologyRule:
    lesion_pattern: str
    body_location: str
    explanation: str
    risk_note: str


RULES = [
    OntologyRule(
        "diffuse_hyperpigmentation", "trunk",
        "Diffuse spotted hyperpigmentation (melanosis) on the trunk is one of the "
        "earliest cutaneous signs of chronic arsenic exposure.",
        "Associated with long-term ingestion of contaminated groundwater; "
        "recommend confirmatory water-source testing and dermatology referral.",
    ),
    OntologyRule(
        "keratotic_lesion", "palm",
        "Punctate or diffuse keratosis on the palm is a classical marker of "
        "arsenical keratosis, often appearing after years of exposure.",
        "Palmar/plantar keratosis carries elevated risk of progression to "
        "Bowen's disease if untreated; recommend closer monitoring.",
    ),
    OntologyRule(
        "keratotic_lesion", "sole",
        "Keratotic thickening on the sole is consistent with arsenical keratosis, "
        "commonly co-occurring with palmar lesions.",
        "Same progression risk as palmar keratosis; monitor both sites together.",
    ),
    OntologyRule(
        "keratotic_lesion", "trunk",
        "Localized keratotic thickening on trunk/forearm skin is consistent with "
        "early arsenical keratosis.",
        "Recommend monitoring for progression and checking palms/soles for "
        "co-occurring lesions.",
    ),
]


def match_rule(lesion_pattern, body_location):
    for rule in RULES:
        if rule.lesion_pattern == lesion_pattern and rule.body_location == body_location:
            return rule
    return None


def generate_report(cnn_label, cnn_confidence, lesion_pattern, body_location):
    if cnn_label != "Arsenic-affected":
        return (f"Prediction: {cnn_label} (confidence {cnn_confidence:.1%}). "
                "No arsenic-related lesion pattern detected in this image.")
    rule = match_rule(lesion_pattern, body_location)
    if rule is None:
        return (f"Prediction: Arsenic-affected (confidence {cnn_confidence:.1%}). "
                f"Lesion pattern '{lesion_pattern}' at '{body_location}' is not yet "
                "covered by the rule base -- flagged for manual dermatologist review.")
    return (f"**Prediction:** Arsenic-affected (confidence {cnn_confidence:.1%})\n\n"
            f"**Clinical reasoning:** {rule.explanation}\n\n"
            f"**Risk note:** {rule.risk_note}")


# ==================================================================
# ---------------- UI ----------------
# ==================================================================
uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded)
    display_arr, model_input = preprocess_for_model(pil_img)

    heatmap, pred_prob = make_gradcam_heatmap(model_input, model, LAST_CONV_LAYER)
    gradcam_overlay = overlay_gradcam(display_arr, heatmap)
    label = "Arsenic-affected" if pred_prob > 0.5 else "Healthy"

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_arr, caption="Uploaded image", use_container_width=True)
    with col2:
        st.image(gradcam_overlay, caption="Grad-CAM (model attention)", use_container_width=True)

    st.metric("Prediction", label, f"{pred_prob:.1%} confidence")

    st.divider()
    st.subheader("Lesion details")
    st.caption(
        "The CNN and Grad-CAM identify *that* something looks affected and roughly "
        "*where* the model is focusing -- but not yet the specific lesion type or "
        "body region. Select those below based on visual inspection of the image "
        "above (this manual step is a known limitation, documented in our writeup)."
    )
    lesion_pattern = st.selectbox(
        "Lesion pattern",
        ["keratotic_lesion", "diffuse_hyperpigmentation", "leucomelanosis", "none"],
    )
    body_location = st.selectbox(
        "Body location", ["trunk", "palm", "sole", "limbs", "unspecified"]
    )

    st.subheader("Ontology-generated clinical report")
    report = generate_report(label, pred_prob, lesion_pattern, body_location)
    st.info(report)
else:
    st.write("Upload an image to see the prediction, Grad-CAM heatmap, and "
             "ontology-based clinical explanation.")
