# Explainable AI for Arsenicosis Screening

## 📖 Overview
This repository contains an end-to-end, explainable AI pipeline for screening arsenicosis. It moves beyond traditional "black-box" binary classifications by combining a Convolutional Neural Network (CNN) with Explainable AI (XAI) techniques and ontology-guided reasoning. The final output is a clinically grounded report that provides evidence, explanation, and actionable insights for healthcare professionals.

## 🚨 Problem Statement
Chronic arsenic exposure through contaminated groundwater affects 20–40 million people in Bangladesh, initially appearing as skin lesions before progressing toward cancer. Existing CNN screening tools output only a binary affected/healthy label, offering no clinically interpretable reasoning behind the prediction. 

## 💡 Our Solution
Our pipeline links three components to turn a binary label into an explainable, clinically-grounded output:
1. **CNN Classification:** Detects whether the skin is affected or healthy.
2. **Explainability (Grad-CAM & LIME):** Identifies what lesion pattern is present and where it is located.
3. **Ontology-Guided Reasoning:** Links detected lesion types and locations to clinical risks using a literature-grounded OWL ontology.

## ⚙️ Pipeline Workflow
1. **Input:** ArsenicSkinImageBD Dataset
2. **CNN Classification:** EfficientNet-B0 CNN (Transfer Learning) generates a probability score (0-1).
3. **Heatmap Generation:** Grad-CAM + LIME generate visual explanations of the prediction.
4. **Validation & Reasoning:**
   * *Faithfulness Check:* IoU (Intersection over Union) compares Grad-CAM high-activation regions against manually annotated lesion boxes.
   * *Outlier Reasoning:* Perturbations are applied to identify reasons for misclassifications or attention drift.
5. **Output:** Clinical Report (Evidence + Explanation + Action) validated via Protégé/HermiT DL Queries.

## 🛠️ Tech Stack
*   **Deep Learning Model:** EfficientNetB0 (Frozen backbone, ImageNet weights)
*   **Explainable AI (XAI):** Grad-CAM, LIME
*   **Ontology & Reasoning:** OWL (Web Ontology Language), Protégé, HermiT Reasoner
*   **Evaluation Metrics:** IoU (Intersection over Union), Accuracy, AUC, Precision, Recall, F1-Score

## 📊 Dataset
*   **Source:** ArsenicSkinImageBD (Emu et al., 2023) - 4 field sites in Chapainawabganj, Bangladesh.
*   **Classes:** 2 (Arsenic-Affected, Healthy)
*   **Data Split:** 
    *   Original Images: 1,482 (741 + 741)
    *   Images Used: 1,287 (741 affected, 546 healthy)
    *   Train/Validation Split: 85% / 15% (1,093 / 194 images)

## 🧠 Model Training Hyperparameters
| Hyperparameter | Value |
| :--- | :--- |
| Input resolution | 224 × 224 × 3 |
| Backbone | EfficientNetB0 (frozen, ImageNet) |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss function | Binary Cross-Entropy |
| Dropout | 0.4 |
| Epochs | 12 (with early stopping) |

## 📈 Results
*   **Validation Accuracy:** 90.7%
*   **Validation AUC:** 0.984
*   **Precision:** 0.918
*   **Recall:** 0.878
*   **F1-Score:** 0.935
*   **Grad-CAM Faithfulness (IoU):** Mean IoU = 0.144 (Best overlap: 0.502 on high-confidence cases).
*   **Ontology Validation:** DL Query `'Trunk Keratosis'` successfully auto-classifies real detected lesions (IMG_1176, IMG_1184) via HermiT reasoning.

## ⚠️ Limitations & Future Work
*   **Current Limitation:** Lesion type and location are currently identified manually from Grad-CAM output to feed into the ontology. 
*   **Future Work:** Automating the extraction of lesion type/location from the Grad-CAM heatmaps to create a fully automated clinical reporting system.

## 📚 References
1. Sil et al., QJM (Oxford Academic), 2020 - WHO arsenicosis definition
2. Guha Mazumder, Diagnosis of Arsenicosis, PubMed 12635831
3. IJDVL, Arsenicosis: Diagnosis and Treatment
4. J Pak Assoc Dermatol 2014;24(4):286-291
5. PathologyOutlines.com, Arsenal Keratosis
6. Emu et al., ArsenicSkinImageBD, Data in Brief, 2023
7. Mehedi et al., ArsenicNet, 2025 (97.69% binary accuracy)

## 🙏 Acknowledgements
We sincerely thank our supervisor, Tanvir Ahmed, for his continuous guidance throughout this Computer Science project. We also thank the Department of Technology, AIUB, for the resources and support that made this research on ontology-guided explainable AI for arsenicosis screening possible.
