# Brain Tumor Classification from MRI Images using Deep and Traditional Machine Learning Approaches

**Course:** CAP5610 Group Project  

**Team:** Brain MRI Group 4  

**Members:** Thomas Belyakov, Sahil Bhikha, Carsen Billings, David Almeida II  

---

## Problem definition

### What are you predicting?

The objective of this project is to develop a supervised machine learning model capable of classifying brain MRI images into one of four categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**.

### Why does it matter?

Given the nature of this data, this project mimics a realistic scenario where a machine learning model would be valuable. In medicine, time is critical when diagnosing illness, especially when different conditions may present similar symptoms. The longer the delay to an accurate diagnosis, the longer treatment must wait. A successful model in this setting matters because treatment decisions, prognosis, and surgical planning depend heavily on tumor subtype. We develop and evaluate models with that clinical importance in mind.

### What type of ML task is this?

This is a **multi-class image classification** problem. The goal is to evaluate how effectively different machine learning approaches distinguish four classes using MRI image data.

---

## Dataset description

| Item | Details |
|------|---------|
| **Source** | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle) |
| **Approximate size** | 7,200 human brain MRI images (~170 MB). Roughly **5,600** in the training subset and **1,600** in the test subset. |
| **Feature types** | Raw pixel data; grayscale images at varying resolutions, **resized during preprocessing**. |
| **Classes** | Glioma, meningioma, pituitary tumor, or no tumor. |
| **Target variable** | The tumor classification label associated with each MRI image. |

### Known issues

- **Variable image dimensions:** Scan sizes can differ across images, which can complicate direct comparison during training. **Standardizing image size** is a key preprocessing step.
- **Class balance:** The classes in this dataset are balanced on retrieval. Because each class has the same number of images, there is a possibility of **duplicate images** across or within splits—worth checking during exploratory analysis.

---

## Baseline models (`Brain_Mri_Baseline.ipynb`)

The baseline stage uses **two complementary models** on the same preprocessing pipeline—**resized grayscale images flattened into pixel vectors**—so later work (CNN, PCA+SVM, etc.) has a clear, repeatable reference.

1. **Random Forest** — `RandomForestClassifier` on raw flattened pixels (scaled 0–1), with balanced class weights. This is a strong **traditional ensemble** baseline on tabular-style features.
2. **Multi-layer perceptron (MLP)** — `MLPClassifier` (e.g. hidden layers 256→128) on **standardized** flattened features. This is a **neural** baseline on the same vectorized representation, not a single-layer perceptron.

Together, these define our **baseline bundle**: tree-based vs. feedforward neural, both before spatial deep learning or separate dimensionality-reduction pipelines.

---

## Proposed models

### Convolutional neural network (CNN)

Beyond these **flattened-feature** baselines (Random Forest and MLP), we plan to evaluate a **CNN**, which learns spatial structure from image tensors instead of a single long pixel vector. CNNs are trained on labeled data via backpropagation and gradient descent, with convolutional layers that share weights across space for richer representations than raw flattening alone.

### PCA + SVM

Stronger linear separability might mean a **less complex** pipeline is sufficient: **Principal Component Analysis (PCA)** to reduce dimensionality of the image data, plus a **Support Vector Machine (SVM)** to separate the four classes with decision boundaries in the reduced space. This path is typically **less computationally intensive** than a full CNN but may cap performance on subtle spatial patterns.

---

## Evaluation strategy

### Train / test split

The data is split into approximately **5,600** training images and **1,600** test images. If needed, the training set is large enough to carve out a **validation** set for tuning (for example, on the order of ~1,600 images, depending on the final split strategy).

### Cross-validation

We plan to use **cross-validation**. Although the test subset has **400 images per class**, the training subset might not be as evenly distributed across classes. **Stratified K-fold cross-validation** helps keep class proportions stable across folds and reduces bias from uneven class sampling during model selection.

### Metrics

| Metric | Role |
|--------|------|
| **Macro F1-score** | Balances precision and recall; the macro variant averages **per-class** F1 with **equal weight**, which helps when class counts or difficulty differ across folds. |
| **Per-class precision and recall** | Precision shows how often a predicted class is correct; recall shows how many true cases of that class are found. **False negatives** (e.g., predicting no tumor when a tumor is present) are especially costly, so recall-related views matter. |
| **Confusion matrix** | Shows predicted vs. actual counts per class and highlights **systematic misclassifications** that may suggest data or model issues. |
| **ROC-AUC (per class)** | For multi-class problems, **one-vs-rest** ROC-AUC per class summarizes how well the model separates each class from the others across thresholds, complementing confusion matrices and precision–recall. |
| **Accuracy** | Overall fraction of correct predictions—useful for reporting, but **not** the primary metric given four classes and possible imbalance or uneven difficulty. |

---

## Scope justification

The dataset scale (~7,200 labeled images, four classes) supports training and comparing several models. Clear **baseline models** (Random Forest and MLP on flattened features), an optional validation split for tuning, and the metrics above give a **repeatable** way to compare approaches. Confusion matrices and per-class metrics help explain **where** models fail, not just aggregate scores.

Because we work from **preprocessed** images, we may need extra checks (resizing, normalization, duplicate detection) to keep training, validation, and testing stable. As a **public**, pooled dataset, scans may come from heterogeneous sources; subtle acquisition differences can affect what the model learns, so results should be interpreted alongside **clinical usability** and the gap between benchmark metrics and real deployment.
