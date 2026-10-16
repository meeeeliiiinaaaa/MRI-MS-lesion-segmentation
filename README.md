# MRI-MS-lesion-segmentation

**Classical machine learning baselines for Multiple Sclerosis lesion segmentation on multi-modal brain MRI (FLAIR, T1, T2).**

Pixel-wise lesion classification using hand-crafted texture and intensity features, comparing five classical ML models: Linear SVM, RBF SVM, Logistic Regression, AdaBoost, and XGBoost, on the [MSLesSeg](https://iplab.dmi.unict.it/mfs/ForArticles/MSLesSeg/) dataset.

*Author: Melina Rajabi, Summer 2025*

---

## Overview

Multiple Sclerosis (MS) lesions appear as small, scattered hyperintense regions in FLAIR MRI, and segmenting them is a highly class-imbalanced problem (lesion voxels are a tiny fraction of the volume). This project explores how far classical, non-deep-learning approaches can go on this task by combining rich per-pixel feature engineering with different classifiers, and evaluating them with a shared preprocessing, prediction, and post-processing pipeline.

Everything lives in a single notebook: `MRI-MS-lesion-segmentation.ipynb`.

## Pipeline

1. **Loading**: NIfTI volumes (`.nii.gz`) for FLAIR, T1, T2, and the ground-truth lesion mask, read with `nibabel`.
2. **Cleaning and normalization**: NaN/Inf handling, percentile clipping (0.5–99.5%), robust (median/IQR) scaling for FLAIR, z-score scaling for T1 and T2.
3. **Feature extraction (per pixel, per 2D slice)**:
   - Raw intensities of FLAIR, T1, T2
   - 5×5 patch statistics per modality (mean, std, median, min, max, range, center, center minus mean)
   - 3×3 neighborhood intensities and gradients
   - Sobel gradient magnitudes and Laplacian-of-Gaussian responses
   - Gabor filter responses (multiple frequencies and orientations)
   - Local Binary Patterns (multi-radius)
   - FLAIR/T1 and FLAIR/T2 intensity ratios
   - Normalized spatial coordinates
4. **Training set construction**: only slices containing lesions are used; all lesion pixels are kept and negatives are subsampled at a 3:1 ratio to handle class imbalance.
5. **Model training**: each model is trained with its own imbalance handling and tuning (see below).
6. **Inference**: slice-by-slice probability maps, per-patient threshold selection, then 3D morphological post-processing (opening, closing, hole filling, small-object removal).
7. **Evaluation and visualization**: accuracy, F1, and Dice per patient, with FLAIR / prediction / ground truth / error-map plots.

## Models

| Model | Notes |
|---|---|
| **Linear SVM** | `LinearSVC` after StandardScaler + PCA (95% variance); grid search over C, lesion-class and background-class sample weights |
| **RBF SVM** | `SVC` with RBF kernel trained on a stratified 10k-sample subset, balanced class weights |
| **Logistic Regression** | RBF-feature-mapped logistic regression with small-lesion-weighted samples, center weighting, and F2-tuned threshold |
| **AdaBoost** | Depth-2 decision-tree stumps, balanced sample weights, staged early stopping on a validation set |
| **XGBoost** | `binary:logistic`, 500 rounds, `scale_pos_weight` for imbalance, L1/L2 regularization |


## Dataset

This project uses **MSLesSeg** (multi-modal MS lesion segmentation with FLAIR, T1, T2 and expert masks). The dataset is not included; please download it from the official source.


