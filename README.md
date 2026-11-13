# Pneumonia Detection from Chest X-Ray Images Using Convolutional Neural Networks

![Course](https://img.shields.io/badge/Course-CSE366-blue)
![University](https://img.shields.io/badge/University-East%20West%20University-darkgreen)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20(T4%20GPU)-orange)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-ff6f00)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Chest%20X--Ray-20beff)
![Status](https://img.shields.io/badge/Status-Submissions%201--3%20complete-brightgreen)

**CSE366 - Artificial Intelligence, Section 1 | East West University**
**Group 09 | CNN Course Project**
Submitted to **Dr. Tania Sultana**, Assistant Professor, Dept. of CSE

This repository contains the notebooks of our course project on automatic detection of pneumonia from chest X-ray images. We analyse the dataset, review related work, apply three ImageNet-pretrained CNNs with and without fine-tuning, and design a custom CNN that is trained from scratch.

## Table of Contents
1. [Group Members](#group-members)
2. [Problem Statement](#1-problem-statement)
3. [Dataset](#2-dataset)
4. [Repository Contents](#3-repository-contents)
5. [Methodology](#4-methodology)
6. [Results](#5-results-test-set-624-images)
7. [How to Run](#6-how-to-run)
8. [Project Status](#7-project-status)
9. [References](#references)

## Group Members

| Name | Student ID |
|---|---|
| Nayeem Ahmed | 2024-1-60-294 |
| Mahbuba Hoque Tabassum | 2023-2-60-030 |
| Sharmin Zaman Setu | 2023-3-60-476 |
| SM Erbajul Hasan | 2023-3-60-010 |

## 1. Problem Statement

Pneumonia is a serious lung infection that is commonly diagnosed from chest X-rays. Reading the images takes expert time, so an automatic classifier could support screening. We treat it as a **binary image classification** task: **NORMAL** vs **PNEUMONIA** (PNEUMONIA is the positive class). Because a missed pneumonia case is more harmful than a false alarm, we report **recall** in addition to accuracy.

**Research questions**
1. How well do ImageNet-pretrained CNNs (DenseNet121, ResNet50, MobileNetV2) classify chest X-rays, and does fine-tuning help?
2. How does a custom CNN trained from scratch compare with them?

## 2. Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), Kaggle, based on D. S. Kermany et al., *Cell*, 2018 (license CC BY 4.0).

| Split (original) | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | 1,583 | 4,273 | 5,856 |

Main findings from our exploratory analysis (Submission 1):
- The training set is **imbalanced**: 25.7% NORMAL vs 74.3% PNEUMONIA (ratio 2.89 : 1).
- All images are JPEG, with very different sizes (width 384 to 2,916 px, height 127 to 2,713 px) and mixed colour modes (5,573 grayscale, 283 RGB).
- The original validation folder has only 16 images, so we do not use it (see Methodology).

## 3. Repository Contents

| Notebook | Content |
|---|---|
| `CNN_Group09_Submission1.ipynb` | Dataset information, exploratory data analysis (EDA) and related work (five peer-reviewed journal papers) |
| `CNN_Group09_Submission2_DenseNet121.ipynb` | Pretrained DenseNet121, with and without fine-tuning |
| `CNN_Group09_Submission2_ResNet50.ipynb` | Pretrained ResNet50, with and without fine-tuning |
| `CNN_Group09_Submission2_MobileNetV2.ipynb` | Pretrained MobileNetV2, with and without fine-tuning |
| `CNN_Group09_Submission3.ipynb` | Custom CNN trained from scratch, with early stopping |

All notebooks are saved **with their outputs** (plots, tables and reports), so the results can be read on GitHub without re-running them.

## 4. Methodology

**Data split (Submissions 2 and 3).** The 16-image original validation folder is too small, so a **stratified 15% of the training folder** is used for validation. The original **test folder (624 images) is used only for the final evaluation**. The resulting split is 4,433 training, 783 validation and 624 test images.

**Preprocessing and augmentation.** Images are decoded with 3 channels and resized to 224 x 224. Augmentation (training only): horizontal flip, rotation (about ±11 degrees), zoom (10%), translation (5%) and brightness change (10%). Pretrained models use their own `preprocess_input`; the custom CNN scales pixels to [0, 1].

**Class imbalance.** Balanced class weights are used (NORMAL 1.944, PNEUMONIA 0.673).

**Common training settings.** Batch size 32, Adam optimiser, binary cross-entropy, random seed 42, learning-rate reduction on plateau, and restoring the weights of the epoch with the lowest validation loss before testing.

**Evaluation metrics.** Accuracy, precision, recall, F1-score, AUC, confusion matrix, classification report, and training and testing time, all computed on the test set.

### Submission 2: Pretrained CNNs (ImageNet weights)
Global average pooling, dropout 0.3 and one sigmoid unit on top of each backbone, trained for 40 epochs in two modes:
- **Without fine-tuning:** the backbone is frozen and only the new head is trained (learning rate 1e-3).
- **With fine-tuning:** the top 40% of the backbone layers are also trained, with BatchNorm layers kept frozen (learning rate 1e-4).

### Submission 3: Custom CNN from scratch
Four convolutional blocks, each with two `Conv2D(3x3) + BatchNorm + ReLU` layers and max pooling (32, 64, 128, 256 filters), followed by global average pooling, a 128-unit dense layer, dropout 0.4 and a sigmoid output (about 1.21 million parameters, no pretrained weights). Training uses **early stopping** on the validation loss (patience 7, only active after the first 5 epochs, maximum 50 epochs).

## 5. Results (test set, 624 images)

| Model | Mode | Accuracy | Precision | Recall | F1 | AUC | Training time |
|---|---|---|---|---|---|---|---|
| DenseNet121 | Without fine-tuning | 0.8221 | 0.7949 | 0.9641 | 0.8714 | 0.9462 | 26.4 min |
| DenseNet121 | With fine-tuning | 0.7676 | 0.7298 | 0.9974 | 0.8429 | 0.9705 | 27.2 min |
| ResNet50 | Without fine-tuning | 0.7981 | 0.7629 | 0.9821 | 0.8587 | 0.9456 | 25.2 min |
| ResNet50 | With fine-tuning | **0.8558** | **0.8178** | 0.9897 | **0.8956** | 0.9637 | 26.1 min |
| MobileNetV2 | Without fine-tuning | 0.8494 | 0.8136 | 0.9846 | 0.8910 | 0.9657 | 25.0 min |
| MobileNetV2 | With fine-tuning | 0.8157 | 0.7734 | 0.9974 | 0.8712 | 0.9727 | 24.4 min |
| Custom CNN (scratch) | Early stopping (25 epochs) | 0.7388 | 0.7060 | 0.9974 | 0.8268 | 0.9358 | 21.9 min |

All pretrained models ran 40 epochs per mode. Training times were measured on a Colab T4 GPU and depend heavily on image loading, so they are only a rough comparison.

### Key findings
- **Best model:** ResNet50 with fine-tuning (accuracy 0.8558, F1 0.8956). Fine-tuning helped ResNet50, but it did **not** improve test accuracy for DenseNet121 and MobileNetV2.
- **High recall, low specificity:** all models detect almost every pneumonia case (recall 0.96 to 0.997) but wrongly flag many healthy X-rays as pneumonia (NORMAL recall about 0.31 to 0.63). For example, the custom CNN misses 1 of 390 pneumonia cases but flags 162 of 234 normal X-rays.
- **Custom CNN vs pretrained models:** the custom CNN has the lowest test accuracy, as expected for a network with only about 1.2 million parameters and no ImageNet knowledge, but a comparable AUC (0.936 against 0.946 to 0.973).
- **Validation-test gap:** validation accuracy (0.93 to 0.99) is much higher than test accuracy (0.74 to 0.86) for every model, which shows that our validation set is too optimistic.

### Limitations
- The validation set comes from the same folder as the training images, and we did not check whether images of the same patient appear in both parts. This may explain the validation-test gap.
- The dataset contains only pediatric X-rays from a single hospital, so results may not transfer to other populations or scanners.
- Each model was trained once with a single random seed, so small differences between models may not be significant.
- These models are for coursework only and are **not** medical devices.

## 6. How to Run

1. Open a notebook in [Google Colab](https://colab.research.google.com) (use the "Open in Colab" badge in the notebook, or upload the `.ipynb` file).
2. Set the runtime to a GPU: `Runtime` > `Change runtime type` > `T4 GPU`.
3. When asked, upload **your own** `kaggle.json` (Kaggle API token) so the dataset can be downloaded. Never commit this file to GitHub.
4. Run all cells in order (`Runtime` > `Run all`).

**Tools and technologies:** Python 3, TensorFlow 2.x / Keras, scikit-learn, pandas, NumPy, Matplotlib, Kaggle API, Google Colab.

## 7. Project Status

| Submission | Content | Status |
|---|---|---|
| 1 | Dataset information, EDA, related work | Done |
| 2 | DenseNet121, ResNet50, MobileNetV2 (with and without fine-tuning) | Done |
| 3 | Custom CNN from scratch | Done |


## References

Related work on pneumonia detection from chest X-rays (the five peer-reviewed papers reviewed in Submission 1):

1. O. Stephen, M. Sain, U. J. Maduh and D.-U. Jeong, "An Efficient Deep Learning Approach to Pneumonia Classification in Healthcare," *Journal of Healthcare Engineering*, article 4180949, 2019.
2. V. Chouhan et al., "A Novel Transfer Learning Based Approach for Pneumonia Detection in Chest X-ray Images," *Applied Sciences*, vol. 10, no. 2, p. 559, 2020.
3. T. Rahman et al., "Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection Using Chest X-Ray," *Applied Sciences*, vol. 10, no. 9, p. 3233, 2020.
4. R. Kundu et al., "Pneumonia detection in chest X-ray images using an ensemble of deep learning models," *PLOS ONE*, vol. 16, no. 9, e0256630, 2021.
5. S. Cha, S. Lee and B. Ko, "Attention-Based Transfer Learning for Efficient Pneumonia Detection in Chest X-ray Images," *Applied Sciences*, vol. 11, no. 3, p. 1242, 2021.

Dataset: D. S. Kermany et al., *Cell*, vol. 172, no. 5, 2018 (Kaggle: `paultimothymooney/chest-xray-pneumonia`, CC BY 4.0).
