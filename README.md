# Pneumonia Detection from Chest X-Ray Images Using CNNs

**Group 09 | CNN Course Project**

Binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** using convolutional neural networks (TensorFlow / Keras, run on Google Colab with a T4 GPU).

## Group Members
| Name | Student ID |
|---|---|
| Nayeem Ahmed | 2024-1-60-294 |
| Mahbuba Hoque Tabassum | 2023-2-60-030 |
| Sharmin Zaman Setu | 2023-3-60-476 |
| SM Erbajul Hasan | 2023-3-60-010 |

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, based on Kermany et al., *Cell*, 2018 (CC BY 4.0).

- 5,856 images in two classes: NORMAL and PNEUMONIA
- Original split: train 5,216 / validation 16 / test 624
- Class imbalance in training data: 1,341 NORMAL vs 3,875 PNEUMONIA (about 2.89 : 1)
- Because the original validation set has only 16 images, we use a stratified 15% validation split from the training folder.

## Notebooks
| File | Content |
|---|---|
| `CNN_Group09_Submission1.ipynb` | Submission 1: dataset information, exploratory data analysis (EDA), related work |
| `CNN_Group09_Submission2_DenseNet121.ipynb` | Submission 2: DenseNet121, with and without fine-tuning |
| `CNN_Group09_Submission2_ResNet50.ipynb` | Submission 2: ResNet50, with and without fine-tuning |
| `CNN_Group09_Submission2_MobileNetV2.ipynb` | Submission 2: MobileNetV2, with and without fine-tuning |

## Submission 2 Setup
- ImageNet-pretrained models from `tf.keras.applications`, input size 224 x 224, batch size 32, 40 epochs, seed 42
- **Without fine-tuning:** frozen backbone, new classification head (learning rate 1e-3)
- **With fine-tuning:** top 40% of backbone layers unfrozen, BatchNorm layers kept frozen (learning rate 1e-4)
- Class weights for the imbalance, data augmentation, dropout (0.3), ReduceLROnPlateau
- Best-validation-loss weights restored before testing
- Reported on the 624-image test set: accuracy, precision, recall, F1, AUC, confusion matrix, training time and testing time

## How to Run
1. Open a notebook in Google Colab (use the "Open in Colab" badge inside the notebook).
2. Set the runtime to GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. Upload your own `kaggle.json` (Kaggle API token) when asked, so the dataset can be downloaded. Do not commit this file to GitHub.
4. Run all cells in order.

## Status
- Submission 1: done
- Submission 2 (DenseNet121, ResNet50, MobileNetV2): done
