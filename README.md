# CNN-Based Pneumonia Detection from Chest X-Ray Images

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org)

## 📌 Overview
Pneumonia is a critical respiratory infection requiring early detection. This project utilizes a Convolutional Neural Network (CNN) architecture to automate the binary classification of Chest X-Ray images into **Normal** and **Pneumonia** categories.

---

## 🔬 Dataset & Preprocessing
- **Classes:** 2 (`NORMAL`, `PNEUMONIA`)
- **Image Preprocessing:** 
  - Image Resizing to standard dimensions
  - Pixel Value Normalization ($[0, 1]$ range)
  - Data Augmentation (Rotation, Zoom, Horizontal Flip) to enhance model generalization

---

## 🏗️ Model Architecture
The deep learning pipeline consists of:
1. **Input Layer:** Normalized image tensor
2. **Convolutional Layers:** Feature extraction with `ReLU` activation
3. **Pooling Layers:** `MaxPooling2D` for downsampling
4. **Regularization:** `Dropout` & `BatchNormalization` to prevent overfitting
5. **Output Layer:** Dense layer with `Sigmoid` / `Softmax` activation for prediction

---

## 💻 Environment & Tools
- **Language:** Python 3.x
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, OpenCV
- **Platform:** Google Colab (GPU Enabled)

---

## 🚀 How to Run
1. Clone or download this repository.
2. Open the `.ipynb` notebook file in **Google Colab** or **Jupyter Notebook**.
3. Enable GPU acceleration (`Runtime` -> `Change runtime type` -> `GPU`).
4. Execute all code cells sequentially.
