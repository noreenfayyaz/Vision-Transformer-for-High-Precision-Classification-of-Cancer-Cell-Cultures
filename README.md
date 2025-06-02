# Vision Transformer for High-Precision Classification of Cancer Cell Cultures

This repository contains code and data for our research on using Vision Transformers (ViT) and CNNs to classify phase-contrast microscopy images of two cancer cell lines: **MDA-MB-231 (breast cancer)** and **PC3 (prostate cancer)**.

---

## 📌 Abstract

Accurate classification of cancer cell morphology is critical for biomedical research and diagnostics. In this project, we investigate the use of Convolutional Neural Networks (CNN) and Vision Transformer (ViT) architectures to classify microscopy images of MDA-MB-231 and PC3 cell cultures under various growth conditions. A dataset was developed by segmenting individual cells and augmenting them across nine morphological classes. Our experiments show that transformer-based models outperform conventional CNNs in both accuracy and generalization.

---

## 🖼️ Figures

<p align="center">
  <img src="https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures/raw/master/Fig1.png" width="600"/>
  <br>
  <b>Figure 1.</b> The ViT-based classification pipeline.
</p>

<p align="center">
  <img src="https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures/raw/master/Fig2.png" width="600"/>
  <br>
  <b>Figure 2.</b> Classification Report of MDA-MB-231 (breast cancer) and PC3 (prostate cancer) on Test Dataset.
</p>

---

## 🧪 Dataset Description

- **Cell Lines**: MDA-MB-231 (breast), PC3 (prostate)
- **Image Type**: Phase-Contrast Microscopy (PCM)
- **Classes**: 9 morphological conditions per cell line
- **Preprocessing**:
  - Cell segmentation via Otsu + Watershed
  - ROI cropping and normalization to 224×224
  - Augmentation: horizontal flip, rotation (±45°), Gaussian blur
- **Final Size**: 18,000 images (9 classes × 2 lines × 1,000 images/class)

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures.git
cd Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

Or set up your environment with:

```bash
conda create -n cancer-classification python=3.9
conda activate cancer-classification
pip install tensorflow-gpu==2.9.1 keras==2.9.0 opencv-python scikit-learn matplotlib seaborn tensorflow-addons
```

### 3. Run Models

```bash
# CNN
python CNN_simple.py

# Vision Transformer - simple
python vit_simple.py

# Vision Transformer - k-fold with grid search
python vit_k_fold.py
```

---

## ⚙️ Prerequisites

- Python 3.9
- keras 2.9.0
- tensorflow-gpu 2.9.1
- CUDA toolkit 11.0
- cuDNN 8.0
- tensorflow-addons 0.17.1
- tensorboard 2.9.1
- scikit-learn 1.0.2
- numpy 1.21.6
- opencv-python 3.4.1
- matplotlib 3.5.3
- seaborn 0.12.2

> ✅ Tested on:  
> GPU: RTX 3070 with 32 GB memory  
> OS: Ubuntu 20.04 / HPC (PBS)

---

## 📊 Results Summary

| Model      | Dataset     | Accuracy  | Precision | Recall | F1-Score |
|------------|-------------|-----------|-----------|--------|----------|
| CNN        | MDA-MB-231  | 71.0%     | 0.71      | 0.70   | 0.70     |
| CNN        | PC3         | 87.2%     | 0.86      | 0.87   | 0.86     |
| ViT Simple | MDA-MB-231  | 84.0%     | 0.85      | 0.84   | 0.84     |
| ViT Simple | PC3         | 92.4%     | 0.92      | 0.92   | 0.92     |
| ViT K-Fold | MDA-MB-231  | **86.1%** | 0.87      | 0.86   | 0.86     |
| ViT K-Fold | PC3         | **94.1%** | 0.94      | 0.94   | 0.94     |


---

## 📄 License

This project is licensed under the [License](LICENSE).

---

## 🙌 Acknowledgements

- Phase-contrast microscopy images were collected in collaboration with Dr. John Wilkinson’s lab.
- Special thanks to HPC support staff for providing compute resources.

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{noreen2025vitcancer,
  author       = {Noreen F. Khan and  Changhui Yan*},
  title        = {Vision Transformer for High-Precision Classification of Cancer Cell Cultures},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures}}
}
```

Or include:  
**Khan, N.F. et al. (2025). _Vision Transformer for High-Precision Classification of Cancer Cell Cultures_. GitHub. https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures**
