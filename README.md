# Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures

This repository is the ```official open-source``` of Leveraging Vision Transformers for High-Precision Classification of Cancer Cell Cultures: A Comparative Study on MDA-MB-231 and PC3 Datasets"

By
Noreen Fayyaz Khan  , Lu Liu ,Lucas Bierscheid, John C. Wilkinson and Changhui Yan


## Description
The classification of cancer cell cultures is critical in understanding tumor behavior, drug responses, and disease progression. Traditional manual evaluation methods are often subjective and prone to errors, necessitating automated approaches based on deep learning. Convolutional Neural Networks (CNNs) have played a crucial role in biomedical research for image classification, but their inability to capture long-range dependencies hinders their effectiveness in complex datasets. Vision Trans- formers (ViTs) have emerged as a promising alternative, leveraging self-attention mechanisms to model global spatial relationships. This study compares CNNs and Vits for the classification of cancer cell culture images, focusing on MDA-MB-231 (triple negative breast cancer) and PC3 (prostate cancer). A structured preprocessing pipeline is implemented that incorporates advanced segmentation techniques, including Otsu thresholding, morphological operations, and the watershed algorithm, to enhance the dataset's quality. Data augmentation techniques are applied to address class imbalance and improve model generalization. Both CNN and ViT models undergo hyperparameter tuning using a 5-fold cross-validation grid search, and their classification effectiveness is evaluated using precision, recall, F1-score, accuracy, confusion matrices, and ROC curves. Our results show that ViTs outperform CNNs in terms of accuracy and generalization capabilities. These findings highlight the potential of transformer-based architectures in biomedical image analysis, paving the way for more robust and scalable deep learning applications in cancer research.

![Figure 1](https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures/raw/master/Fig1.png)

**Figure 1.** Overview of the model architecture.


## Prerequisites
"
Python 3.9

keras 2.9.0

tensorflow-gpu 2.9.1

CUDA toolkit 11.0

cuDNN 8.0

tensorflow-addons 0.17.1

tensorboard 2.9.1

scikit-learn 1.0.2

numpy 1.21.6

opencv-python 3.4.1

matplotlib 3.5.3

seaborn 0.12.2

glob2 0.7

"
This method was tested in:
- **GPU**: RTX 3070 with 32 GB memory
- Operating System: Ubuntu 20.04 / CentOS HPC environment


## Usage
Clone the repository:
```bash
git clone https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures.git
```

## Installation 
To create a conda environment with the required packages, use the following command:
```bash
conda env create -f flax_tf.yml
```

## OurDataset 
1) Data Collection
Microscopy images of two cancer cell lines, MDA-MB-231 and PC3, were obtained in co-author John Wilkinson’s group using Phase-Contrast Microscopy (PCM). The MDA-MB-231 cell line is a triple-negative breast cancer cell culture (TNBC), while PC3 is derived from prostate cancer. Exponentially growing cells were harvested by trypsinization and plated at three different densities: 10,000 cells/well, 20,000 cells/well, and 40,000 cells/well. The cells were then cultured in RPMI-1640 with 10% Fetal Bovine Serum for three days, during which the cancer cells exhibited different morphological characteristics, such as shape and size, depending on cell type, growth condition, and growth time. Cell images were collected each day using a Nikon TS100 microscope with a 10X phase-contrast objective. The images of each cell line were classified into nine classes, which were defined by the combination of initial cell density and growth time. For example, PC3-20KD2 is the class of images of PC3 cells taken at the end of the second day with an initial cell density of 20,000 cells/well. There were 6 replicates for each cell line. For each class, 90 images were taken.
2) Data Preprocessing
To ensure the data was clean, consistent, and suitable for training deep learning models, we preprocessed the images using a protocol specifically tailored to address the challenges of cell culture images. The preprocessing protocol includes two steps: image cropping and augmentation. The purpose of image cropping is to obtain images of single cells isolated from their neighbors and the environment. This step is necessary for detecting morphological features of the cells, such as size and shape. The augmentation step enhances the diversity of the dataset.
The creation of image files from cell culture to obtain images of individual cells presents unique challenges. Simply cutting the original images into smaller patches, such as 500×500 pixels, would result in empty patches with no cells for images taken on day 1, and patches with many partial cells in images taken on day 3. A commonly used approach is to isolate single cells using segmentation and patch each cell image to a preferred size with a black background. However, this approach does not work well when the cell density is high, where adjacent cells are indistinguishable by segmentation. Additionally, contextual information is lost when a black background is manually introduced, which could affect the accuracy of the resulting model.
To overcome these challenges, we implemented a method to isolate and normalize single-cell images. The method started by converting the input image to grayscale. Then, the Otsu method was used to distinguish cells from the background by automatically finding an optimal threshold. The resulting binary image was refined using morphological operations to remove minor artifacts and gap filling to close small gaps in cell structures. The watershed segmentation, which treated the image as a topographical map, was then used to separate overlapping or adjacent cells. The boundary boxes were detected for the regions of interest (ROIs), and ROIs that were too small or touched the edge of the image were filtered out. Finally, the remaining ROIs were cropped and normalized to a size of 500×500 pixels.
Image augmentation was also used to address the issues of data sparsity and imbalance. This involved a series of transformations, including horizontal flip, rotations within -45° to 45°, and Gaussian blurring. The augmentation method was applied to the isolated cell images until each class contained 1,000 images of isolated cells. This preprocessing pipeline resulted in a robust dataset for downstream steps of model training, ensuring that the data were clean, consistent, diverse, and balanced. The final dataset consists of 9 classes for each cell line, with each class having 1,000 images.

You can download OurDataset using the download link provided below.
```
https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures/tree/master/MDA231_Final/MDA231_B

```
```
https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures/tree/master/PC3B_Final/PC3B_dataset
```

|  File Name |  Download   |   
|:-----------|:-----------|
```

OurDataset
├── testing
        └── PC3_10kD1
            ├── PC3_10kD1(800).png
            ├── ...
            └── PC3_10kD1(1000).png
        ├── PC3_10kD2
        ├── ...
        └── PC3_40kD3
└── training
       └── PC3_10kD1
            ├── PC3_10kD1(1).png
            ├── ...
            └── PC3_10kD1(799).png
        ├── PC3_10kD2
        ├── ...
        └── PC3_40kD3
```

## Training/Testing

To train and testthe model with all default values, use the following script:

```bash
CNN_k_fold.py
CNN_simple.py
vit_k_fold.py
vit_simple.py
 
```
## Qualitative Evaluation
Visualization Results of the proposed method. Overall score of precision, recall and F1-Score for each
experiment.

![Framework](https://github.com/noreenfayyaz/Vision-Transformer-for-High-Precision-Classification-of-Cancer-Cell-Cultures/raw/master/Fig2.png)

## Citation
If you find this code useful, please cite our paper:
```

```

## Acknowledgments
Internal funding has been secured to support this research article, demonstrating the commitment of their institution to advancing knowledge within their academic community. This work used resources of the Center for Computationally Assisted Science and Technology (CCAST) at North Dakota State University.

## Contact
```
noreen.f.khan@ndsu.edu
noreen.fayyaz@fu.edu.pk

```
