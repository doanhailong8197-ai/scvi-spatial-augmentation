# Balanced Generative Augmentation Improves SPANN for Imbalanced Spatial Cell-Type Annotation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20scVI--tools-orange.svg)]()

> Official source code repository for the manuscript: *"Balanced Generative Augmentation Improves SPANN for Imbalanced Spatial Cell-Type Annotation"*.

## 📖 Overview

Spatial transcriptomics provides unprecedented insights into tissue architecture, but transferring cell-type labels from single-cell RNA sequencing (scRNA-seq) references is often hindered by severe class imbalance. Rare cell populations are easily overshadowed by majority classes, leading to misclassification and poor novel cell discovery.

This repository provides a systematic framework to address this challenge. We introduce robust data augmentation strategies—including deep generative modeling via **scVI** and stochastic interpolation via **Mixup**—to explicitly balance the reference scRNA-seq data prior to cross-modality integration. Our pipeline seamlessly integrates with **[SPANN](https://github.com/ddb-qiwang/SPANN-torch)** to significantly improve both closed-set annotation accuracy and open-set novel cell discovery.

---

## 📂 Repository Structure

The project is modularized to ensure reproducibility and readability:

* `data/`: Data directory (ignored in git).
    * `seqFISH/`:
    * `CID4435/`:
* `external/SPANN/`: Git submodule linking to the original SPANN repository.
* `figures/`: Publication-ready plots (PDF format).
* `notebooks/main_workflow.ipynb`: Primary execution pipeline.
* `src/`: Core computational modules.
    * `augmentation/scvi_balancer.py`: Deep generative augmentation.
    * `augmentation/mixup_balancer.py`: Stochastic hard-label mixup.
    * `evaluation.py`: K-ACC, U-ACC, and H-Score metrics.
    * `plotting.py`: Standardized visualization functions.
    * `stats.py`: Imbalance Ratio (IR) and Gini calculations.
* `requirements.txt`: Package dependencies.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
Because this project utilizes SPANN as a Git Submodule, you **must** include the `--recurse-submodules` flag when cloning:

git clone --recurse-submodules [https://github.com/doanhailong8197-ai/scvi-spatial-augmentation.git](https://github.com/doanhailong8197-ai/scvi-spatial-augmentation.git)
cd scvi-spatial-augmentation

### 2.Environment Setup

We recommend creating a virtual environment before installing the dependencies:

conda create -n spatial_aug python=3.10
conda activate spatial_aug

pip install -r requirements.txt

### 3.Data Availability

The raw single-cell RNA-seq and spatial transcriptomics datasets used in this study are publicly available.
To reproduce our results, please download the required .h5ad files from our Google Drive Repository.

Instructions:
1. Download the files adata_rna.h5ad and adata_seqfish_40.h5ad from https://drive.google.com/drive/folders/1uwImdE890sQZZSY7kG44XuY3ehoWow9Y?usp=share_link
2. Place them precisely into the data/raw/ directory inside this project:
scvi-spatial-augmentation/
└── data/
    └── seqFISH/
        ├── adata_rna.h5ad
        └── adata_seqfish_40.h5ad

### 4.Usage

The entire experimental workflow—from data loading, imbalance analysis, data augmentation (scVI/Mixup), SPANN integration, to final evaluation—is orchestrated through a single Jupyter Notebook.

1. Launch Jupyter Notebook:
jupyter notebook

2. Open notebooks/main_workflow.ipynb.

3. Follow the sequential execution cells. You can easily toggle between augmentation methods by changing the AUGMENTATION_METHOD variable in the notebook:

AUGMENTATION_METHOD = "scVI"  # Options: "scVI", "Mixup", or "None"

All generated evaluation metrics will be printed in the console, and publication-ready figures will be automatically saved to the figures/ directory.

