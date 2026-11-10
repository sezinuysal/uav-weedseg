# UAV Weed Segmentation: Uncertainty-Aware Crop & Weed Segmentation

Semantic segmentation of crops and weeds in UAV (drone) images of maize fields, with a focus on **model evaluation and prediction uncertainty**.
The project trains and evaluates **DeepLabV3+ (ResNet-50)** and **MaskFormer** models on the public **WeedsGalore** multispectral dataset.

## What this project does

- **Training** a DeepLabV3+ (ResNet-50) model with dropout on RGB images (3 classes), with periodic checkpointing.
- **Evaluation** with per-class IoU (Jaccard index) and a confusion matrix.
- **Uncertainty analysis** using **Monte Carlo Dropout**: multiple stochastic forward passes produce a mean prediction and a **predictive entropy map** that highlights where the model is unsure.
- **Experiment tracking**: validation images are logged with their predictions to **Weights & Biases**.
- **MaskFormer pipeline** (Detectron2): a dataset loader for WeedsGalore (3-band RGB or 5-band B/G/R/RE/NIR input), config generation and training scripts (RGB 6-class and CPU variants).
- **EDA** of the dataset in a notebook.

## Repository structure

```
src/
  1_env/                  environment setup (torch.load patch) + DeepLabV3+ training runner with checkpoints
  2_deeplabv3/            IoU + confusion matrix, Monte Carlo Dropout evaluation, W&B validation logging
  3_maskformer_dataset/   Detectron2 dataset loader for WeedsGalore + MaskFormer config generator
  4_maskformer_train/     MaskFormer training scripts (RGB 6-class, CPU)
notebooks/
  eda.ipynb               exploratory data analysis
third_party/
  weedsgalore/            WeedsGalore dataset code and baseline models (see Acknowledgements)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

MaskFormer scripts additionally need [Detectron2](https://github.com/facebookresearch/detectron2) and the [MaskFormer](https://github.com/facebookresearch/MaskFormer) code.
Download the WeedsGalore dataset from the link in `third_party/weedsgalore/README.md`.

> **Note:** the scripts were written and run in a Kaggle notebook environment, so some paths (for example `/kaggle/input/...` and `/kaggle/working/...`) are Kaggle-specific. Adjust them to your local paths before running.

## Key ideas

**Monte Carlo Dropout.** Dropout layers are kept active at inference time. Running several stochastic forward passes (5 in the evaluation script) gives a mean class probability per pixel. The entropy of this mean distribution is used as an uncertainty map, which helps find regions where the model is likely to make mistakes (for example weed/crop boundaries).

**Evaluation.** Besides overall mIoU, per-class IoU and a confusion matrix show which classes are confused with each other.

## Acknowledgements

This project builds on the work of others:

- **WeedsGalore dataset and baseline code:** Celikkan et al., *WeedsGalore: A Multispectral and Multitemporal UAV-Based Dataset for Crop and Weed Segmentation in Agricultural Maize Fields*, WACV 2025. Included under `third_party/weedsgalore/` with its original license. See [arXiv:2502.13103](https://arxiv.org/abs/2502.13103).
- **MaskFormer:** [facebookresearch/MaskFormer](https://github.com/facebookresearch/MaskFormer) (used via Detectron2).
- **DeepLabV3+** with a ResNet-50 backbone, as implemented in the WeedsGalore baseline code.
