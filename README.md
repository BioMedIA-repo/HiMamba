# HiMamba

Official implementation of **HiMamba: A Hierarchical State Space Framework for
Asymmetric Multimodal Data Mining in Clinical Assessment**.

HiMamba fuses a hip X-ray image with five structured clinical indicators
(sex, age, height, weight, BMI) to regress the Appendicular Skeletal Muscle
Index (ASMI). It contains two complementary modules:

- **FAMF (Fine-grained Aligned Mamba Fusion)**: aligns each clinical indicator
  to image spatial tokens via L2-normalized cosine similarity, interleaves the
  matched pairs and models their interactions with a Mamba selective-scan block.
- **GCAS (Global Clinical Attentive Screening)**: uses the global clinical
  token to cross-attend to image regions, screening task-relevant muscle areas.

Training optimizes `L = MSE + λ · InfoNCE`.

## Table of Contents
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Repository Structure](#repository-structure)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Requirements
```bash
pip install -r requirements.txt
```

## Dataset Preparation
The clinical cohort used in the paper is private and therefore not
redistributed here. Organize the raw data as follows, where each case has one
folder of anteroposterior pelvic DICOM images and one row in the raw CSV:

```
data/
├── raw/
│   ├── clinical.csv            # ID, AGE, Gender, Height, Weight, BMI, ASMI
│   └── images/<ID>/*.dcm
```

### 1. Preprocess images and tabular data
Resizes every X-ray to 224×224, normalizes the five clinical indicators and
the ASMI label with min-max scaling, and writes the processed CSV:
```bash
python -m dataloaders.preprocess \
    --raw_csv data/raw/clinical.csv \
    --dicom_root data/raw/images \
    --output_csv data/sarcopenia_processed.csv \
    --output_image_dir data/images_224
```

### 2. Generate the three-fold splits
```bash
python -m dataloaders.make_splits \
    --data_csv data/sarcopenia_processed.csv \
    --output data/3_fold_splits.pkl
```

### 3. ImageNet-pretrained backbone
Download `resnet18-f37072fd.pth` into `checkpoints/`, or pass `--pretrained`
to let torchvision fetch the weights automatically.

## Training
Three-fold cross-validation (100 epochs, batch size 64, Adam, linear warm-up
followed by polynomial decay, mixed precision):
```bash
python train.py \
    --data_csv data/sarcopenia_processed.csv \
    --split_file data/3_fold_splits.pkl \
    --image_dir data/images_224 \
    --pretrained_path checkpoints/resnet18-f37072fd.pth \
    --save_dir runs/himamba --gpu 0
```
The best checkpoint of each fold (selected by validation PCC) is saved under
`runs/himamba/checkpoints`, with prediction scatter plots in
`runs/himamba/figures` and TensorBoard events in `runs/himamba/tensorboard`.

## Testing
```bash
python test.py \
    --data_csv data/sarcopenia_processed.csv \
    --split_file data/3_fold_splits.pkl \
    --image_dir data/images_224 \
    --checkpoint_dir runs/himamba/checkpoints
```
Reports per-fold and mean±std PCC, MAE and R²; ICC(3,1) is additionally
reported when `pingouin` is installed.

## Repository Structure
```
HiMamba/
├── dataloaders/
│   ├── dataset.py          # Dataset, collation, DataLoader factory
│   ├── auto_augment.py     # Photometric augmentation
│   ├── preprocess.py       # Raw DICOM/CSV -> processed CSV + 224x224 .npy
│   └── make_splits.py      # Three-fold train/val/test split generation
├── networks/
│   ├── himamba.py          # HiMamba: FAMF + GCAS + regression head
│   ├── image_encoder.py    # ResNet-18/34 image encoder
│   ├── tabular_encoder.py  # Transformer encoder of clinical indicators
│   ├── mamba_block.py      # Pure-PyTorch selective-scan Mamba block
│   └── factory.py
├── utils/
│   ├── parser.py           # Centralized command-line arguments
│   ├── losses.py           # Symmetric InfoNCE
│   ├── metrics.py          # PCC / MAE / R² / ICC and scatter plots
│   ├── lr_schedule.py      # Warm-up + polynomial LR decay
│   ├── checkpoint.py
│   └── engine.py           # Shared train/eval loops
├── train.py
├── test.py
└── requirements.txt
```

## Acknowledgement
The selective-scan implementation follows
[Mamba](https://github.com/state-spaces/mamba), the tabular Transformer is
adapted from [TIP](https://github.com/siyi-wind/TIP), and the multimodal
contrastive design draws on MM-CL and AlignMamba. We thank the authors for
open-sourcing their work.

## Citation
If this code is helpful, please cite our paper:
```
@article{himamba2026,
  title={HiMamba: A Hierarchical State Space Framework for Asymmetric
         Multimodal Data Mining in Clinical Assessment},
  author={Cheng, Biandi and Wang, Peng and Cui, Hui and Xuan, Ping and Cong, Cong
          and Su, Ran and Wei, Leyi and Jin, Qiangguo and Guo, Fei and Chen, Yupin},
  year={2026}
}
```
