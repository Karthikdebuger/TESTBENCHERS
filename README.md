# SemiCon AI Hackathon — AI-Based Restoration of Degraded Semiconductor Images

## Overview
Unified deep learning model for restoring degraded semiconductor inspection images.  
Handles **speckle noise**, **Gaussian noise**, and **low-resolution downsampling** in any order and combination.

**Architecture:** NAFNet backbone + self-supervised degradation conditioning + unrolled stages + global residual learning.

## Quick Start (Zero Back-and-Forth)

### 0. Download Pre-Trained Weights
The trained model weights (`model.pt`, 1.3GB) are hosted on Google Drive due to GitHub file size limits.
- **Download Link**: [Google Drive - model.pt](https://drive.google.com/drive/folders/1PS2Og-_RbREirUDsEa4ehLZ2ezrFTKtk?usp=sharing)
- Place the downloaded `model.pt` file exactly inside the `models/` directory of this repository before running.

### 1. Clone & Install
```bash
git clone <repo-url>
cd TeamSemicon_SemiCon_ImageRestoration
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
Place paired images in:
```
data/
├── train/
│   ├── noisy/    # degraded inputs
│   └── gt/       # ground truth
├── val/
│   ├── noisy/
│   └── gt/
└── test/
    ├── noisy/
    └── gt/
```
Supported formats: `.png`, `.tif`, `.tiff`, `.npy`, `.pt`, `.jpg`, `.bmp`

### Execution

The evaluation script `run.py` is compliant with the KLA Hackathon requirements. It natively supports `.npy` inference and handles out-of-distribution values via robust dynamic normalization.

```bash
# Run inference
python run.py <path/to/test/images> <path/to/output>
```

### Constraints Satisfied
* **End-to-End Processing**: Generates perfectly clamped `[0,1]` `.npy` grayscale output arrays.
* **Offline Execution**: Fully decoupled from API requirements.
* **TTA Ready**: Uses an 8-fold Test-Time Augmentation loop inside the execution script for maximum validation scores without retraining.

### 5. View Results
Restored images → `outputs/restored_test/`  
Metrics (PSNR/SSIM/LPIPS) → printed to console & saved to `outputs/metrics.json`

## Repository Structure
```
├── README.md                      # this file
├── requirements.txt               # pinned dependencies
├── run.py                         # execution script — the graded file
├── models/
│   └── model.pt                   # trained checkpoint (download via Drive link)
├── Restored_Test_Outputs/         # where model output .npy files are saved
├── configs/
│   └── train.yaml                 # hyperparameters
└── src/
    ├── model.py                   # NAFNet + conditioning + unrolled stages
    ├── dataset.py                 # loader, normalization
    ├── augment.py                 # randomized augmentation
    ├── losses.py                  # Charbonnier + SSIM + FFT
    ├── train.py                   # training loop
    └── utils.py                   # utilities
```

## Key Design Decisions
- **One unified model** — not an ensemble of specialists
- **Self-supervised conditioning** — no external degradation labels required
- **Global residual learning** — output = bilinear_upsample(input) + residual
- **Soft-clamped output** — sigmoid-scaled, not hard-clipped
- **Dynamic per-image normalization** — preserves out-of-range signals
- **Order-randomized augmentation** — directly addresses "any order" requirement

## Hardware Requirements
- GPU with ≥8GB VRAM recommended (auto-halves batch size on OOM)
- CPU fallback supported for evaluation

## License
Academic use — SEMICON India Hackathon 2026
