# ClimDetect - Detecting Human "Fingerprints" on Climate Change

An AI model that looks at **one day's global climate snapshot** (temperature, humidity, precipitation on a 64×128 grid) and predicts that **year's Annual Global Mean Temperature (AGMT) anomaly** - testing whether the human-caused warming signal is detectable even in short-term, noisy daily weather.

---

## Overview

Weather is noisy day to day - any single day can be unusually hot, cold, wet, or dry by chance. Underneath that noise, human activity (burning fossil fuels) causes a slow, steady warming trend. This project asks: **can an AI spot that trend from just one random day's weather map**, without ever being told the year? If it can, that's evidence the human-caused "fingerprint" is detectable even in everyday weather, not just long-term averages - the same idea behind climate science's **detection and attribution** research.

## Dataset

- **Source:** [ClimDetect/ClimDetect](https://huggingface.co/datasets/ClimDetect/ClimDetect) on Hugging Face, built from CMIP6 climate model simulations.
- **Subset used:** the official `mini` split (23,548 rows) - further subsampled to 3,000 train / 500 validation / 500 test rows for fast iteration.
- **Input:** `(3, 64, 128)` array - temperature (`tas`), humidity (`huss`), precipitation (`pr`) on a global grid.
- **Target:** a single number - that sample's year's standardized AGMT anomaly.
- No database required - loaded directly via the `datasets` library from Parquet files.

## Approach

Two models are trained on identical data for a fair comparison:

| Model | Description |
|---|---|
| **Baseline** | Linear Regression on the flattened 24,576-pixel vector |
| **Main model** | A small CNN (3 conv blocks → global average pool → 2 linear layers) that preserves spatial structure |

A CNN was chosen over the original paper's Vision Transformer as a scoped, faster-to-train, easier-to-explain MVP architecture.

```
Input (3×64×128)
  → Conv2d(3→16) + ReLU + MaxPool
  → Conv2d(16→32) + ReLU + MaxPool
  → Conv2d(32→64) + ReLU + MaxPool
  → AdaptiveAvgPool2d(1)
  → Linear(64→32) + ReLU
  → Linear(32→1)
```

## Results

| Model | Test MSE | Test R² |
|---|---|---|
| Linear Regression (baseline) | **0.0255** | **0.9845** |
| CNN (main model) | **0.0357** | **0.9541** |

**Honest finding:** in this run, the simple baseline slightly outperformed the CNN - most likely because the CNN was trained on only 3,000 examples for 15 epochs, not enough for it to fully exploit spatial structure. This is reported transparently rather than hidden; see [Limitations](#limitations--future-work).

## Explainability

A gradient-based saliency map shows which pixels most influenced a given prediction. In our run, salient pixels were scattered across the whole map rather than concentrated in one region - suggesting the model draws on small, spread-out global patterns rather than a single shortcut, though the raw-gradient method itself is known to produce a speckled, noisy-looking map.

## Live Demo

The notebook includes an interactive cell (`DEMO_INDEX`) to pick any of the 500 held-out test examples - data the model never trained on - and compare its prediction to the real answer, live:

```python
DEMO_INDEX = 7  # change to any number 0-499 and re-run this one cell
```

## Repository Contents

| File | Description |
|---|---|
| `climdetect_mvp.ipynb` | Main notebook: data loading, baseline, CNN, training, evaluation, live demo, saliency map |
| `DOCUMENTATION.md` | Full technical write-up explaining every cell, concept, and result in depth |
| `ClimDetect_Presentation.pptx` | Slide deck for presenting the project |

## How to Run

1. Open `climdetect_mvp.ipynb` in [Google Colab](https://colab.research.google.com/).
2. (Optional) Runtime → Change runtime type → GPU, for faster training.
3. Run all cells top to bottom (`Shift + Enter` on each, in order).
4. To try the live demo, edit `DEMO_INDEX` in Section 8.5 and re-run just that cell.

## Limitations & Future Work

- Trained on a 3,000-row subsample, not the full 1.17M-row (or even full 23.5k-row `mini`) dataset.
- Only 15 epochs; validation loss fluctuated rather than settling smoothly.
- CNN did not beat the baseline in this run - likely a data-size effect, not an architecture flaw.
- Simple raw-gradient saliency rather than a smoother method like Grad-CAM.

**Future work:** train on the full `mini` split or beyond, fine-tune a pretrained Vision Transformer (the original paper's approach), add a learning-rate schedule or data augmentation, and try Grad-CAM for cleaner explainability maps.

## Credits

- Dataset: [ClimDetect/ClimDetect](https://huggingface.co/datasets/ClimDetect/ClimDetect) (arXiv:2408.15993)
- Built with PyTorch, scikit-learn, and the Hugging Face `datasets` library.
