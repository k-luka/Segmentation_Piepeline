# UNet Pet Segmentation

Learning project that trains a UNet to segment pet masks from the Oxford-IIIT Pet dataset. Hydra drives configuration, torchvision handles data loading/augmentations, and training/evaluation happens through the `Trainer` class in `src/train_or_eval.py`.

## Project Layout

- `src/main.py` – Hydra entrypoint that builds the model, dataloaders, optimizer/scheduler combo, and launches training/evaluation/inference.
- `src/train_or_eval.py` – Trainer utilities (Combined CE+Dice loss, training loop, metrics, inference snapshot saving).
- `src/dataset.py` – Oxford-IIIT Pet dataset wrapper with paired augmentations plus a simple inference dataset that returns both normalized tensors and display-ready images.
- `model/` – Network definition (`model/UNet.py`) and reusable UNet blocks.
- `conf/config.yaml` – All Hydra-configurable knobs (paths, hyperparameters, wandb project).

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA (or CPU/MPS fallback)
- torchvision >= 0.15 (for datasets/transforms)
- Hydra, OmegaConf, wandb, tqdm, matplotlib, Pillow

Install deps (example):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install hydra-core wandb tqdm matplotlib pillow
```

## Usage

### Train / Evaluate

```bash
cd UNet
python -m src.main
```

Training automatically evaluates every `train.eval_strat` epochs (see `conf/config.yaml`). When a new best dice score is found the script saves a checkpoint to `model.checkpoint` and writes qualitative inference masks to `model.result_dir`.

### Config Overrides

Hydra lets you override any config on the CLI:

```bash
python -m src.main train.num_epochs=40 train.lr=5e-5 data.batch_size=32
```

### Inference Notebook

`inference.ipynb` demonstrates running the trained checkpoint on custom images; it reuses the same inference dataloader defined in `src/dataset.py`.

## Notes

- Dataloader uses augmentations (pad-to-square, resize, random flip/affine) with bilinear interpolation on images and nearest-neighbor on masks to keep boundaries sharp.
- Loss is a tunable combo of CrossEntropy and Dice (`CombinedLoss`), controlled via `alpha`.
- Scheduler stacks a warmup `LinearLR` followed by `CosineAnnealingLR`; adjust in `src/main.py` if needed.
