# GC-MS Spectrum Restoration

This repository contains an unconditional masked autoencoder (MAE) pipeline for GC-MS spectrum restoration under synthetic degradation. The current project version focuses on two curated datasets:

- `dataset1`: clinical breathomics GC-MS data
- `dataset2`: plant volatile metabolomics GC-MS data

The codebase supports:

- preprocessing the current `dataset2`
- training the unconditional MAE with a frozen auxiliary classifier
- saving per-epoch training histories to CSV
- plotting MAE loss curves
- running focused `case3` ablation experiments

## Project Structure

Main files:

- [train_unconditional_mae.py](./train_unconditional_mae.py): main training entry for the unconditional MAE
- [model.py](./model.py): MAE model definition
- [dataset.py](./dataset.py): on-the-fly degraded sample construction
- [degradation.py](./degradation.py): degradation logic for case 0-3
- [data_pipeline.py](./data_pipeline.py): train/val/test split and dataloader helpers
- [data_sources.py](./data_sources.py): Excel/NPZ dataset loading
- [utils.py](./utils.py): weighted reconstruction loss, metrics, checkpoint utilities
- [preprocess_dataset2.py](./preprocess_dataset2.py): preprocess the current `dataset2`
- [plot_training_loss_history.py](./plot_training_loss_history.py): plot MAE loss curves from saved history CSV files
- [run_case3_ablation.py](./run_case3_ablation.py): run focused `case3` ablation experiments

Data directories:

- [massspec_data/dataset1](./massspec_data/dataset1)
- [massspec_data/dataset2](./massspec_data/dataset2)

Outputs:

- training checkpoints: [checkpoints](./checkpoints)
- generated analysis files: [analysis_outputs](./analysis_outputs)

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` uses the CUDA 12.6 PyTorch index.
- If you are using a different CUDA version or CPU-only PyTorch, adjust the `torch` installation method accordingly.
- `classifier.py` supports optional XGBoost usage, but `xgboost` is not required for the main training pipeline.

## Datasets

Current expected training files:

- `massspec_data/dataset1/dataset1_preprocessed.xlsx`
- `massspec_data/dataset2/dataset2_preprocessed.xlsx`

If you need to regenerate `dataset2`, run:

```bash
python preprocess_dataset2.py
```

## Training

Run `dataset1`:

```bash
python train_unconditional_mae.py --dataset_mode single --single_dataset_type excel --single_excel massspec_data/dataset1/dataset1_preprocessed.xlsx --save_dir checkpoints --mae_save_name dataset1_mae.pt --frozen_mlp_save_name dataset1_frozen_mlp.pt
```

Run `dataset2`:

```bash
python train_unconditional_mae.py --dataset_mode single --single_dataset_type excel --single_excel massspec_data/dataset2/dataset2_preprocessed.xlsx --save_dir checkpoints --mae_save_name dataset2_mae.pt --frozen_mlp_save_name dataset2_frozen_mlp.pt
```

Quick smoke run:

```bash
python train_unconditional_mae.py --dataset_mode single --single_dataset_type excel --single_excel massspec_data/dataset1/dataset1_preprocessed.xlsx --save_dir checkpoints --mae_save_name dataset1_mae.pt --frozen_mlp_save_name dataset1_frozen_mlp.pt --epochs 50 --mlp_epochs 70
```

Default split ratios:

- train: `0.6`
- val: `0.2`
- test: `0.2`

## Training Outputs

For each run, the training script saves:

- MAE checkpoint: `checkpoints/<name>.pt`
- frozen MLP checkpoint: `checkpoints/<name>.pt`
- MAE history CSV: `checkpoints/<mae_name_without_ext>_history.csv`
- frozen MLP history CSV: `checkpoints/<mlp_name_without_ext>_history.csv`

The MAE history CSV contains per-epoch fields such as:

- `train_loss`
- `train_recon_loss`
- `train_cls_loss_real`
- `train_cls_acc_real`
- `val_loss`
- `val_mse`
- `val_mae`
- `val_pearson`
- `val_degraded_mlp_acc`
- `val_recon_mlp_acc`
- `is_best`

These files are intended for plotting convergence curves and reporting training dynamics in the paper.

## Plot Loss Curves

Plot the current `dataset1` and `dataset2` MAE histories:

```bash
python plot_training_loss_history.py
```

Default output:

- [analysis_outputs/mae_training_dynamics.png](./analysis_outputs/mae_training_dynamics.png)

## Case 3 Ablation

Run the focused `case3` ablation:

```bash
python run_case3_ablation.py
```

This script compares:

- `baseline`
- `no_aux_cls`
- `unweighted_recon`

for both `dataset1` and `dataset2`, and writes:

- [analysis_outputs/case3_ablation_results.csv](./analysis_outputs/case3_ablation_results.csv)

## Notes On Current Design

- `case3` corresponds to the compound degradation setting with baseline drift, missing peaks, and overlapping peaks.
- The MAE uses a weighted reconstruction loss with additional emphasis on nonzero, degraded, and peak regions.
- A frozen auxiliary MLP classifier is used during training to encourage class-relevant structure preservation in the restored spectra.

## Suggested Workflow

1. Prepare or verify `dataset1` and `dataset2` Excel files.
2. Run training for each dataset.
3. Inspect `*_history.csv` files.
4. Generate the training-dynamics figure.
5. Run `case3` ablation if you need the paper table.

## Cleanup Reminder

The repository has already been cleaned to keep only the current training, preprocessing, and ablation workflow. If you add new one-off analysis scripts later, it is a good idea to keep them separate from the core training pipeline.
