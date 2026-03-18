# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Pipeline

Each step is run sequentially with `--config config.yaml`:

```bash
python3 step_1_explore_celeba.py --config config.yaml
python3 step_2_train_baseline.py --config config.yaml
python3 step_3_non_iid_partition_attribute.py --config config.yaml
python3 step_4_train_5_models_on_data_paritions.py --config config.yaml
python3 step_5_audit.py --config config.yaml
```

Later steps skip recomputation if outputs already exist (checkpoints + results JSON). To re-run a step, delete its outputs from `checkpoints/`, `results/`, or `plots/` under the NFS path.

All outputs go to the NFS path defined in `config.yaml` under `nfs.root`.

## Architecture

This is a **federated learning fairness auditing research pipeline** for CelebA. It studies how well cross-node black-box auditing can estimate demographic parity (DP) gaps.

**Dataset**: CelebA (flwrlabs/celeba), predicting `Smiling` (target) using `Male` as the sensitive attribute. `High_Cheekbones` drives the non-IID data partitioning.

**Model**: LeNet-5 variant (`step_2`, `step_4`) — two conv layers → FC layers → 2-class softmax, trained with early stopping and LR reduction.

### Pipeline Steps

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `step_1_explore_celeba.py` | Attribute statistics, DP gaps, correlation plots |
| 2 | `step_2_train_baseline.py` | Train single LeNet on full dataset |
| 3 | `step_3_non_iid_partition_attribute.py` | Dirichlet(α=0.5) partition across 5 nodes |
| 4 | `step_4_train_5_models_on_data_paritions.py` | Train one LeNet per node, 4 GPUs in parallel |
| 5 | `step_5_audit.py` | Cross-node black-box fairness auditing |

### Step 4 Parallelism

Nodes train on 4 GPUs in parallel using `torch.multiprocessing`. The 5th node queues until a GPU frees. The HuggingFace dataset is loaded **once** in the main process and shared as Arrow-backed memory — do not reload it in worker processes.

### Step 5 Auditing Modes

Three auditing modes run in a single execution:
1. **Full local** — each node auditor uses all its local data to query target models (20 auditor×target pairs)
2. **Budgeted** — auditor samples N queries, repeated 10× per budget level (~1,000 audits total)
3. **Global** — trusted third party uses the full dataset (5 target models)

Parallelized by target node (one per GPU); target model stays resident on GPU while auditors query sequentially.

**Key metric**: Estimated DP gap vs. three ground truths (data distribution, model-val, model-full). Errors computed with bootstrap CIs (1,000 samples).

## Configuration

`config.yaml` controls all experiment parameters. Key sections:

- `nfs.root` — shared NFS storage path for all outputs
- `dataset` — HuggingFace dataset, sensitive/target/partition attributes
- `model` — architecture hyperparameters (image size, channels, dropout)
- `training` — epochs, batch size, lr, early stopping patience
- `partition` — number of nodes, Dirichlet α, min samples per node
- `audit` — query budget, random seed