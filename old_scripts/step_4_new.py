"""
Step 4: Train 5 Node Models in Parallel
=========================================
Trains one LeNet model per federated node using the non-IID partition
from Step 3. Uses torch.multiprocessing to run 4 nodes in parallel
across 4 GPUs, with the 5th queued to start as soon as any GPU frees up.

Dataset is loaded ONCE in the main process and shared across all worker
processes via HuggingFace's Arrow-backed shared memory — no redundant
copies in RAM.

Run with:
    python3 step_4_train_5_models_on_data_paritions.py --config config.yaml

Env vars:
    NFS_ROOT=/your/nfs/path   override nfs.root from config

Per-node outputs (saved as each node finishes):
    checkpoints/node_{i}_best.pt
    plots/step4_node_{i}_training_curves.png
    plots/step4_node_{i}_dp_gap.png
    results/step4_node_{i}_results.json

After all nodes finish:
    plots/step4_summary_dp_gaps.png
    plots/step4_summary_training_curves.png
    results/step4_all_nodes_results.json
    logs/step4.log
"""

import os
import sys
import json
import shutil
import logging
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import yaml

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Step 4: Train 5 Node Models in Parallel')
parser.add_argument('--config', type=str, default='config.yaml')
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

DATASET_NAME   = cfg['dataset']['name']
SENSITIVE_ATTR = cfg['dataset']['sensitive_attr']
TARGET_ATTR    = cfg['dataset']['target_attr']
EXP_NAME       = cfg['experiment']['name']
NUM_NODES      = cfg['partition']['num_nodes']
ALPHA          = cfg['partition']['alpha']
PART_SEED      = cfg['partition']['seed']
IMAGE_SIZE     = cfg['model']['image_size']
IN_CHANNELS    = cfg['model']['in_channels']
NUM_CLASSES    = cfg['model']['num_classes']
DROPOUT        = cfg['model']['dropout']
EPOCHS         = cfg['training']['epochs']
BATCH_SIZE     = cfg['training']['batch_size']
LR             = cfg['training']['lr']
SEED           = cfg['training']['seed']
PATIENCE       = cfg['training']['patience']
LR_FACTOR      = cfg['training']['lr_factor']
LR_PATIENCE    = cfg['training']['lr_patience']
NUM_WORKERS    = cfg['training']['num_workers']
PARTITION_ATTR = cfg['partition']['partition_attr']   # ← new: drives the split


# ── NFS paths ──────────────────────────────────────────────────────────────────
NFS_ROOT  = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR   = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
HF_CACHE  = os.environ.get('HF_DATASETS_CACHE',
                            os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE

PLOT_DIR      = os.path.join(EXP_DIR, 'plots')
RESULTS_DIR   = os.path.join(EXP_DIR, 'results')
PARTITION_DIR = os.path.join(EXP_DIR, 'partitions')
CKPT_DIR      = os.path.join(EXP_DIR, 'checkpoints')
LOG_DIR       = os.path.join(EXP_DIR, 'logs')

for d in [PLOT_DIR, RESULTS_DIR, CKPT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

shutil.copy(args.config, os.path.join(EXP_DIR, 'config.yaml'))

NODE_COLORS = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']

# ── Logging — main process only ────────────────────────────────────────────────
# Worker processes use print() with [Node X] prefix to avoid log handler
# conflicts across processes. Main process logger mirrors to file.
log_path = os.path.join(LOG_DIR, 'step4.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode='w')
    ]
)
log = logging.getLogger()


# ─────────────────────────────────────────────────────────────────────────────
# Model & Dataset definitions
# ─────────────────────────────────────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
                 dropout=DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5), nn.ReLU(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),          nn.ReLU(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 120), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(120, 84),           nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CelebADataset(Dataset):
    def __init__(self, hf_dataset, indices, transform=None):
        self.data      = hf_dataset
        self.indices   = np.array(indices, dtype=np.int64)
        self.transform = transform
        all_smiling    = np.array(hf_dataset[TARGET_ATTR],    dtype=np.int64)
        all_gender     = np.array(hf_dataset[SENSITIVE_ATTR], dtype=np.int64)
        self.smiling   = all_smiling[self.indices]
        self.gender    = all_gender[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.data[int(real_idx)]['image'].convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.smiling[idx]


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_dp_gap(preds, gender_labels):
    male_mask   = (gender_labels == 1)
    female_mask = (gender_labels == 0)
    p_male   = float(preds[male_mask].mean())   if male_mask.sum()   > 0 else 0.0
    p_female = float(preds[female_mask].mean()) if female_mask.sum() > 0 else 0.0
    return p_male, p_female, abs(p_male - p_female)


# ─────────────────────────────────────────────────────────────────────────────
# Per-node plots (saved as each node finishes)
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history, node_id, stopped_epoch, plot_dir):
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f'Node {node_id} — Training Curves', fontsize=13, fontweight='bold')

    axes[0].plot(epochs_range, history['train_loss'], label='Train', marker='o', markersize=3)
    axes[0].plot(epochs_range, history['val_loss'],   label='Val',   marker='o', markersize=3)
    axes[0].axvline(stopped_epoch, color='gray', linestyle='--',
                    alpha=0.6, label=f'Best epoch ({stopped_epoch})')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('CrossEntropy Loss')
    axes[0].legend(); axes[0].spines[['top','right']].set_visible(False)

    axes[1].plot(epochs_range, history['train_acc'], label='Train', marker='o', markersize=3)
    axes[1].plot(epochs_range, history['val_acc'],   label='Val',   marker='o', markersize=3)
    axes[1].axvline(stopped_epoch, color='gray', linestyle='--',
                    alpha=0.6, label=f'Best epoch ({stopped_epoch})')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[1].legend(); axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plot_dir, f'step4_node_{node_id}_training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_node_dp_gap(result, node_id, plot_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f'Node {node_id} — Demographic Parity Analysis',
                 fontsize=13, fontweight='bold')

    colors = ['steelblue', 'salmon']
    for ax, (p_m, p_f, gap, title) in zip(axes, [
        (result['p_smile_true_male'],  result['p_smile_true_female'],
         result['dp_gap_data'],        f'Ground Truth\nDP gap = {result["dp_gap_data"]:.4f}'),
        (result['p_smile_pred_male'],  result['p_smile_pred_female'],
         result['dp_gap_model'],       f'Model Predictions\nDP gap = {result["dp_gap_model"]:.4f}'),
    ]):
        bars = ax.bar(['Male', 'Female'], [p_m, p_f],
                      color=colors, edgecolor='white', linewidth=1.2)
        ax.set_title(title); ax.set_ylabel('P(Smiling = 1)')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
        for bar, v in zip(bars, [p_m, p_f]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plot_dir, f'step4_node_{node_id}_dp_gap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Worker function — runs in a separate process, one per node
# ─────────────────────────────────────────────────────────────────────────────
def train_node(node_id, gpu_id, dataset, node_indices, result_queue,
               cfg, nfs_paths):
    """
    Train one node's model on its local data partition.

    Args:
        node_id      : 1-indexed node identifier
        gpu_id       : which GPU to use (0-3)
        dataset      : shared HuggingFace dataset (Arrow-backed, read-only)
        node_indices : this node's data indices
        result_queue : mp.Queue to send results back to main process
        cfg          : full config dict
        nfs_paths    : dict of output directory paths
    """
    tag = f'[Node {node_id} | GPU {gpu_id}]'

    try:
        # ── Resume check — skip training if checkpoint + results already exist ──
        ckpt_path_check   = os.path.join(nfs_paths['ckpt_dir'],
                                         f'node_{node_id}_best.pt')
        results_path_check = os.path.join(nfs_paths['results_dir'],
                                          f'step4_node_{node_id}_results.json')

        if os.path.exists(ckpt_path_check) and os.path.exists(results_path_check):
            print(f'{tag} Checkpoint and results already exist — skipping training.',
                  flush=True)
            print(f'{tag}   Checkpoint : {ckpt_path_check}', flush=True)
            print(f'{tag}   Results    : {results_path_check}', flush=True)
            with open(results_path_check, 'r') as f:
                result = json.load(f)
            result_queue.put(('success', node_id, result))
            return

        device = torch.device(f'cuda:{gpu_id}')
        torch.manual_seed(cfg['training']['seed'] + node_id)
        np.random.seed(cfg['training']['seed'] + node_id)

        print(f'{tag} Starting training on {torch.cuda.get_device_name(gpu_id)}',
              flush=True)

        # ── Dataset split: 90% train / 10% val ────────────────────────────────
        n          = len(node_indices)
        val_size   = int(cfg['training']['val_split'] * n)
        rng        = np.random.default_rng(cfg['training']['seed'] + node_id)
        shuffled   = rng.permutation(n)
        val_idx    = node_indices[shuffled[:val_size]]
        train_idx  = node_indices[shuffled[val_size:]]

        train_ds = CelebADataset(dataset, train_idx, transform=train_transform)
        val_ds   = CelebADataset(dataset, val_idx,   transform=eval_transform)

        # num_workers=0 inside spawned processes — avoids /dev/shm exhaustion.
        # Each node is already its own process; nested DataLoader workers
        # would create 4*num_workers processes all competing for shared memory.
        train_loader = DataLoader(
            train_ds, batch_size=cfg['training']['batch_size'],
            shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg['training']['batch_size'],
            shuffle=False, num_workers=0, pin_memory=False
        )

        print(f'{tag} Train: {len(train_ds):,}  Val: {len(val_ds):,}', flush=True)

        # ── Model ──────────────────────────────────────────────────────────────
        model     = LeNet5().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=cfg['training']['lr_factor'],
            patience=cfg['training']['lr_patience'],
            verbose=False
        )

        # ── Training loop ──────────────────────────────────────────────────────
        history           = {'train_loss': [], 'train_acc': [],
                             'val_loss':   [], 'val_acc':   []}
        best_val_loss     = float('inf')
        epochs_no_improve = 0
        best_epoch        = 1
        ckpt_path         = os.path.join(nfs_paths['ckpt_dir'],
                                         f'node_{node_id}_best.pt')

        for epoch in range(1, cfg['training']['epochs'] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device)
            scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss     = val_loss
                epochs_no_improve = 0
                best_epoch        = epoch
                torch.save(model.state_dict(), ckpt_path)
                marker = ' ← best'
            else:
                epochs_no_improve += 1
                marker = ''

            print(f'{tag} Epoch {epoch:>3}/{cfg["training"]["epochs"]} | '
                  f'train_loss={train_loss:.4f} train_acc={train_acc:.2%} | '
                  f'val_loss={val_loss:.4f} val_acc={val_acc:.2%}{marker}',
                  flush=True)

            if epochs_no_improve >= cfg['training']['patience']:
                print(f'{tag} Early stopping at epoch {epoch}', flush=True)
                break

        # ── Load best checkpoint for evaluation ───────────────────────────────
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Evaluate on full node dataset (train + val combined)
        full_ds     = CelebADataset(dataset, node_indices, transform=eval_transform)
        full_loader = DataLoader(
            full_ds, batch_size=cfg['training']['batch_size'],
            shuffle=False, num_workers=0, pin_memory=False
        )
        preds, labels = get_predictions(model, full_loader, device)
        gender_labels = full_ds.gender

        acc = accuracy_score(labels, preds)
        p_m_pred, p_f_pred, dp_gap_model = compute_dp_gap(preds, gender_labels)
        p_m_true, p_f_true, dp_gap_data  = compute_dp_gap(labels, gender_labels)

        print(f'{tag} Done — acc={acc:.2%} | '
              f'dp_gap_model={dp_gap_model:.4f} | dp_gap_data={dp_gap_data:.4f}',
              flush=True)

        # ── Per-node plots ─────────────────────────────────────────────────────
        result = {
            'node_id'            : node_id,
            'gpu_id'             : gpu_id,
            'n_train'            : len(train_ds),
            'n_val'              : len(val_ds),
            'n_total'            : len(full_ds),
            'best_epoch'         : best_epoch,
            'best_val_loss'      : best_val_loss,
            'accuracy'           : float(acc),
            'p_smile_pred_male'  : p_m_pred,
            'p_smile_pred_female': p_f_pred,
            'dp_gap_model'       : dp_gap_model,
            'p_smile_true_male'  : p_m_true,
            'p_smile_true_female': p_f_true,
            'dp_gap_data'        : dp_gap_data,
            'history'            : history,
            'ckpt_path'          : ckpt_path,
        }

        curves_path = plot_training_curves(
            history, node_id, best_epoch, nfs_paths['plot_dir'])
        dp_path = plot_node_dp_gap(result, node_id, nfs_paths['plot_dir'])

        # Save per-node JSON immediately — survives if later nodes fail
        node_results_path = os.path.join(
            nfs_paths['results_dir'], f'step4_node_{node_id}_results.json')
        result_to_save = {k: v for k, v in result.items() if k != 'history'}
        result_to_save['history'] = history
        with open(node_results_path, 'w') as f:
            json.dump(result_to_save, f, indent=2)

        print(f'{tag} Saved → {curves_path}', flush=True)
        print(f'{tag} Saved → {dp_path}',     flush=True)
        print(f'{tag} Saved → {node_results_path}', flush=True)

        result_queue.put(('success', node_id, result))

    except Exception as e:
        import traceback
        print(f'{tag} ERROR: {e}', flush=True)
        traceback.print_exc()
        result_queue.put(('error', node_id, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# Summary plots — generated after all nodes finish
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_dp_gaps(all_results, dp_gap_global, plot_dir):
    """Side-by-side DP gap comparison: data vs model, per node."""
    node_ids  = [r['node_id']       for r in all_results]
    dp_data   = [r['dp_gap_data']   for r in all_results]
    dp_model  = [r['dp_gap_model']  for r in all_results]
    node_lbls = [f'Node {i}'        for i in node_ids]
    x = np.arange(len(node_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, dp_data,  width, label='Data DP gap',
                   color='lightcoral', edgecolor='white', linewidth=1.2)
    bars2 = ax.bar(x + width/2, dp_model, width, label='Model DP gap',
                   color='steelblue',  edgecolor='white', linewidth=1.2)
    ax.axhline(dp_gap_global, color='black', linestyle='--', linewidth=1.5,
               label=f'Global DP gap ({dp_gap_global:.4f})')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f'{h:.3f}', ha='center', fontsize=8)

    ax.set_title('Data vs Model DP Gap per Node\n'
                 '(bars above global line = node amplifies global bias)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('DP gap'); ax.set_xticks(x); ax.set_xticklabels(node_lbls)
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'step4_summary_dp_gaps.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_summary_training_curves(all_results, plot_dir):
    """All 5 nodes' val loss curves on a single axis for easy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — All Nodes', fontsize=13, fontweight='bold')

    for r in all_results:
        node_id = r['node_id']
        color   = NODE_COLORS[node_id - 1]
        epochs  = range(1, len(r['history']['val_loss']) + 1)
        axes[0].plot(epochs, r['history']['val_loss'],
                     label=f'Node {node_id}', color=color, linewidth=1.5)
        axes[1].plot(epochs, r['history']['val_acc'],
                     label=f'Node {node_id}', color=color, linewidth=1.5)

    axes[0].set_title('Validation Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].spines[['top','right']].set_visible(False)

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    axes[1].legend(); axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plot_dir, 'step4_summary_training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("  Step 4: Train 5 Node Models in Parallel")
    log.info("=" * 70)
    log.info(f"\n  Config         : {args.config}")
    log.info(f"  Experiment     : {EXP_NAME}")
    log.info(f"  NFS root       : {NFS_ROOT}")
    log.info(f"  Num nodes      : {NUM_NODES}")
    log.info(f"  Log file       : {log_path}")

    # ── Check GPUs ────────────────────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    log.info(f"\n  GPUs available : {n_gpus}")
    for i in range(n_gpus):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        log.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.0f} GB)")

    if n_gpus == 0:
        log.info("\n  ⚠️  No GPUs found — falling back to CPU (will be slow)")
        gpu_ids = [0] * NUM_NODES   # all use 'cpu' effectively
    else:
        # Round-robin assign GPUs to nodes
        # With 4 GPUs and 5 nodes: [0,1,2,3,0] — Node 5 shares GPU 0
        # The queue-based approach below ensures max n_gpus run at once
        gpu_ids = [i % n_gpus for i in range(NUM_NODES)]
        log.info(f"\n  GPU assignment : {dict(zip(range(1, NUM_NODES+1), gpu_ids))}")
        log.info(f"  Max parallel   : {min(NUM_NODES, n_gpus)} nodes at a time")

    # ── Load dataset ONCE in main process ─────────────────────────────────────
    log.info(f"\n  Loading CelebA once in main process (shared across workers)...")
    dataset = load_dataset(DATASET_NAME, split='train')
    # HuggingFace datasets are already memory-mapped via Arrow under the hood.
    # Do NOT call with_format('arrow') -- that breaks PIL image access in workers.
    log.info(f"  ✓ Loaded {len(dataset):,} samples (Arrow memory-mapped)")

    # ── Load partition ─────────────────────────────────────────────────────────
    partition_fname = f'partition_alpha{ALPHA}_seed{PART_SEED}_{PARTITION_ATTR}.json'
    partition_path  = os.path.join(PARTITION_DIR, partition_fname)
    log.info(f"\n  Loading partition from {partition_path}...")
    with open(partition_path, 'r') as f:
        partition_data = json.load(f)
    node_indices = [np.array(idx, dtype=np.int64)
                    for idx in partition_data['node_indices']]
    log.info(f"  ✓ Loaded partition (α={partition_data['alpha']}, "
             f"seed={partition_data['seed']})")
    for i, idx in enumerate(node_indices):
        log.info(f"    Node {i+1}: {len(idx):,} samples")

    # Load global DP gap for reference in summary plots
    results_path = os.path.join(RESULTS_DIR, 'step3_partition_stats.json')
    with open(results_path) as f:
        step3_results = json.load(f)
    dp_gap_global = step3_results['global']['dp_gap_data']

    # ── NFS paths dict to pass to workers ─────────────────────────────────────
    nfs_paths = {
        'plot_dir'   : PLOT_DIR,
        'results_dir': RESULTS_DIR,
        'ckpt_dir'   : CKPT_DIR,
    }

    # ── Launch parallel training ───────────────────────────────────────────────
    log.info(f"\n{'─'*70}")
    log.info(f"  Launching {NUM_NODES} node training jobs "
             f"(max {min(NUM_NODES, n_gpus)} parallel)")
    log.info(f"{'─'*70}\n")

    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    active_procs = {}   # {node_id: Process}
    pending      = list(range(1, NUM_NODES + 1))   # nodes yet to start
    all_results  = []
    start_time   = time.time()

    def launch_node(node_id):
        gpu_id = gpu_ids[node_id - 1]
        p = mp.Process(
            target=train_node,
            args=(node_id, gpu_id, dataset, node_indices[node_id-1],
                  result_queue, cfg, nfs_paths),
            name=f'Node-{node_id}'
        )
        p.start()
        active_procs[node_id] = p
        log.info(f"  ▶ Launched Node {node_id} on GPU {gpu_id}  (pid={p.pid})")

    # Launch first batch (up to n_gpus nodes simultaneously)
    initial_batch = min(n_gpus, NUM_NODES)
    for _ in range(initial_batch):
        if pending:
            launch_node(pending.pop(0))

    # Collect results and launch queued nodes as GPUs free up
    completed = 0
    while completed < NUM_NODES:
        status, node_id, payload = result_queue.get()   # blocks until a node finishes

        proc = active_procs.pop(node_id)
        proc.join()
        completed += 1

        elapsed = time.time() - start_time
        if status == 'success':
            result = payload
            all_results.append(result)
            log.info(f"\n  ✓ Node {node_id} finished  "
                     f"(acc={result['accuracy']:.2%} | "
                     f"dp_gap_model={result['dp_gap_model']:.4f} | "
                     f"elapsed={elapsed:.0f}s)")
        else:
            log.info(f"\n  ✗ Node {node_id} FAILED: {payload}")

        # Launch next pending node on the freed GPU
        if pending:
            launch_node(pending.pop(0))

    total_time = time.time() - start_time
    log.info(f"\n  All nodes finished in {total_time:.0f}s "
             f"({total_time/60:.1f} min)\n")

    if not all_results:
        log.info("  ✗ All nodes failed — no results to summarise")
        return

    # Sort by node_id for consistent ordering
    all_results.sort(key=lambda r: r['node_id'])

    # ── Summary table ──────────────────────────────────────────────────────────
    log.info(f"{'─'*70}")
    log.info("  Summary Table")
    log.info(f"{'─'*70}\n")
    log.info(f"  {'Node':<8} {'N':>7} {'Acc':>7} {'DP(data)':>10} "
             f"{'DP(model)':>11} {'Δ':>8} {'Best Ep':>8}")
    log.info("  " + "-" * 62)
    for r in all_results:
        delta = r['dp_gap_model'] - r['dp_gap_data']
        direction = '▲' if delta > 0 else '▼'
        log.info(f"  Node {r['node_id']:<4} {r['n_total']:>7,} "
                 f"{r['accuracy']:>6.2%} "
                 f"{r['dp_gap_data']:>10.4f} "
                 f"{r['dp_gap_model']:>11.4f} "
                 f"{direction}{abs(delta):>6.4f} "
                 f"{r['best_epoch']:>8}")
    log.info(f"\n  ▲ = model amplified data bias | ▼ = model reduced data bias")
    log.info(f"  Global DP gap (data): {dp_gap_global:.4f}")

    # ── Summary plots ──────────────────────────────────────────────────────────
    log.info(f"\n{'─'*70}")
    log.info("  Generating Summary Plots")
    log.info(f"{'─'*70}\n")

    out1 = plot_summary_dp_gaps(all_results, dp_gap_global, PLOT_DIR)
    log.info(f"  ✓ Saved → {out1}")

    out2 = plot_summary_training_curves(all_results, PLOT_DIR)
    log.info(f"  ✓ Saved → {out2}")

    # ── Save combined results JSON ─────────────────────────────────────────────
    combined = {
        'experiment'    : EXP_NAME,
        'alpha'         : ALPHA,
        'num_nodes'     : NUM_NODES,
        'total_time_s'  : total_time,
        'dp_gap_global' : dp_gap_global,
        'nodes'         : [{k: v for k, v in r.items() if k != 'history'}
                           for r in all_results],
    }
    combined_path = os.path.join(RESULTS_DIR, 'step4_all_nodes_results.json')
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    log.info(f"  ✓ Saved → {combined_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    dp_model_gaps = [r['dp_gap_model'] for r in all_results]
    dp_data_gaps  = [r['dp_gap_data']  for r in all_results]
    accs          = [r['accuracy']     for r in all_results]

    log.info(f"\n{'='*70}")
    log.info("  Step 4 Complete")
    log.info(f"{'='*70}")
    log.info(f"\n  Nodes trained      : {len(all_results)}/{NUM_NODES}")
    log.info(f"  Total time         : {total_time:.0f}s ({total_time/60:.1f} min)")
    log.info(f"  Accuracy range     : {min(accs):.2%} – {max(accs):.2%}")
    log.info(f"  DP gap (model)     : {min(dp_model_gaps):.4f} – "
             f"{max(dp_model_gaps):.4f}")
    log.info(f"  DP gap (data)      : {min(dp_data_gaps):.4f} – "
             f"{max(dp_data_gaps):.4f}")
    log.info(f"\n  Per-node outputs:")
    for r in all_results:
        log.info(f"    Node {r['node_id']}: checkpoints/node_{r['node_id']}_best.pt  |  "
                 f"plots/step4_node_{r['node_id']}_*.png  |  "
                 f"results/step4_node_{r['node_id']}_results.json")
    log.info(f"\n  Summary outputs:")
    log.info(f"    {out1}")
    log.info(f"    {out2}")
    log.info(f"    {combined_path}")
    log.info(f"    {log_path}")
    log.info(f"\n  Next → python step5_audit.py --config {args.config}")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
