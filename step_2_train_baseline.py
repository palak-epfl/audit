"""
Step 2: Single LeNet Baseline
==============================
Trains one LeNet model on the full CelebA dataset to predict Smiling,
then evaluates accuracy and demographic parity on the test set.

Run with:
    python step2_train_baseline.py
    python step2_train_baseline.py --config path/to/config.yaml

Produces:
    outputs/step2_training_curves.png
    outputs/step2_dp_gap.png
    checkpoints/baseline_best.pt
    results/step2_results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datasets import load_dataset
import yaml

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Step 2: Train LeNet baseline on CelebA')
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to config.yaml (default: config.yaml)')
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Shortcuts
DATASET_NAME   = cfg['dataset']['name']
SENSITIVE_ATTR = cfg['dataset']['sensitive_attr']
TARGET_ATTR    = cfg['dataset']['target_attr']
IMAGE_SIZE     = cfg['model']['image_size']
IN_CHANNELS    = cfg['model']['in_channels']
NUM_CLASSES    = cfg['model']['num_classes']
DROPOUT        = cfg['model']['dropout']
EPOCHS         = cfg['training']['epochs']
BATCH_SIZE     = cfg['training']['batch_size']
LR             = cfg['training']['lr']
SEED           = cfg['training']['seed']
VAL_SPLIT      = cfg['training']['val_split']
PATIENCE       = cfg['training']['patience']
LR_FACTOR      = cfg['training']['lr_factor']
LR_PATIENCE    = cfg['training']['lr_patience']
NUM_WORKERS    = cfg['training']['num_workers']

# ── NFS-aware path setup ───────────────────────────────────────────────────────
# NFS_ROOT env var overrides config value — useful for cluster environments
# where the mount path may differ from what's in the config file.
NFS_ROOT    = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_NAME    = cfg['experiment']['name']
EXP_DIR     = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)

# HuggingFace cache is shared across all experiments — no need to re-download
# CelebA for every new experiment run
HF_CACHE    = os.environ.get('HF_DATASETS_CACHE',
                              os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE

# Per-experiment subdirectories
CKPT_DIR    = os.path.join(EXP_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(EXP_DIR, 'results')
OUTPUT_DIR  = os.path.join(EXP_DIR, 'plots')

for d in [HF_CACHE, CKPT_DIR, RESULTS_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Save a copy of the config into the experiment folder so you always know
# exactly what settings produced a given set of results
import shutil
shutil.copy(args.config, os.path.join(EXP_DIR, 'config.yaml'))

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Logging helper ─────────────────────────────────────────────────────────────
def log(msg=''):
    print(msg, flush=True)

# ── Print header ───────────────────────────────────────────────────────────────
log("=" * 60)
log("  Step 2: Single LeNet Baseline")
log("=" * 60)
log(f"\n  Config file    : {args.config}")
log(f"  Experiment     : {EXP_NAME}")
log(f"  NFS root       : {NFS_ROOT}")
log(f"  Experiment dir : {EXP_DIR}")
log(f"  HF cache       : {HF_CACHE}")
log(f"  Device         : {DEVICE}")
if DEVICE == 'cuda':
    log(f"  GPU            : {torch.cuda.get_device_name(0)}")
    log(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
log(f"  Epochs         : {EPOCHS}  (early stopping patience={PATIENCE})")
log(f"  Batch size     : {BATCH_SIZE}")
log(f"  Learning rate  : {LR}")
log(f"  Image size     : {IMAGE_SIZE}×{IMAGE_SIZE}")
log(f"  Sensitive attr : {SENSITIVE_ATTR}")
log(f"  Target attr    : {TARGET_ATTR}\n")

# ── Dataset ────────────────────────────────────────────────────────────────────
log("Loading CelebA...")
dataset = load_dataset(DATASET_NAME, split=cfg['dataset']['split'])
log(f"✓ Loaded {len(dataset):,} samples\n")

gender  = np.array(dataset[SENSITIVE_ATTR], dtype=np.int64)
smiling = np.array(dataset[TARGET_ATTR],    dtype=np.int64)

# ── Transforms ─────────────────────────────────────────────────────────────────
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

# ── PyTorch Dataset ────────────────────────────────────────────────────────────
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
        img      = self.data[int(real_idx)]['image'].convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.smiling[idx]


# ── Train / val / test split ───────────────────────────────────────────────────
from datasets import get_dataset_split_names
available_splits = get_dataset_split_names(DATASET_NAME)
log(f"Available splits: {available_splits}")

if 'test' in available_splits:
    ds_train = load_dataset(DATASET_NAME, split='train')
    ds_test  = load_dataset(DATASET_NAME, split='test')
    ds_val   = load_dataset(DATASET_NAME, split='valid') \
               if 'valid' in available_splits else None

    train_idx = np.arange(len(ds_train))
    test_idx  = np.arange(len(ds_test))

    if ds_val is not None:
        val_idx = np.arange(len(ds_val))
        train_ds = CelebADataset(ds_train, train_idx, transform=train_transform)
        val_ds   = CelebADataset(ds_val,   val_idx,   transform=eval_transform)
        test_ds  = CelebADataset(ds_test,  test_idx,  transform=eval_transform)
    else:
        # Carve val from train
        n         = len(train_idx)
        val_size  = int(VAL_SPLIT * n)
        rng       = np.random.default_rng(SEED)
        shuffled  = rng.permutation(n)
        val_idx   = train_idx[shuffled[:val_size]]
        train_idx = train_idx[shuffled[val_size:]]
        train_ds  = CelebADataset(ds_train, train_idx, transform=train_transform)
        val_ds    = CelebADataset(ds_train, val_idx,   transform=eval_transform)
        test_ds   = CelebADataset(ds_test,  np.arange(len(ds_test)), transform=eval_transform)
else:
    # Single split — manual 80/10/10
    n         = len(dataset)
    rng       = np.random.default_rng(SEED)
    shuffled  = rng.permutation(n)
    train_end = int(0.80 * n)
    val_end   = int(0.90 * n)
    train_idx = shuffled[:train_end]
    val_idx   = shuffled[train_end:val_end]
    test_idx  = shuffled[val_end:]
    train_ds  = CelebADataset(dataset, train_idx, transform=train_transform)
    val_ds    = CelebADataset(dataset, val_idx,   transform=eval_transform)
    test_ds   = CelebADataset(dataset, test_idx,  transform=eval_transform)

log(f"\n✓ Dataset splits:")
log(f"  Train : {len(train_ds):,}")
log(f"  Val   : {len(val_ds):,}")
log(f"  Test  : {len(test_ds):,}\n")

# ── Load full-dataset DP gap from Step 1 results (population reference) ───────
step1_stats_path = os.path.join(RESULTS_DIR, 'step1_stats.json')
dp_gap_full_male   = None
dp_gap_full_female = None
dp_gap_full        = None
if os.path.exists(step1_stats_path):
    with open(step1_stats_path) as _f:
        _s1 = json.load(_f)
    dp_gap_full_male   = _s1['train']['p_smile_given_male']
    dp_gap_full_female = _s1['train']['p_smile_given_female']
    dp_gap_full        = abs(dp_gap_full_male - dp_gap_full_female)
    log(f"  Full dataset Demographic Parity (from step1_stats.json): {dp_gap_full:.4f}")
else:
    log("  step1_stats.json not found — full dataset Demographic Parity will not be shown.")
    log("  (Run step1_explore.py first to generate it.)")

# ── DataLoaders ────────────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == 'cuda'))

# ── Model ──────────────────────────────────────────────────────────────────────
class LeNet5(nn.Module):
    """
    LeNet-5 adapted for 64×64 RGB input with Dropout.
    Architecture:
        Input (3, 64, 64)
        → Conv(6, 5x5) → ReLU → AvgPool(2x2)   → (6, 30, 30)
        → Conv(16, 5x5) → ReLU → AvgPool(2x2)  → (16, 13, 13)
        → Flatten                                → (2704,)
        → FC(120) → ReLU → Dropout
        → FC(84)  → ReLU → Dropout
        → FC(2)                                  → logits
    """
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


model     = LeNet5().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_FACTOR,
    patience=LR_PATIENCE, verbose=False
)

total_params = sum(p.numel() for p in model.parameters())
log(f"✓ LeNet5 instantiated")
log(f"  Total parameters: {total_params:,}\n")

# ── Training helpers ───────────────────────────────────────────────────────────
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
    all_preds = []
    all_labels = []
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

# ── Resume check — skip training if checkpoint already exists ─────────────────
ckpt_path    = os.path.join(CKPT_DIR, 'baseline_best.pt')
results_path = os.path.join(RESULTS_DIR, 'step2_baseline_results.json')
skip_training = os.path.exists(ckpt_path)

if skip_training:
    log("=" * 60)
    log("  Checkpoint already exists — skipping training.")
    log(f"    Checkpoint : {ckpt_path}")
    log("  Proceeding to evaluation and plots using existing checkpoint.")
    log("  Delete the checkpoint to force retraining.")
    log("=" * 60)

# ── Training loop ──────────────────────────────────────────────────────────────
history       = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
stopped_epoch = EPOCHS

if not skip_training:
    log("=" * 60)
    log(f"  Training  (max {EPOCHS} epochs, early stopping patience={PATIENCE})")
    log("=" * 60)
    log(f"\n  {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>9} | {'Val Acc':>8}")
    log("  " + "-" * 55)

    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            marker = '  ← best'
        else:
            epochs_no_improve += 1
            marker = f'  (no improve {epochs_no_improve}/{PATIENCE})'

        log(f"  {epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
            f"{val_loss:>9.4f} | {val_acc:>7.2%}{marker}")

        if epochs_no_improve >= PATIENCE:
            log(f"\n  ⏹  Early stopping at epoch {epoch}")
            stopped_epoch = epoch
            break

    log(f"\n✓ Training complete")
    log(f"  Best val loss  : {best_val_loss:.4f}")
    log(f"  Stopped epoch  : {stopped_epoch}")
    log(f"  Checkpoint     : {ckpt_path}\n")

# ── Evaluate best checkpoint on test set ──────────────────────────────────────
log("Loading best checkpoint for test evaluation...")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

test_preds, test_labels = get_predictions(model, test_loader, DEVICE)

acc  = accuracy_score(test_labels, test_preds)
prec = precision_score(test_labels, test_preds, zero_division=0)
rec  = recall_score(test_labels, test_preds, zero_division=0)
f1   = 2 * prec * rec / (prec + rec + 1e-9)

# Get gender labels for test set
if 'test' in available_splits and ds_val is not None:
    test_gender = test_ds.gender
else:
    test_gender = gender[test_idx]

# Model Demographic Parity gap on test set predictions
p_male_pred, p_female_pred, dp_gap_model = compute_dp_gap(test_preds, test_gender)

# Ground truth Demographic Parity gap on test set data
p_male_true, p_female_true, dp_gap_true = compute_dp_gap(test_labels, test_gender)

# ── Evaluate on training set ───────────────────────────────────────────────────
log("\n  Computing DP gap on training set...")
eval_train_ds     = CelebADataset(
    train_ds.data, train_ds.indices, transform=eval_transform)
eval_train_loader = DataLoader(eval_train_ds, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True)
train_preds, train_labels = get_predictions(model, eval_train_loader, DEVICE)
train_gender = train_ds.gender
p_male_train_pred, p_female_train_pred, dp_gap_train_model = compute_dp_gap(train_preds, train_gender)
p_male_train_true, p_female_train_true, dp_gap_train_true  = compute_dp_gap(train_labels, train_gender)
log(f"  Train Demographic Parity (data)  : {dp_gap_train_true:.4f}")
log(f"  Train Demographic Parity (model) : {dp_gap_train_model:.4f}")

log(f"\n── Test Set Results ────────────────────────────────────")
log(f"  Accuracy  : {acc:.4f}  ({acc:.1%})")
log(f"  Precision : {prec:.4f}")
log(f"  Recall    : {rec:.4f}")
log(f"  F1        : {f1:.4f}")
log(f"\n── Demographic Parity Analysis ─────────────────────────")
log(f"  Test males  : {(test_gender==1).sum():,}  |  females: {(test_gender==0).sum():,}")
log(f"\n  Ground truth:")
log(f"    P(Y=1 | Male)   = {p_male_true:.4f}")
log(f"    P(Y=1 | Female) = {p_female_true:.4f}")
log(f"    Demographic Parity (data)   = {dp_gap_true:.4f}")
log(f"\n  Model predictions:")
log(f"    P(Ŷ=1 | Male)   = {p_male_pred:.4f}")
log(f"    P(Ŷ=1 | Female) = {p_female_pred:.4f}")
log(f"    Demographic Parity (model)  = {dp_gap_model:.4f}")

if dp_gap_model > dp_gap_true:
    log(f"\n  → Model AMPLIFIES data imbalance (+{dp_gap_model - dp_gap_true:.4f})")
else:
    log(f"\n  → Model REDUCES data imbalance (-{dp_gap_true - dp_gap_model:.4f})")

# ── Plot 1: Training curves ────────────────────────────────────────────────────
epochs_range = range(1, len(history['train_loss']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('LeNet Baseline — Training Curves', fontsize=13, fontweight='bold')

axes[0].plot(epochs_range, history['train_loss'], label='Train', marker='o', markersize=3)
axes[0].plot(epochs_range, history['val_loss'],   label='Val',   marker='o', markersize=3)
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('CrossEntropy Loss')
axes[0].legend()
axes[0].spines[['top', 'right']].set_visible(False)

axes[1].plot(epochs_range, history['train_acc'], label='Train', marker='o', markersize=3)
axes[1].plot(epochs_range, history['val_acc'],   label='Val',   marker='o', markersize=3)
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
axes[1].legend()
axes[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'step2_training_curves.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
log(f"\n✓ Saved → {out1}")

# ── Plot 2: DP gap comparison — train set | test set | model predictions ──────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Demographic Parity Analysis\n'
             'Train set  |  Test set  |  Model predictions (test set)\n'
             'Dotted lines = full dataset reference (Step 1)',
             fontsize=12, fontweight='bold')

colors = ['steelblue', 'salmon']
panels = [
    (p_male_train_true, p_female_train_true, dp_gap_train_true,
     f'Ground Truth — Train Set\nDemographic Parity = {dp_gap_train_true:.4f}'),
    (p_male_true, p_female_true, dp_gap_true,
     f'Ground Truth — Test Set\nDemographic Parity = {dp_gap_true:.4f}'),
    (p_male_pred, p_female_pred, dp_gap_model,
     f'Model Predictions — Test Set\nDemographic Parity = {dp_gap_model:.4f}'),
]

for ax, (p_m, p_f, gap, title) in zip(axes, panels):
    bars = ax.bar(['Male', 'Female'], [p_m, p_f],
                  color=colors, edgecolor='white', linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('P(Smiling = 1)')
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, [p_m, p_f]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

    # Full dataset reference lines from Step 1
    if dp_gap_full is not None:
        ax.axhline(dp_gap_full_male,   color='steelblue', linestyle=':',
                   linewidth=1.5, alpha=0.7,
                   label=f'Full dataset male ({dp_gap_full_male:.3f})')
        ax.axhline(dp_gap_full_female, color='salmon', linestyle=':',
                   linewidth=1.5, alpha=0.7,
                   label=f'Full dataset female ({dp_gap_full_female:.3f})')
        ax.legend(fontsize=7, loc='lower right')

    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'step2_dp_gap.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
log(f"✓ Saved → {out2}")

# ── Save results JSON ──────────────────────────────────────────────────────────
results = {
    'step'          : 2,
    'config'        : args.config,
    'stopped_epoch' : stopped_epoch,
    'best_val_loss' : best_val_loss,
    'test': {
        'accuracy'  : float(acc),
        'precision' : float(prec),
        'recall'    : float(rec),
        'f1'        : float(f1),
    },
    'dp_gap': {
        'p_smile_train_true_male'   : float(p_male_train_true),
        'p_smile_train_true_female' : float(p_female_train_true),
        'dp_gap_train_true'         : float(dp_gap_train_true),
        'dp_gap_train_model'        : float(dp_gap_train_model),
        'p_smile_true_male'         : float(p_male_true),
        'p_smile_true_female'       : float(p_female_true),
        'dp_gap_true'               : float(dp_gap_true),
        'dp_gap_full_male'          : float(dp_gap_full_male)   if dp_gap_full is not None else None,
        'dp_gap_full_female'        : float(dp_gap_full_female) if dp_gap_full is not None else None,
        'dp_gap_full_dataset'       : float(dp_gap_full)        if dp_gap_full is not None else None,
        'p_smile_pred_male'   : float(p_male_pred),
        'p_smile_pred_female' : float(p_female_pred),
        'dp_gap_model'        : float(dp_gap_model),
    }
}

results_path = os.path.join(RESULTS_DIR, 'step2_baseline_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
log(f"✓ Saved → {results_path}")

# ── Final summary ──────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log(f"  Step 2 Complete")
log(f"{'='*60}")
log(f"  Test accuracy   : {acc:.1%}")
log(f"  Demographic Parity (model)  : {dp_gap_model:.4f}")
log(f"  Demographic Parity (data)   : {dp_gap_true:.4f}")
log(f"\n  Outputs:")
log(f"    {out1}")
log(f"    {out2}")
log(f"    {ckpt_path}")
log(f"    {results_path}")
log(f"\n  Next → python step3_partition.py --config {args.config}")
log(f"{'='*60}")
