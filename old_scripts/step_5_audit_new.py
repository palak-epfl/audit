"""
Step 5: Black-Box Fairness Auditing
=====================================
Three auditing modes:
  1. Full local   — auditor uses ALL its local data as queries (20 pairs)
  2. Budgeted     — auditor samples N queries, repeated 10x per budget (1,000 audits)
  3. Global       — trusted third-party auditor uses full dataset (5 targets)

Parallelised across GPUs: one target node per GPU. Each GPU loads one target
model and keeps it resident while all auditors query it sequentially.
This minimises model loading overhead.

Run with:
    python step5_audit.py
    python3 step_5_audit.py --config config.yaml

Env vars:
    NFS_ROOT=/your/nfs/path

Produces:
    results/step5_audit_results.json        ← all raw audit results
    plots/step5_full_estimated_vs_true.png
    plots/step5_full_error_heatmap.png
    plots/step5_budget_sample_efficiency.png
    plots/step5_budget_error_vs_mismatch.png
    plots/step5_global_vs_local.png
    plots/step5_ranking_accuracy.png
    plots/step5_distribution_pca_auditors.png
    logs/step5.log
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
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from scipy import stats as scipy_stats
import yaml

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Step 5: Black-Box Fairness Auditing')
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
AUDIT_SEED     = cfg['audit']['seed']
PARTITION_ATTR = cfg['partition']['partition_attr']

# Auditing config — extend config.yaml with these if not present
BUDGET_SIZES  = cfg['audit'].get('budget_sizes',  [100, 500, 1000, 2000, 5000])
NUM_REPEATS   = cfg['audit'].get('num_repeats',   10)

# ── NFS paths ──────────────────────────────────────────────────────────────────
NFS_ROOT  = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR   = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
HF_CACHE  = os.environ.get('HF_DATASETS_CACHE',
                            os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE

PLOT_DIR    = os.path.join(EXP_DIR, 'plots')
RESULTS_DIR = os.path.join(EXP_DIR, 'results')
CKPT_DIR    = os.path.join(EXP_DIR, 'checkpoints')
LOG_DIR     = os.path.join(EXP_DIR, 'logs')

for d in [PLOT_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

shutil.copy(args.config, os.path.join(EXP_DIR, 'config.yaml'))

NODE_COLORS = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, 'step5.log')
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
# Model & Dataset
# ─────────────────────────────────────────────────────────────────────────────
import torch.nn as nn

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
        all_smiling = np.array(hf_dataset[TARGET_ATTR],    dtype=np.int64)
        all_gender  = np.array(hf_dataset[SENSITIVE_ATTR], dtype=np.int64)
        self.smiling = all_smiling[self.indices]
        self.gender  = all_gender[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.data[int(real_idx)]['image'].convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.smiling[idx]


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────────────────────
# Core audit helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_dp_gap(preds, gender_labels):
    male_mask   = (gender_labels == 1)
    female_mask = (gender_labels == 0)
    p_male   = float(preds[male_mask].mean())   if male_mask.sum()   > 0 else 0.0
    p_female = float(preds[female_mask].mean()) if female_mask.sum() > 0 else 0.0
    return p_male, p_female, abs(p_male - p_female)


@torch.no_grad()
def get_predictions(model, dataset, indices, device, batch_size=256):
    """Run inference on a set of indices, return predictions as numpy array."""
    ds     = CelebADataset(dataset, indices, transform=eval_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    model.eval()
    all_preds = []
    for imgs, _ in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
    return np.array(all_preds), ds.gender


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for the mean of values.
    Returns (mean, lower, upper).
    """
    rng     = np.random.default_rng(seed)
    values  = np.array(values)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha)
    upper = np.percentile(boot_means, 100 * (1 - alpha))
    return float(values.mean()), float(lower), float(upper)


def run_single_audit(model, dataset, query_indices, query_gender,
                     true_dp_gap, device):
    """
    Run one audit: query target model with given indices,
    use query_gender labels to estimate DP gap.

    Returns dict with estimated DP gap and error metrics.
    """
    preds, _ = get_predictions(model, dataset, query_indices, device)
    est_p_male, est_p_female, est_dp = compute_dp_gap(preds, query_gender)

    abs_err = abs(est_dp - true_dp_gap)
    rel_err = abs_err / true_dp_gap if true_dp_gap > 0 else 0.0

    return {
        'est_p_male'   : est_p_male,
        'est_p_female' : est_p_female,
        'est_dp_gap'   : est_dp,
        'true_dp_gap'  : true_dp_gap,
        'abs_error'    : abs_err,
        'rel_error'    : rel_err,
        'n_queries'    : len(query_indices),
        'query_pct_male': float(query_gender.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Worker: audits one TARGET node (called in a separate process per GPU)
# ─────────────────────────────────────────────────────────────────────────────
def audit_target_node(target_id, gpu_id, dataset, node_indices,
                      true_dp_gaps, result_queue, cfg, nfs_paths):
    """
    Load target model onto gpu_id, then run all three audit modes
    for every auditor against this target.

    Sends one result dict per audit mode to result_queue.
    """
    tag = f'[Target {target_id} | GPU {gpu_id}]'

    try:
        device = torch.device(f'cuda:{gpu_id}')

        # Load target model
        ckpt_path = os.path.join(nfs_paths['ckpt_dir'],
                                 f'node_{target_id}_best.pt')
        model = LeNet5().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f'{tag} Model loaded from {ckpt_path}', flush=True)

        true_dp = true_dp_gaps[target_id]
        N       = len(dataset)
        global_indices = np.arange(N)
        global_gender  = np.array(dataset[cfg['dataset']['sensitive_attr']],
                                  dtype=np.int64)

        num_nodes    = cfg['partition']['num_nodes']
        budget_sizes = cfg['audit'].get('budget_sizes',  [100, 500, 1000, 2000, 5000])
        num_repeats  = cfg['audit'].get('num_repeats',   10)
        batch_size   = cfg['training']['batch_size']

        # ── Mode 1: Full local audits (all auditors → this target) ────────────
        full_results = []
        for auditor_id in range(1, num_nodes + 1):
            if auditor_id == target_id:
                continue
            auditor_indices = node_indices[auditor_id - 1]
            auditor_gender  = np.array(dataset[cfg['dataset']['sensitive_attr']],
                                       dtype=np.int64)[auditor_indices]

            r = run_single_audit(model, dataset, auditor_indices,
                                 auditor_gender, true_dp, device)
            r.update({
                'mode'       : 'full_local',
                'auditor_id' : auditor_id,
                'target_id'  : target_id,
            })
            full_results.append(r)
            print(f'{tag} Full local  | Auditor {auditor_id} → '
                  f'est_dp={r["est_dp_gap"]:.4f}  '
                  f'abs_err={r["abs_error"]:.4f}', flush=True)

        # ── Mode 2: Budgeted audits (all auditors, all budgets, all repeats) ──
        budget_results = []
        for auditor_id in range(1, num_nodes + 1):
            if auditor_id == target_id:
                continue
            auditor_indices = node_indices[auditor_id - 1]
            auditor_gender  = np.array(dataset[cfg['dataset']['sensitive_attr']],
                                       dtype=np.int64)[auditor_indices]
            n_available = len(auditor_indices)

            for budget in budget_sizes:
                actual_budget = min(budget, n_available)
                repeat_results = []

                for rep in range(num_repeats):
                    rng = np.random.default_rng(
                        cfg['audit']['seed'] + auditor_id * 1000
                        + target_id * 100 + rep)
                    sampled_local = rng.choice(
                        n_available, size=actual_budget, replace=False)
                    sampled_global = auditor_indices[sampled_local]
                    sampled_gender = auditor_gender[sampled_local]

                    r = run_single_audit(model, dataset, sampled_global,
                                         sampled_gender, true_dp, device)
                    r.update({
                        'mode'       : 'budgeted',
                        'auditor_id' : auditor_id,
                        'target_id'  : target_id,
                        'budget'     : budget,
                        'repeat'     : rep,
                        'actual_budget': actual_budget,
                    })
                    repeat_results.append(r)

                # Aggregate across repeats
                est_dps   = [r['est_dp_gap'] for r in repeat_results]
                abs_errs  = [r['abs_error']  for r in repeat_results]
                rel_errs  = [r['rel_error']  for r in repeat_results]

                mean_dp, ci_lo, ci_hi = bootstrap_ci(est_dps, seed=cfg['audit']['seed'])
                mean_abs, _, _        = bootstrap_ci(abs_errs, seed=cfg['audit']['seed'])
                mean_rel, _, _        = bootstrap_ci(rel_errs, seed=cfg['audit']['seed'])

                budget_agg = {
                    'mode'           : 'budgeted_agg',
                    'auditor_id'     : auditor_id,
                    'target_id'      : target_id,
                    'budget'         : budget,
                    'actual_budget'  : actual_budget,
                    'true_dp_gap'    : true_dp,
                    'mean_est_dp'    : mean_dp,
                    'std_est_dp'     : float(np.std(est_dps)),
                    'ci_lower'       : ci_lo,
                    'ci_upper'       : ci_hi,
                    'mean_abs_error' : mean_abs,
                    'mean_rel_error' : mean_rel,
                    'repeats'        : repeat_results,
                }
                budget_results.append(budget_agg)

                print(f'{tag} Budgeted    | Auditor {auditor_id} '
                      f'budget={budget} → '
                      f'mean_est_dp={mean_dp:.4f} ± {np.std(est_dps):.4f}  '
                      f'mean_abs_err={mean_abs:.4f}', flush=True)

        # ── Mode 3: Global audit (full dataset → this target) ─────────────────
        r_global = run_single_audit(model, dataset, global_indices,
                                    global_gender, true_dp, device)
        r_global.update({
            'mode'      : 'global',
            'auditor_id': 'global',
            'target_id' : target_id,
        })
        print(f'{tag} Global      | '
              f'est_dp={r_global["est_dp_gap"]:.4f}  '
              f'abs_err={r_global["abs_error"]:.4f}', flush=True)

        result_queue.put(('success', target_id, {
            'full'  : full_results,
            'budget': budget_results,
            'global': r_global,
        }))

    except Exception as e:
        import traceback
        print(f'{tag} ERROR: {e}', flush=True)
        traceback.print_exc()
        result_queue.put(('error', target_id, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_full_estimated_vs_true(full_results, plot_dir):
    """Scatter: estimated vs true DP gap, coloured by auditor node."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for auditor_id in range(1, 6):
        pairs = [r for r in full_results if r['auditor_id'] == auditor_id]
        if not pairs:
            continue
        est  = [r['est_dp_gap']  for r in pairs]
        true = [r['true_dp_gap'] for r in pairs]
        ax.scatter(true, est, color=NODE_COLORS[auditor_id-1],
                   label=f'Auditor {auditor_id}', s=100, zorder=3)

    lims = [0, max(r['true_dp_gap'] for r in full_results) * 1.3]
    ax.plot(lims, lims, 'k--', alpha=0.4, label='Perfect estimate')
    ax.set_xlabel('True DP Gap');  ax.set_ylabel('Estimated DP Gap')
    ax.set_title('Full Local Audit\nEstimated vs True DP Gap',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(plot_dir, 'step5_full_estimated_vs_true.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_full_error_heatmaps(full_results, num_nodes, plot_dir):
    """Abs + relative error heatmaps (auditor × target)."""
    abs_mat = np.full((num_nodes, num_nodes), np.nan)
    rel_mat = np.full((num_nodes, num_nodes), np.nan)
    for r in full_results:
        i = r['auditor_id'] - 1
        j = r['target_id']  - 1
        abs_mat[i, j] = r['abs_error']
        rel_mat[i, j] = r['rel_error']

    node_labels = [f'Node {i+1}' for i in range(num_nodes)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Full Local Audit — Error Heatmaps',
                 fontsize=13, fontweight='bold')

    for ax, mat, title, fmt in zip(
        axes,
        [abs_mat, rel_mat],
        ['Absolute Error', 'Relative Error'],
        ['{:.3f}', '{:.0%}']
    ):
        im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(num_nodes)); ax.set_xticklabels(node_labels, fontsize=9)
        ax.set_yticks(range(num_nodes)); ax.set_yticklabels(node_labels, fontsize=9)
        ax.set_xlabel('Target Node'); ax.set_ylabel('Auditor Node')
        ax.set_title(title)
        for i in range(num_nodes):
            for j in range(num_nodes):
                val = mat[i, j]
                txt = fmt.format(val) if not np.isnan(val) else '—'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = os.path.join(plot_dir, 'step5_full_error_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_budget_sample_efficiency(budget_results, plot_dir):
    """
    Sample efficiency curve: mean abs error vs query budget,
    one line per auditor-target pair, with ±1 std shading.
    Also shows a separate panel averaged across all pairs.
    """
    budget_sizes = sorted(set(r['budget'] for r in budget_results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Budgeted Audit — Sample Efficiency',
                 fontsize=13, fontweight='bold')

    # Left: one line per auditor-target pair
    ax = axes[0]
    pairs = set((r['auditor_id'], r['target_id']) for r in budget_results)
    for (aud, tgt) in sorted(pairs):
        pair_results = sorted(
            [r for r in budget_results
             if r['auditor_id'] == aud and r['target_id'] == tgt],
            key=lambda r: r['budget']
        )
        budgets   = [r['budget']         for r in pair_results]
        mean_errs = [r['mean_abs_error'] for r in pair_results]
        std_errs  = [r['std_est_dp']     for r in pair_results]
        color = NODE_COLORS[aud - 1]
        ax.plot(budgets, mean_errs, color=color, alpha=0.5,
                linewidth=1, marker='o', markersize=3)

    ax.set_xlabel('Query Budget'); ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Per Pair (coloured by auditor)')
    ax.set_xscale('log')
    ax.spines[['top','right']].set_visible(False)

    # Right: averaged across all pairs
    ax = axes[1]
    mean_across_pairs = []
    std_across_pairs  = []
    for budget in budget_sizes:
        errs = [r['mean_abs_error']
                for r in budget_results if r['budget'] == budget]
        mean_across_pairs.append(np.mean(errs))
        std_across_pairs.append(np.std(errs))

    mean_arr = np.array(mean_across_pairs)
    std_arr  = np.array(std_across_pairs)
    ax.plot(budget_sizes, mean_arr, color='steelblue',
            linewidth=2, marker='o', label='Mean across all pairs')
    ax.fill_between(budget_sizes, mean_arr - std_arr,
                    mean_arr + std_arr, alpha=0.2, color='steelblue')
    ax.set_xlabel('Query Budget'); ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Averaged Across All Pairs\n(shading = ±1 std)')
    ax.set_xscale('log')
    ax.legend()
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plot_dir, 'step5_budget_sample_efficiency.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_budget_error_vs_mismatch(budget_results, node_stats, plot_dir):
    """
    At each budget size: scatter of mean abs error vs gender distribution
    mismatch between auditor and target. Shows if mismatch hurts more
    when budget is small.
    """
    budget_sizes = sorted(set(r['budget'] for r in budget_results))
    n_budgets    = len(budget_sizes)
    ncols        = min(3, n_budgets)
    nrows        = (n_budgets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle('Does Distribution Mismatch Hurt More at Low Budget?',
                 fontsize=13, fontweight='bold')

    for idx, budget in enumerate(budget_sizes):
        ax   = axes[idx // ncols][idx % ncols]
        rows = [r for r in budget_results if r['budget'] == budget]

        mismatches = []
        errors     = []
        colors     = []
        for r in rows:
            aud_pct_male = node_stats[r['auditor_id'] - 1]['pct_male']
            tgt_pct_male = node_stats[r['target_id']  - 1]['pct_male']
            mismatches.append(abs(aud_pct_male - tgt_pct_male))
            errors.append(r['mean_abs_error'])
            colors.append(NODE_COLORS[r['auditor_id'] - 1])

        ax.scatter(mismatches, errors, c=colors, s=60, zorder=3)
        if len(mismatches) > 1:
            z = np.polyfit(mismatches, errors, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(mismatches), max(mismatches), 100)
            ax.plot(xs, p(xs), 'k--', alpha=0.5)
            # Pearson correlation
            corr, pval = scipy_stats.pearsonr(mismatches, errors)
            ax.set_title(f'Budget = {budget}\nr={corr:.2f}, p={pval:.3f}',
                         fontsize=10)
        else:
            ax.set_title(f'Budget = {budget}', fontsize=10)

        ax.set_xlabel('|% Male (Auditor) − % Male (Target)|', fontsize=8)
        ax.set_ylabel('Mean Abs Error', fontsize=8)
        ax.spines[['top','right']].set_visible(False)

    # Hide unused subplots
    for idx in range(n_budgets, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plot_dir, 'step5_budget_error_vs_mismatch.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_global_vs_local(full_results, global_results, plot_dir):
    """
    For each target node: compare global audit estimate vs all local
    auditor estimates. Shows whether a global auditor is better/worse
    than local auditors.
    """
    target_ids = sorted(set(r['target_id'] for r in global_results))
    n_targets  = len(target_ids)

    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 5),
                             sharey=True)
    if n_targets == 1:
        axes = [axes]
    fig.suptitle('Global Auditor vs Local Auditors per Target Node',
                 fontsize=13, fontweight='bold')

    for ax, target_id in zip(axes, target_ids):
        # Local estimates for this target
        local = [r for r in full_results if r['target_id'] == target_id]
        local_ests    = [r['est_dp_gap'] for r in local]
        local_aud_ids = [r['auditor_id'] for r in local]

        # Global estimate
        glob = next(r for r in global_results if r['target_id'] == target_id)
        true_dp = glob['true_dp_gap']

        # Plot
        x_local = range(len(local))
        bars = ax.bar(x_local, local_ests,
                      color=[NODE_COLORS[i-1] for i in local_aud_ids],
                      edgecolor='white', linewidth=1.2, label='Local auditors')
        ax.axhline(true_dp, color='black', linestyle='--',
                   linewidth=1.5, label=f'True DP ({true_dp:.3f})')
        ax.axhline(glob['est_dp_gap'], color='red', linestyle='-.',
                   linewidth=1.5, label=f'Global ({glob["est_dp_gap"]:.3f})')

        ax.set_title(f'Target: Node {target_id}', fontweight='bold')
        ax.set_xlabel('Auditor Node')
        ax.set_xticks(list(x_local))
        ax.set_xticklabels([f'N{i}' for i in local_aud_ids])
        ax.set_ylim(0, max(local_ests + [true_dp, glob['est_dp_gap']]) * 1.3)
        ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)

    axes[0].set_ylabel('Estimated DP Gap')
    plt.tight_layout()
    out = os.path.join(plot_dir, 'step5_global_vs_local.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_ranking_accuracy(full_results, budget_results, global_results,
                          true_dp_gaps, plot_dir):
    """
    Ranking accuracy: does the auditor correctly rank target nodes
    by their DP gap?

    For each auditor (including global), compute Spearman rank correlation
    between estimated DP gaps and true DP gaps across all targets.
    Show per mode and per budget.
    """
    target_ids = sorted(true_dp_gaps.keys())
    true_ranks = [true_dp_gaps[t] for t in target_ids]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Ranking Accuracy — Does the Auditor Correctly Rank Nodes by DP Gap?',
                 fontsize=12, fontweight='bold')

    # ── Panel 1: Full local — one bar per auditor ──────────────────────────────
    ax = axes[0]
    auditor_ids  = sorted(set(r['auditor_id'] for r in full_results))
    spearman_full = []
    for aud_id in auditor_ids:
        aud_results = sorted(
            [r for r in full_results if r['auditor_id'] == aud_id],
            key=lambda r: r['target_id']
        )
        est_dps = [r['est_dp_gap'] for r in aud_results]
        # Only include targets this auditor actually audited
        true_subset = [true_dp_gaps[r['target_id']] for r in aud_results]
        if len(est_dps) > 1:
            corr, _ = scipy_stats.spearmanr(true_subset, est_dps)
        else:
            corr = 0.0
        spearman_full.append(corr)

    bars = ax.bar([f'Node {i}' for i in auditor_ids], spearman_full,
                  color=[NODE_COLORS[i-1] for i in auditor_ids],
                  edgecolor='white', linewidth=1.2)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    ax.axhline(0.0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.set_title('Full Local Audit\nSpearman ρ per Auditor')
    ax.set_ylabel('Spearman ρ (rank correlation)')
    ax.set_ylim(-1, 1.2)
    for bar, v in zip(bars, spearman_full):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + 0.03 if v >= 0 else v - 0.1,
                f'{v:.2f}', ha='center', fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # ── Panel 2: Budgeted — spearman vs budget size, averaged across auditors ──
    ax = axes[1]
    budget_sizes_sorted = sorted(set(r['budget'] for r in budget_results))
    mean_spearman_per_budget = []
    std_spearman_per_budget  = []

    for budget in budget_sizes_sorted:
        budget_spear = []
        for aud_id in auditor_ids:
            aud_bud = sorted(
                [r for r in budget_results
                 if r['auditor_id'] == aud_id and r['budget'] == budget],
                key=lambda r: r['target_id']
            )
            if not aud_bud:
                continue
            est_dps     = [r['mean_est_dp']            for r in aud_bud]
            true_subset = [true_dp_gaps[r['target_id']] for r in aud_bud]
            if len(est_dps) > 1:
                corr, _ = scipy_stats.spearmanr(true_subset, est_dps)
                budget_spear.append(corr)
        mean_spearman_per_budget.append(np.mean(budget_spear))
        std_spearman_per_budget.append(np.std(budget_spear))

    mean_arr = np.array(mean_spearman_per_budget)
    std_arr  = np.array(std_spearman_per_budget)
    ax.plot(budget_sizes_sorted, mean_arr, color='steelblue',
            linewidth=2, marker='o')
    ax.fill_between(budget_sizes_sorted,
                    mean_arr - std_arr, mean_arr + std_arr,
                    alpha=0.2, color='steelblue')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    ax.axhline(0.0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Query Budget')
    ax.set_ylabel('Mean Spearman ρ')
    ax.set_title('Budgeted Audit\nRanking Accuracy vs Query Budget')
    ax.set_ylim(-1, 1.2)
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    # ── Panel 3: Global auditor spearman ──────────────────────────────────────
    ax = axes[2]
    global_sorted = sorted(global_results, key=lambda r: r['target_id'])
    glob_est  = [r['est_dp_gap']              for r in global_sorted]
    glob_true = [true_dp_gaps[r['target_id']] for r in global_sorted]
    if len(glob_est) > 1:
        glob_corr, glob_pval = scipy_stats.spearmanr(glob_true, glob_est)
    else:
        glob_corr, glob_pval = 0.0, 1.0

    ax.scatter(glob_true, glob_est, color='black', s=120, zorder=3)
    lims = [0, max(glob_true + glob_est) * 1.3]
    ax.plot(lims, lims, 'k--', alpha=0.4, label='Perfect')
    for r in global_sorted:
        ax.annotate(f"Node {r['target_id']}",
                    xy=(true_dp_gaps[r['target_id']], r['est_dp_gap']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.set_xlabel('True DP Gap'); ax.set_ylabel('Global Estimated DP Gap')
    ax.set_title(f'Global Auditor\nSpearman ρ={glob_corr:.2f}  p={glob_pval:.3f}')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plot_dir, 'step5_ranking_accuracy.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("  Step 5: Black-Box Fairness Auditing")
    log.info("=" * 70)
    log.info(f"\n  Config         : {args.config}")
    log.info(f"  Experiment     : {EXP_NAME}")
    log.info(f"  NFS root       : {NFS_ROOT}")
    log.info(f"  Audit modes    : full_local, budgeted, global")
    log.info(f"  Budget sizes   : {BUDGET_SIZES}")
    log.info(f"  Repeats/budget : {NUM_REPEATS}")
    log.info(f"  Log file       : {log_path}")

    n_gpus = torch.cuda.device_count()
    log.info(f"\n  GPUs available : {n_gpus}")
    for i in range(n_gpus):
        log.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")

    total_audits = (
        NUM_NODES * (NUM_NODES - 1) +                           # full local
        NUM_NODES * (NUM_NODES - 1) * len(BUDGET_SIZES) * NUM_REPEATS +  # budgeted
        NUM_NODES                                               # global
    )
    log.info(f"\n  Total audits   : {total_audits:,}")
    log.info(f"    Full local   : {NUM_NODES*(NUM_NODES-1)}")
    log.info(f"    Budgeted     : {NUM_NODES*(NUM_NODES-1)*len(BUDGET_SIZES)*NUM_REPEATS:,}")
    log.info(f"    Global       : {NUM_NODES}")

    # ── Load dataset ───────────────────────────────────────────────────────────
    log.info(f"\n  Loading CelebA...")
    dataset = load_dataset(DATASET_NAME, split='train')
    N = len(dataset)
    log.info(f"  ✓ {N:,} samples loaded")

    # ── Load partition ─────────────────────────────────────────────────────────
    partition_fname = f'partition_alpha{ALPHA}_seed{PART_SEED}_{PARTITION_ATTR}.json'
    partition_path  = os.path.join(EXP_DIR, 'partitions', partition_fname)
    with open(partition_path) as f:
        partition_data = json.load(f)
    node_indices = [np.array(idx, dtype=np.int64)
                    for idx in partition_data['node_indices']]
    log.info(f"  ✓ Partition loaded from {partition_path}")

    # ── Load true DP gaps from Step 4 results ─────────────────────────────────
    step4_path = os.path.join(RESULTS_DIR, 'step4_all_nodes_results.json')
    with open(step4_path) as f:
        step4 = json.load(f)
    true_dp_gaps = {r['node_id']: r['dp_gap_model'] for r in step4['nodes']}
    node_stats   = {r['node_id']: r               for r in step4['nodes']}

    log.info(f"\n  True DP gaps (model) from Step 4:")
    for node_id, dp in true_dp_gaps.items():
        log.info(f"    Node {node_id}: {dp:.4f}")

    # ── Load step 3 node stats for mismatch analysis ───────────────────────────
    step3_path = os.path.join(RESULTS_DIR, 'step3_partition_stats.json')
    with open(step3_path) as f:
        step3 = json.load(f)
    node_partition_stats = step3['nodes']   # list, index 0 = Node 1

    # ── Launch parallel auditing (one process per target node) ────────────────
    log.info(f"\n{'─'*70}")
    log.info(f"  Launching {NUM_NODES} audit workers "
             f"(max {min(NUM_NODES, n_gpus)} parallel)")
    log.info(f"{'─'*70}\n")

    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    active_procs = {}
    pending      = list(range(1, NUM_NODES + 1))
    all_target_results = {}
    start_time   = time.time()
    gpu_ids      = [i % n_gpus for i in range(NUM_NODES)]

    def launch_target(target_id):
        gpu_id = gpu_ids[target_id - 1]
        nfs_paths = {'ckpt_dir': CKPT_DIR,
                     'results_dir': RESULTS_DIR,
                     'plot_dir': PLOT_DIR}
        p = mp.Process(
            target=audit_target_node,
            args=(target_id, gpu_id, dataset, node_indices,
                  true_dp_gaps, result_queue, cfg, nfs_paths),
            name=f'Target-{target_id}'
        )
        p.start()
        active_procs[target_id] = p
        log.info(f"  ▶ Launched Target {target_id} on GPU {gpu_id}  (pid={p.pid})")

    # Launch first batch
    for _ in range(min(n_gpus, NUM_NODES)):
        if pending:
            launch_target(pending.pop(0))

    completed = 0
    while completed < NUM_NODES:
        status, target_id, payload = result_queue.get()
        proc = active_procs.pop(target_id)
        proc.join()
        completed += 1
        elapsed = time.time() - start_time

        if status == 'success':
            all_target_results[target_id] = payload
            log.info(f"\n  ✓ Target {target_id} finished  (elapsed={elapsed:.0f}s)")
        else:
            log.info(f"\n  ✗ Target {target_id} FAILED: {payload}")

        if pending:
            launch_target(pending.pop(0))

    total_time = time.time() - start_time
    log.info(f"\n  All targets finished in {total_time:.0f}s "
             f"({total_time/60:.1f} min)\n")

    # ── Flatten results ────────────────────────────────────────────────────────
    full_results   = []
    budget_results = []
    global_results = []

    for target_id, res in all_target_results.items():
        full_results.extend(res['full'])
        budget_results.extend(res['budget'])
        global_results.append(res['global'])

    if not full_results:
        log.info("  ✗ No successful audits — exiting")
        return

    # ── Summary statistics ─────────────────────────────────────────────────────
    log.info(f"{'─'*70}")
    log.info("  Full Local Audit — Summary")
    log.info(f"{'─'*70}\n")

    abs_errs = np.array([r['abs_error'] for r in full_results])
    rel_errs = np.array([r['rel_error'] for r in full_results])
    log.info(f"  {'Auditor':>8} → {'Target':<8} "
             f"{'Est DP':>8} {'True DP':>8} {'Abs Err':>9} {'Rel Err':>9}")
    log.info("  " + "-" * 58)
    for r in sorted(full_results, key=lambda x: (x['auditor_id'], x['target_id'])):
        log.info(f"  Node {r['auditor_id']:>2}    → Node {r['target_id']:<5}"
                 f" {r['est_dp_gap']:>8.4f} {r['true_dp_gap']:>8.4f}"
                 f" {r['abs_error']:>9.4f} {r['rel_error']:>8.1%}")

    log.info(f"\n  Overall — abs error: {abs_errs.mean():.4f} ± {abs_errs.std():.4f}"
             f"  |  rel error: {rel_errs.mean():.1%} ± {rel_errs.std():.1%}")

    log.info(f"\n{'─'*70}")
    log.info("  Global Audit — Summary")
    log.info(f"{'─'*70}\n")
    for r in sorted(global_results, key=lambda x: x['target_id']):
        log.info(f"  Global → Node {r['target_id']}  "
                 f"est_dp={r['est_dp_gap']:.4f}  "
                 f"true_dp={r['true_dp_gap']:.4f}  "
                 f"abs_err={r['abs_error']:.4f}  "
                 f"rel_err={r['rel_error']:.1%}")

    # ── Generate plots ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*70}")
    log.info("  Generating Plots")
    log.info(f"{'─'*70}\n")

    out = plot_full_estimated_vs_true(full_results, PLOT_DIR)
    log.info(f"  ✓ {out}")

    out = plot_full_error_heatmaps(full_results, NUM_NODES, PLOT_DIR)
    log.info(f"  ✓ {out}")

    out = plot_budget_sample_efficiency(budget_results, PLOT_DIR)
    log.info(f"  ✓ {out}")

    out = plot_budget_error_vs_mismatch(
        budget_results, node_partition_stats, PLOT_DIR)
    log.info(f"  ✓ {out}")

    out = plot_global_vs_local(full_results, global_results, PLOT_DIR)
    log.info(f"  ✓ {out}")

    out = plot_ranking_accuracy(
        full_results, budget_results, global_results, true_dp_gaps, PLOT_DIR)
    log.info(f"  ✓ {out}")

    # ── Save results JSON ──────────────────────────────────────────────────────
    results = {
        'experiment'      : EXP_NAME,
        'alpha'           : ALPHA,
        'num_nodes'       : NUM_NODES,
        'budget_sizes'    : BUDGET_SIZES,
        'num_repeats'     : NUM_REPEATS,
        'total_time_s'    : total_time,
        'true_dp_gaps'    : true_dp_gaps,
        'full_results'    : full_results,
        'budget_results'  : budget_results,
        'global_results'  : global_results,
        'summary': {
            'full_mean_abs_error': float(abs_errs.mean()),
            'full_std_abs_error' : float(abs_errs.std()),
            'full_mean_rel_error': float(rel_errs.mean()),
            'full_std_rel_error' : float(rel_errs.std()),
        }
    }
    results_path = os.path.join(RESULTS_DIR, 'step5_audit_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  ✓ Results saved → {results_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("  Step 5 Complete")
    log.info(f"{'='*70}")
    log.info(f"\n  Total audits run   : {total_audits:,}")
    log.info(f"  Total time         : {total_time:.0f}s ({total_time/60:.1f} min)")
    log.info(f"  Full local abs err : {abs_errs.mean():.4f} ± {abs_errs.std():.4f}")
    log.info(f"  Full local rel err : {rel_errs.mean():.1%} ± {rel_errs.std():.1%}")
    log.info(f"\n  Outputs saved to: {EXP_DIR}")
    log.info(f"    plots/step5_full_estimated_vs_true.png")
    log.info(f"    plots/step5_full_error_heatmap.png")
    log.info(f"    plots/step5_budget_sample_efficiency.png")
    log.info(f"    plots/step5_budget_error_vs_mismatch.png")
    log.info(f"    plots/step5_global_vs_local.png")
    log.info(f"    plots/step5_ranking_accuracy.png")
    log.info(f"    results/step5_audit_results.json")
    log.info(f"    logs/step5.log")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()