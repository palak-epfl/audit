"""
Step 5c: Black-Box Fairness Auditing — Extended Collaboration
==============================================================
Extends step_5_audit.py with pairwise AND triplet collaborative auditing.
All existing audit modes are retained unchanged.

Audit modes:
  1.   Full local          — each single node audits independently (full data)
  1.5  Collab pairs        — every pair of non-target nodes pools their full data
  1.6  Collab triples      — every triple of non-target nodes pools their full data
  2.   Budgeted single     — single auditors, sampled budget, 10 repeats
  2.5  Budgeted pairs      — pair coalitions, sampled from pooled data, 10 repeats
  2.6  Budgeted triples    — triple coalitions, sampled from pooled data, 10 repeats
  3a.  Global all          — trusted third-party with full dataset
  3b.  Global excl (full)  — full dataset minus target's own partition
  3c.  Global excl budget  — global_excl pool, sampled budget, 10 repeats

Skip logic:
  If step5c_audit_results_{PARTITION_ATTR}.json already exists and contains
  all expected keys, the GPU computation is skipped and only plots are
  regenerated from the saved JSON.

Run with:
    python3 step_5_audit_more_collaboration.py --config config.yaml

Env vars:
    NFS_ROOT=/your/nfs/path

Produces:
    results/{date}/step5c_audit_results_{attr}.json
    plots/{date}/step5c_collab_all_modes_{attr}.png
    plots/{date}/step5_*_{attr}.png          (same plots as step_5_audit.py)
    logs/{date}/step5c_{attr}.log
"""

import os
import sys
import json
import shutil
import logging
import argparse
import datetime
import time
from itertools import combinations
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
parser = argparse.ArgumentParser(
    description='Step 5c: Black-Box Fairness Auditing — Extended Collaboration')
parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--force', action='store_true',
                    help='Re-run all audits even if results JSON already exists')
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

DATASET_NAME   = cfg['dataset']['name']
SENSITIVE_ATTR = cfg['dataset']['sensitive_attr']
TARGET_ATTR    = cfg['dataset']['target_attr']
NUM_NODES      = cfg['partition']['num_nodes']
ALPHA          = cfg['partition']['alpha']
PART_SEED      = cfg['partition']['seed']
RUN_DATE       = cfg['experiment'].get('run_date') or datetime.date.today().isoformat()
EXP_NAME       = (cfg['experiment'].get('name')
                  or f"lenet_alpha{ALPHA}_{NUM_NODES}nodes_seed{PART_SEED}")
IMAGE_SIZE     = cfg['model']['image_size']
IN_CHANNELS    = cfg['model']['in_channels']
NUM_CLASSES    = cfg['model']['num_classes']
DROPOUT        = cfg['model']['dropout']
AUDIT_SEED     = cfg['audit']['seed']
PARTITION_ATTR = cfg['partition']['partition_attr']

BUDGET_SIZES = cfg['audit'].get('budget_sizes', [100, 500, 1000, 2000, 5000])
NUM_REPEATS  = cfg['audit'].get('num_repeats',  10)

# ── NFS paths ──────────────────────────────────────────────────────────────────
NFS_ROOT = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR  = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
HF_CACHE = os.environ.get('HF_DATASETS_CACHE',
                           os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE

PLOT_DIR    = os.path.join(EXP_DIR, 'plots',       RUN_DATE)
RESULTS_DIR = os.path.join(EXP_DIR, 'results',     RUN_DATE)
CKPT_DIR    = os.path.join(EXP_DIR, 'checkpoints', RUN_DATE)
LOG_DIR     = os.path.join(EXP_DIR, 'logs',        RUN_DATE)

for d in [PLOT_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

shutil.copy(args.config, os.path.join(EXP_DIR, 'config.yaml'))

NODE_COLORS = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, f'step5c_{PARTITION_ATTR}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode='w')
    ]
)
log = logging.getLogger()

# Results file written/read by this script
RESULTS_FNAME = f'step5c_audit_results_{PARTITION_ATTR}.json'

# Keys that must be present for the skip logic to accept an existing JSON
REQUIRED_KEYS = {
    'full_results', 'collab_pair_results', 'collab_triple_results',
    'budget_results', 'collab_pair_budget_results', 'collab_triple_budget_results',
    'global_all_results', 'global_excl_results', 'global_excl_budget_results',
}


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
    rng        = np.random.default_rng(seed)
    values     = np.array(values)
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
    preds, _ = get_predictions(model, dataset, query_indices, device)
    est_p_male, est_p_female, est_dp = compute_dp_gap(preds, query_gender)
    abs_err = abs(est_dp - true_dp_gap)
    rel_err = abs_err / true_dp_gap if true_dp_gap > 0 else 0.0
    return {
        'est_p_male'            : est_p_male,
        'est_p_female'          : est_p_female,
        'est_dp_gap'            : est_dp,
        'true_dp_gap_model_val' : true_dp_gap,
        'abs_error'             : abs_err,
        'rel_error'             : rel_err,
        'n_queries'             : len(query_indices),
        'query_pct_male'        : float(query_gender.mean()),
    }


def _add_gt_errors(r, true_dp_data, true_dp_model_full):
    est = r['est_dp_gap']
    r['abs_error_model_val']    = r['abs_error']
    r['rel_error_model_val']    = r['rel_error']
    r['true_dp_gap_data']       = true_dp_data
    r['abs_error_data']         = abs(est - true_dp_data)
    r['rel_error_data']         = r['abs_error_data'] / true_dp_data if true_dp_data > 0 else 0.0
    r['true_dp_gap_model_full'] = true_dp_model_full
    r['abs_error_model_full']   = abs(est - true_dp_model_full)
    r['rel_error_model_full']   = (r['abs_error_model_full'] / true_dp_model_full
                                   if true_dp_model_full > 0 else 0.0)


def _run_collab_audits(model, dataset, node_indices, target_id, other_nodes,
                       coalition_size, true_dp, true_dp_data, true_dp_model_full,
                       gender_attr, device, tag):
    """
    Run all C(len(other_nodes), coalition_size) collaborative audits for one
    target, pooling data from every coalition of the given size.
    Returns a list of result dicts.
    """
    results = []
    for coalition in combinations(other_nodes, coalition_size):
        idx_parts    = [node_indices[n - 1] for n in coalition]
        gender_parts = [
            np.array(dataset[gender_attr], dtype=np.int64)[idx]
            for idx in idx_parts
        ]
        combined_indices = np.concatenate(idx_parts)
        combined_gender  = np.concatenate(gender_parts)

        r = run_single_audit(model, dataset, combined_indices,
                             combined_gender, true_dp, device)
        _add_gt_errors(r, true_dp_data, true_dp_model_full)
        label = '+'.join(str(n) for n in coalition)
        r.update({
            'mode'        : f'collab_{coalition_size}',
            'auditor_ids' : list(coalition),
            'auditor_id'  : label,
            'target_id'   : target_id,
        })
        results.append(r)
        print(f'{tag} Collab {coalition_size}-node | ({label}) → '
              f'n={len(combined_indices):,}  '
              f'est_dp={r["est_dp_gap"]:.4f}  '
              f'abs_err(model_val)={r["abs_error"]:.4f}  '
              f'abs_err(data)={r["abs_error_data"]:.4f}', flush=True)
    return results


def _run_collab_budget_audits(model, dataset, node_indices, target_id, other_nodes,
                              coalition_size, budget_sizes, num_repeats,
                              true_dp, true_dp_data, true_dp_model_full,
                              gender_attr, device, tag, audit_seed):
    """
    Budgeted collaborative audits for all coalitions of a given size.
    For each (coalition, budget): sample from the combined pool, repeat
    num_repeats times with different seeds, aggregate with bootstrap CI.
    """
    results = []
    for coalition in combinations(other_nodes, coalition_size):
        idx_parts    = [node_indices[n - 1] for n in coalition]
        gender_parts = [np.array(dataset[gender_attr], dtype=np.int64)[idx]
                        for idx in idx_parts]
        combined_indices = np.concatenate(idx_parts)
        combined_gender  = np.concatenate(gender_parts)
        n_available      = len(combined_indices)
        label            = '+'.join(str(n) for n in coalition)
        # Unique seed offset per coalition so repeats differ across coalitions
        coalition_seed   = sum(n * (10 ** i) for i, n in enumerate(sorted(coalition)))

        for budget in budget_sizes:
            actual_budget  = min(budget, n_available)
            repeat_results = []

            for rep in range(num_repeats):
                rng = np.random.default_rng(
                    audit_seed + coalition_seed + target_id * 100 + rep)
                sampled   = rng.choice(n_available, size=actual_budget, replace=False)
                s_indices = combined_indices[sampled]
                s_gender  = combined_gender[sampled]

                r = run_single_audit(model, dataset, s_indices, s_gender,
                                     true_dp, device)
                _add_gt_errors(r, true_dp_data, true_dp_model_full)
                r.update({
                    'mode'         : f'collab_{coalition_size}_budgeted',
                    'auditor_ids'  : list(coalition),
                    'auditor_id'   : label,
                    'target_id'    : target_id,
                    'budget'       : budget,
                    'repeat'       : rep,
                    'actual_budget': actual_budget,
                })
                repeat_results.append(r)

            est_dps       = [r['est_dp_gap']           for r in repeat_results]
            abs_errs      = [r['abs_error']            for r in repeat_results]
            abs_errs_data = [r['abs_error_data']       for r in repeat_results]
            abs_errs_full = [r['abs_error_model_full'] for r in repeat_results]

            mean_dp,       ci_lo, ci_hi = bootstrap_ci(est_dps,       seed=audit_seed)
            mean_abs,      _,     _     = bootstrap_ci(abs_errs,       seed=audit_seed)
            mean_abs_data, _,     _     = bootstrap_ci(abs_errs_data,  seed=audit_seed)
            mean_abs_full, _,     _     = bootstrap_ci(abs_errs_full,  seed=audit_seed)

            results.append({
                'mode'                     : f'collab_{coalition_size}_budgeted_agg',
                'auditor_ids'              : list(coalition),
                'auditor_id'               : label,
                'target_id'                : target_id,
                'budget'                   : budget,
                'actual_budget'            : actual_budget,
                'true_dp_gap_model_val'    : true_dp,
                'true_dp_gap_data'         : true_dp_data,
                'true_dp_gap_model_full'   : true_dp_model_full,
                'mean_est_dp'              : mean_dp,
                'std_est_dp'               : float(np.std(est_dps)),
                'ci_lower'                 : ci_lo,
                'ci_upper'                 : ci_hi,
                'mean_abs_error'           : mean_abs,
                'mean_abs_error_model_val' : mean_abs,
                'mean_abs_error_data'      : mean_abs_data,
                'mean_abs_error_model_full': mean_abs_full,
                'repeats'                  : repeat_results,
            })
            print(f'{tag} Collab {coalition_size} bud | ({label}) budget={budget} → '
                  f'mean_est_dp={mean_dp:.4f} ± {np.std(est_dps):.4f}  '
                  f'mean_abs_err={mean_abs:.4f}', flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Worker: audits one TARGET node (called in a separate process per GPU)
# ─────────────────────────────────────────────────────────────────────────────
def audit_target_node(target_id, gpu_id, dataset, node_indices,
                      true_dp_gaps, true_dp_gaps_data,
                      result_queue, cfg, nfs_paths):
    tag = f'[Target {target_id} | GPU {gpu_id}]'
    try:
        device = torch.device(f'cuda:{gpu_id}')

        ckpt_path = os.path.join(nfs_paths['ckpt_dir'],
                                 f'node_{target_id}_{PARTITION_ATTR}_best.pt')
        model = LeNet5().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        print(f'{tag} Model loaded from {ckpt_path}', flush=True)

        true_dp      = true_dp_gaps[target_id]
        true_dp_data = true_dp_gaps_data[target_id]

        # model-full ground truth
        target_all_indices = node_indices[target_id - 1]
        target_all_gender  = np.array(dataset[cfg['dataset']['sensitive_attr']],
                                      dtype=np.int64)[target_all_indices]
        full_preds, _ = get_predictions(model, dataset, target_all_indices, device)
        _, _, true_dp_model_full = compute_dp_gap(full_preds, target_all_gender)

        print(f'{tag} Ground truths — dp_gap_data={true_dp_data:.4f}  '
              f'dp_gap_model_val={true_dp:.4f}  '
              f'dp_gap_model_full={true_dp_model_full:.4f}', flush=True)

        N              = len(dataset)
        global_indices = np.arange(N)
        global_gender  = np.array(dataset[cfg['dataset']['sensitive_attr']],
                                   dtype=np.int64)
        num_nodes      = cfg['partition']['num_nodes']
        budget_sizes   = cfg['audit'].get('budget_sizes',  [100, 500, 1000, 2000, 5000])
        num_repeats    = cfg['audit'].get('num_repeats',   10)
        gender_attr    = cfg['dataset']['sensitive_attr']
        other_nodes    = [n for n in range(1, num_nodes + 1) if n != target_id]

        # ── Mode 1: Full local ────────────────────────────────────────────────
        full_results = []
        for auditor_id in other_nodes:
            auditor_indices = node_indices[auditor_id - 1]
            auditor_gender  = np.array(dataset[gender_attr],
                                       dtype=np.int64)[auditor_indices]
            r = run_single_audit(model, dataset, auditor_indices,
                                 auditor_gender, true_dp, device)
            _add_gt_errors(r, true_dp_data, true_dp_model_full)
            r.update({'mode': 'full_local', 'auditor_id': auditor_id,
                      'target_id': target_id})
            full_results.append(r)
            print(f'{tag} Full local  | Auditor {auditor_id} → '
                  f'est_dp={r["est_dp_gap"]:.4f}  '
                  f'abs_err(model_val)={r["abs_error"]:.4f}  '
                  f'abs_err(data)={r["abs_error_data"]:.4f}  '
                  f'abs_err(model_full)={r["abs_error_model_full"]:.4f}',
                  flush=True)

        # ── Mode 1.5: Pairwise collaborative audits ───────────────────────────
        collab_pair_results = _run_collab_audits(
            model, dataset, node_indices, target_id, other_nodes,
            coalition_size=2,
            true_dp=true_dp, true_dp_data=true_dp_data,
            true_dp_model_full=true_dp_model_full,
            gender_attr=gender_attr, device=device, tag=tag)

        # ── Mode 1.6: Triplet collaborative audits ────────────────────────────
        collab_triple_results = _run_collab_audits(
            model, dataset, node_indices, target_id, other_nodes,
            coalition_size=3,
            true_dp=true_dp, true_dp_data=true_dp_data,
            true_dp_model_full=true_dp_model_full,
            gender_attr=gender_attr, device=device, tag=tag)

        # ── Mode 2: Budgeted audits ───────────────────────────────────────────
        budget_results = []
        for auditor_id in other_nodes:
            auditor_indices = node_indices[auditor_id - 1]
            auditor_gender  = np.array(dataset[gender_attr],
                                       dtype=np.int64)[auditor_indices]
            n_available = len(auditor_indices)

            for budget in budget_sizes:
                actual_budget  = min(budget, n_available)
                repeat_results = []

                for rep in range(num_repeats):
                    rng = np.random.default_rng(
                        cfg['audit']['seed'] + auditor_id * 1000
                        + target_id * 100 + rep)
                    sampled_local  = rng.choice(n_available, size=actual_budget,
                                                replace=False)
                    sampled_global = auditor_indices[sampled_local]
                    sampled_gender = auditor_gender[sampled_local]

                    r = run_single_audit(model, dataset, sampled_global,
                                         sampled_gender, true_dp, device)
                    _add_gt_errors(r, true_dp_data, true_dp_model_full)
                    r.update({
                        'mode': 'budgeted', 'auditor_id': auditor_id,
                        'target_id': target_id, 'budget': budget,
                        'repeat': rep, 'actual_budget': actual_budget,
                    })
                    repeat_results.append(r)

                est_dps       = [r['est_dp_gap']           for r in repeat_results]
                abs_errs      = [r['abs_error']            for r in repeat_results]
                rel_errs      = [r['rel_error']            for r in repeat_results]
                abs_errs_data = [r['abs_error_data']       for r in repeat_results]
                abs_errs_full = [r['abs_error_model_full'] for r in repeat_results]

                mean_dp,        ci_lo, ci_hi = bootstrap_ci(est_dps,       seed=cfg['audit']['seed'])
                mean_abs,       _,     _     = bootstrap_ci(abs_errs,       seed=cfg['audit']['seed'])
                mean_rel,       _,     _     = bootstrap_ci(rel_errs,       seed=cfg['audit']['seed'])
                mean_abs_data,  _,     _     = bootstrap_ci(abs_errs_data,  seed=cfg['audit']['seed'])
                mean_abs_full,  _,     _     = bootstrap_ci(abs_errs_full,  seed=cfg['audit']['seed'])

                budget_results.append({
                    'mode'                        : 'budgeted_agg',
                    'auditor_id'                  : auditor_id,
                    'target_id'                   : target_id,
                    'budget'                      : budget,
                    'actual_budget'               : actual_budget,
                    'true_dp_gap_model_val'        : true_dp,
                    'true_dp_gap_data'             : true_dp_data,
                    'true_dp_gap_model_full'       : true_dp_model_full,
                    'mean_est_dp'                  : mean_dp,
                    'std_est_dp'                   : float(np.std(est_dps)),
                    'ci_lower'                     : ci_lo,
                    'ci_upper'                     : ci_hi,
                    'mean_abs_error'               : mean_abs,
                    'mean_abs_error_model_val'     : mean_abs,
                    'mean_rel_error'               : mean_rel,
                    'mean_abs_error_data'          : mean_abs_data,
                    'mean_abs_error_model_full'    : mean_abs_full,
                    'repeats'                      : repeat_results,
                })
                print(f'{tag} Budgeted    | Auditor {auditor_id} budget={budget} → '
                      f'mean_est_dp={mean_dp:.4f} ± {np.std(est_dps):.4f}  '
                      f'mean_abs_err={mean_abs:.4f}', flush=True)

        # ── Mode 2.5: Budgeted pairwise collaborative audits ──────────────────
        collab_pair_budget_results = _run_collab_budget_audits(
            model, dataset, node_indices, target_id, other_nodes,
            coalition_size=2, budget_sizes=budget_sizes, num_repeats=num_repeats,
            true_dp=true_dp, true_dp_data=true_dp_data,
            true_dp_model_full=true_dp_model_full,
            gender_attr=gender_attr, device=device, tag=tag,
            audit_seed=cfg['audit']['seed'])

        # ── Mode 2.6: Budgeted triplet collaborative audits ───────────────────
        collab_triple_budget_results = _run_collab_budget_audits(
            model, dataset, node_indices, target_id, other_nodes,
            coalition_size=3, budget_sizes=budget_sizes, num_repeats=num_repeats,
            true_dp=true_dp, true_dp_data=true_dp_data,
            true_dp_model_full=true_dp_model_full,
            gender_attr=gender_attr, device=device, tag=tag,
            audit_seed=cfg['audit']['seed'])

        # ── Mode 3: Global audit ──────────────────────────────────────────────
        r_global_all = run_single_audit(model, dataset, global_indices,
                                        global_gender, true_dp, device)
        _add_gt_errors(r_global_all, true_dp_data, true_dp_model_full)
        r_global_all.update({'mode': 'global_all', 'auditor_id': 'global_all',
                              'target_id': target_id})
        print(f'{tag} Global (all)  | '
              f'est_dp={r_global_all["est_dp_gap"]:.4f}  '
              f'abs_err={r_global_all["abs_error"]:.4f}', flush=True)

        excl_mask           = np.ones(N, dtype=bool)
        excl_mask[node_indices[target_id - 1]] = False
        global_excl_indices = np.where(excl_mask)[0]
        global_excl_gender  = global_gender[excl_mask]
        r_global_excl = run_single_audit(model, dataset, global_excl_indices,
                                         global_excl_gender, true_dp, device)
        _add_gt_errors(r_global_excl, true_dp_data, true_dp_model_full)
        r_global_excl.update({'mode': 'global_excl', 'auditor_id': 'global_excl',
                               'target_id': target_id})
        print(f'{tag} Global (excl) | '
              f'n_queries={len(global_excl_indices):,}  '
              f'est_dp={r_global_excl["est_dp_gap"]:.4f}  '
              f'abs_err={r_global_excl["abs_error"]:.4f}', flush=True)

        # ── Mode 3c: Budgeted global_excl ─────────────────────────────────────
        n_excl = len(global_excl_indices)
        global_excl_budget_results = []
        for budget in budget_sizes:
            actual_budget  = min(budget, n_excl)
            repeat_results = []
            for rep in range(num_repeats):
                rng = np.random.default_rng(
                    cfg['audit']['seed'] + target_id * 100 + rep + 9999)
                sampled   = rng.choice(n_excl, size=actual_budget, replace=False)
                s_indices = global_excl_indices[sampled]
                s_gender  = global_excl_gender[sampled]
                r = run_single_audit(model, dataset, s_indices, s_gender,
                                     true_dp, device)
                _add_gt_errors(r, true_dp_data, true_dp_model_full)
                r.update({
                    'mode': 'global_excl_budgeted', 'auditor_id': 'global_excl',
                    'target_id': target_id, 'budget': budget,
                    'repeat': rep, 'actual_budget': actual_budget,
                })
                repeat_results.append(r)

            est_dps       = [r['est_dp_gap']           for r in repeat_results]
            abs_errs_val  = [r['abs_error']            for r in repeat_results]
            abs_errs_data = [r['abs_error_data']       for r in repeat_results]
            abs_errs_full = [r['abs_error_model_full'] for r in repeat_results]

            mean_dp,       ci_lo, ci_hi = bootstrap_ci(est_dps,       seed=cfg['audit']['seed'])
            mean_abs,      _,     _     = bootstrap_ci(abs_errs_val,  seed=cfg['audit']['seed'])
            mean_abs_data, _,     _     = bootstrap_ci(abs_errs_data, seed=cfg['audit']['seed'])
            mean_abs_full, _,     _     = bootstrap_ci(abs_errs_full, seed=cfg['audit']['seed'])

            global_excl_budget_results.append({
                'mode'                     : 'global_excl_budgeted_agg',
                'auditor_id'               : 'global_excl',
                'target_id'                : target_id,
                'budget'                   : budget,
                'actual_budget'            : actual_budget,
                'true_dp_gap_model_val'    : true_dp,
                'true_dp_gap_data'         : true_dp_data,
                'true_dp_gap_model_full'   : true_dp_model_full,
                'mean_est_dp'              : mean_dp,
                'std_est_dp'               : float(np.std(est_dps)),
                'ci_lower'                 : ci_lo,
                'ci_upper'                 : ci_hi,
                'mean_abs_error'           : mean_abs,
                'mean_abs_error_model_val' : mean_abs,
                'mean_abs_error_data'      : mean_abs_data,
                'mean_abs_error_model_full': mean_abs_full,
                'repeats'                  : repeat_results,
            })
            print(f'{tag} Global excl bud | budget={budget} → '
                  f'mean_est_dp={mean_dp:.4f} ± {np.std(est_dps):.4f}  '
                  f'mean_abs_err={mean_abs:.4f}', flush=True)

        result_queue.put(('success', target_id, {
            'full'                    : full_results,
            'collab_pairs'            : collab_pair_results,
            'collab_triples'          : collab_triple_results,
            'budget'                  : budget_results,
            'collab_pairs_budget'     : collab_pair_budget_results,
            'collab_triples_budget'   : collab_triple_budget_results,
            'global_all'              : r_global_all,
            'global_excl'             : r_global_excl,
            'global_excl_budget'      : global_excl_budget_results,
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
    fig, ax = plt.subplots(figsize=(7, 6))
    for auditor_id in range(1, NUM_NODES + 1):
        pairs = [r for r in full_results if r['auditor_id'] == auditor_id]
        if not pairs:
            continue
        est  = [r['est_dp_gap']            for r in pairs]
        true = [r['true_dp_gap_model_val'] for r in pairs]
        ax.scatter(true, est, color=NODE_COLORS[auditor_id-1],
                   label=f'Auditor {auditor_id}', s=100, zorder=3)
    lims = [0, max(r['true_dp_gap_model_val'] for r in full_results) * 1.3]
    ax.plot(lims, lims, 'k--', alpha=0.4, label='Perfect estimate')
    ax.set_xlabel('True DP Gap'); ax.set_ylabel('Estimated DP Gap')
    ax.set_title('Full Local Audit\nEstimated vs True DP Gap',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(plot_dir, f'step5c_full_estimated_vs_true_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_full_error_heatmaps(full_results, num_nodes, plot_dir):
    abs_mat = np.full((num_nodes, num_nodes), np.nan)
    rel_mat = np.full((num_nodes, num_nodes), np.nan)
    for r in full_results:
        abs_mat[r['auditor_id'] - 1, r['target_id'] - 1] = r['abs_error']
        rel_mat[r['auditor_id'] - 1, r['target_id'] - 1] = r['rel_error']
    node_labels = [f'Node {i+1}' for i in range(num_nodes)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Full Local Audit — Error Heatmaps', fontsize=13, fontweight='bold')
    for ax, mat, title, fmt in zip(
        axes, [abs_mat, rel_mat],
        ['Absolute Error', 'Relative Error'], ['{:.3f}', '{:.0%}']
    ):
        im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(num_nodes)); ax.set_xticklabels(node_labels, fontsize=9)
        ax.set_yticks(range(num_nodes)); ax.set_yticklabels(node_labels, fontsize=9)
        ax.set_xlabel('Target Node'); ax.set_ylabel('Auditor Node')
        ax.set_title(title)
        for i in range(num_nodes):
            for j in range(num_nodes):
                val = mat[i, j]
                ax.text(j, i, fmt.format(val) if not np.isnan(val) else '—',
                        ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = os.path.join(plot_dir, f'step5c_full_error_heatmap_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_global_vs_local(full_results, global_all_results, global_excl_results, plot_dir):
    target_ids = sorted(set(r['target_id'] for r in global_all_results))
    fig, axes  = plt.subplots(1, len(target_ids),
                               figsize=(4 * len(target_ids), 5), sharey=True)
    if len(target_ids) == 1:
        axes = [axes]
    fig.suptitle('Global Auditor vs Local Auditors per Target Node',
                 fontsize=13, fontweight='bold')
    for ax, target_id in zip(axes, target_ids):
        local         = [r for r in full_results if r['target_id'] == target_id]
        local_ests    = [r['est_dp_gap']  for r in local]
        local_aud_ids = [r['auditor_id'] for r in local]
        g_all  = next(r for r in global_all_results  if r['target_id'] == target_id)
        g_excl = next(r for r in global_excl_results if r['target_id'] == target_id)
        x_local = range(len(local))
        ax.bar(x_local, local_ests,
               color=[NODE_COLORS[i-1] for i in local_aud_ids],
               edgecolor='white', linewidth=1.2, label='Local auditors')
        ax.axhline(g_all['true_dp_gap_model_val'], color='black',   linestyle='--',
                   linewidth=1.5, label=f'True DP — model/val ({g_all["true_dp_gap_model_val"]:.3f})')
        ax.axhline(g_all['true_dp_gap_model_full'], color='dimgray', linestyle='-.',
                   linewidth=1.5, label=f'True DP — model/full ({g_all["true_dp_gap_model_full"]:.3f})')
        ax.axhline(g_all['true_dp_gap_data'],       color='gray',    linestyle=':',
                   linewidth=1.5, label=f'True DP — data labels ({g_all["true_dp_gap_data"]:.3f})')
        ax.axhline(g_all['est_dp_gap'],  color='red',    linestyle='-.',
                   linewidth=1.5, label=f'Global all ({g_all["est_dp_gap"]:.3f})')
        ax.axhline(g_excl['est_dp_gap'], color='orange', linestyle=':',
                   linewidth=1.5, label=f'Global excl ({g_excl["est_dp_gap"]:.3f})')
        ax.set_title(f'Target: Node {target_id}', fontweight='bold')
        ax.set_xlabel('Auditor Node')
        ax.set_xticks(list(x_local))
        ax.set_xticklabels([f'N{i}' for i in local_aud_ids])
        all_vals = local_ests + [g_all['true_dp_gap_model_val'],
                                  g_all['true_dp_gap_data'],
                                  g_all['true_dp_gap_model_full'],
                                  g_all['est_dp_gap'], g_excl['est_dp_gap']]
        ax.set_ylim(0, max(all_vals) * 1.3)
        ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Estimated DP Gap')
    plt.tight_layout()
    out = os.path.join(plot_dir, f'step5c_global_vs_local_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_collab_all_modes(full_results, collab_pair_results,
                          collab_triple_results, plot_dir):
    """
    Per target node: side-by-side bars for single, pair, and triple
    collaborative auditors. Values labelled on top of bars.
    Groups separated by vertical dividers.
    """
    target_ids = sorted(set(r['target_id'] for r in full_results))
    fig, axes  = plt.subplots(1, len(target_ids),
                               figsize=(6 * len(target_ids), 5), sharey=True)
    if len(target_ids) == 1:
        axes = [axes]
    fig.suptitle('Single vs Pairwise vs Triplet Collaborative Auditors\n'
                 '(estimated DP gap per auditor coalition, values on bars)',
                 fontsize=13, fontweight='bold')

    for ax, target_id in zip(axes, target_ids):
        singles = sorted([r for r in full_results          if r['target_id'] == target_id],
                         key=lambda r: r['auditor_id'])
        pairs   = sorted([r for r in collab_pair_results   if r['target_id'] == target_id],
                         key=lambda r: r['auditor_id'])
        triples = sorted([r for r in collab_triple_results if r['target_id'] == target_id],
                         key=lambda r: r['auditor_id'])

        true_dp = singles[0]['true_dp_gap_model_val'] if singles else 0.0

        all_groups = [
            (singles, NODE_COLORS[:len(singles)], '',   'Single'),
            (pairs,   ['#b0b0b0'] * len(pairs),  '//', 'Pair'),
            (triples, ['#606060'] * len(triples), 'xx', 'Triple'),
        ]

        x_pos   = 0
        dividers = []
        for group_results, colors, hatch, _ in all_groups:
            if x_pos > 0:
                dividers.append(x_pos - 0.5)
            for r, col in zip(group_results, colors):
                val = r['est_dp_gap']
                ax.bar(x_pos, val, color=col, edgecolor='white', linewidth=0.8,
                       width=0.7, hatch=hatch, zorder=3)
                ax.text(x_pos, val + 0.003, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=6, rotation=90)
                x_pos += 1

        for xd in dividers:
            ax.axvline(xd, color='gray', linestyle=':', linewidth=1)

        ax.axhline(true_dp, color='black', linestyle='--', linewidth=1.5,
                   label=f'True DP model/val ({true_dp:.3f})', zorder=2)

        # Group labels below dividers
        n_s, n_p, n_t = len(singles), len(pairs), len(triples)
        centres = [n_s / 2 - 0.5,
                   n_s + n_p / 2 - 0.5,
                   n_s + n_p + n_t / 2 - 0.5]
        for cx, lbl in zip(centres, ['Single', 'Pair', 'Triple']):
            ax.text(cx, -0.04, lbl, ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=8, color='gray')

        # x-tick labels
        all_labels = ([f'N{r["auditor_id"]}' for r in singles] +
                      [f'N{r["auditor_id"]}' for r in pairs] +
                      [f'N{r["auditor_id"]}' for r in triples])
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, fontsize=6, rotation=45, ha='right')

        all_vals = [r['est_dp_gap'] for r in singles + pairs + triples] + [true_dp]
        ax.set_ylim(0, max(all_vals) * 1.35)
        ax.set_title(f'Target: Node {target_id}', fontweight='bold')
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Estimated DP Gap')
    plt.tight_layout()
    out = os.path.join(plot_dir, f'step5c_collab_all_modes_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


def plot_budget_collab_comparison(budget_results, collab_pair_budget_results,
                                   collab_triple_budget_results,
                                   global_excl_budget_results, plot_dir):
    """
    Mean absolute error vs budget for single / pair / triple / global_excl,
    one subplot per target node.  Uses mean_abs_error (model_val ground truth).
    """
    target_ids = sorted(set(r['target_id'] for r in budget_results))
    fig, axes  = plt.subplots(1, len(target_ids),
                               figsize=(5 * len(target_ids), 5), sharey=True)
    if len(target_ids) == 1:
        axes = [axes]
    fig.suptitle('Mean Absolute Error vs Query Budget\n'
                 'Single / Pair / Triple Collaborative / Global-Excl',
                 fontsize=13, fontweight='bold')

    for ax, target_id in zip(axes, target_ids):
        budget_sizes = sorted(set(r['budget'] for r in budget_results
                                  if r['target_id'] == target_id))
        # Single: average across all auditors per budget
        single_means, single_stds = [], []
        for b in budget_sizes:
            vals = [r['mean_abs_error'] for r in budget_results
                    if r['target_id'] == target_id and r['budget'] == b]
            single_means.append(np.mean(vals))
            single_stds.append(np.std(vals))

        # Pair: average across all coalitions per budget
        pair_means, pair_stds = [], []
        for b in budget_sizes:
            vals = [r['mean_abs_error'] for r in collab_pair_budget_results
                    if r['target_id'] == target_id and r['budget'] == b]
            pair_means.append(np.mean(vals) if vals else np.nan)
            pair_stds.append(np.std(vals) if vals else np.nan)

        # Triple: average across all coalitions per budget
        triple_means, triple_stds = [], []
        for b in budget_sizes:
            vals = [r['mean_abs_error'] for r in collab_triple_budget_results
                    if r['target_id'] == target_id and r['budget'] == b]
            triple_means.append(np.mean(vals) if vals else np.nan)
            triple_stds.append(np.std(vals) if vals else np.nan)

        # Global_excl budgeted
        ge_means, ge_stds = [], []
        for b in budget_sizes:
            vals = [r['mean_abs_error'] for r in global_excl_budget_results
                    if r['target_id'] == target_id and r['budget'] == b]
            ge_means.append(np.mean(vals) if vals else np.nan)
            ge_stds.append(np.std(vals) if vals else np.nan)

        xs = np.array(budget_sizes)
        for means, stds, label, color, ls in [
            (single_means, single_stds,  'Single',       'steelblue',   '-'),
            (pair_means,   pair_stds,    'Pair collab',  '#b0b0b0',     '--'),
            (triple_means, triple_stds,  'Triple collab','#606060',     '-.'),
            (ge_means,     ge_stds,      'Global excl',  'darkorange',  ':'),
        ]:
            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)
            ax.plot(xs, means, color=color, linestyle=ls, marker='o',
                    linewidth=1.8, markersize=5, label=label)
            ax.fill_between(xs, means - stds, means + stds,
                            alpha=0.15, color=color)

        ax.set_xscale('log')
        ax.set_xlabel('Query budget')
        ax.set_title(f'Target: Node {target_id}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Mean Abs Error (model_val GT)')
    plt.tight_layout()
    out = os.path.join(plot_dir, f'step5c_budget_collab_comparison_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot-only path: regenerate all plots from an existing results JSON
# ─────────────────────────────────────────────────────────────────────────────
def regenerate_plots(data, node_partition_stats):
    full_results                = data['full_results']
    collab_pair_results         = data['collab_pair_results']
    collab_triple_results       = data['collab_triple_results']
    budget_results              = data['budget_results']
    collab_pair_budget_results  = data['collab_pair_budget_results']
    collab_triple_budget_results = data['collab_triple_budget_results']
    global_all_results          = data['global_all_results']
    global_excl_results         = data['global_excl_results']
    global_excl_budget_results  = data['global_excl_budget_results']

    log.info(f"\n{'─'*70}")
    log.info("  Generating Plots (from cached results)")
    log.info(f"{'─'*70}\n")

    out = plot_full_estimated_vs_true(full_results, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_full_error_heatmaps(full_results, NUM_NODES, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_global_vs_local(full_results, global_all_results,
                                global_excl_results, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_collab_all_modes(full_results, collab_pair_results,
                                 collab_triple_results, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_budget_collab_comparison(budget_results, collab_pair_budget_results,
                                        collab_triple_budget_results,
                                        global_excl_budget_results, PLOT_DIR)
    log.info(f"  ✓ {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("  Step 5c: Black-Box Fairness Auditing — Extended Collaboration")
    log.info("=" * 70)
    log.info(f"\n  Config         : {args.config}")
    log.info(f"  Experiment     : {EXP_NAME}")
    log.info(f"  Run date       : {RUN_DATE}")
    log.info(f"  NFS root       : {NFS_ROOT}")
    log.info(f"  Budget sizes   : {BUDGET_SIZES}")
    log.info(f"  Repeats/budget : {NUM_REPEATS}")
    log.info(f"  Log file       : {log_path}")

    results_path = os.path.join(RESULTS_DIR, RESULTS_FNAME)

    # ── Skip logic ─────────────────────────────────────────────────────────────
    if not args.force and os.path.exists(results_path):
        with open(results_path) as f:
            cached = json.load(f)
        if REQUIRED_KEYS.issubset(cached.keys()):
            log.info(f"\n  ✓ Found complete results at {results_path}")
            log.info(f"  Skipping GPU computation — regenerating plots only.")
            log.info(f"  (Use --force to re-run all audits.)\n")

            # Load step 3 node stats for any plot that needs them
            step3_path = os.path.join(RESULTS_DIR,
                                       f'step3_{ALPHA}_{PARTITION_ATTR}_partition_stats.json')
            with open(step3_path) as f:
                step3 = json.load(f)
            regenerate_plots(cached, step3['nodes'])
            return
        else:
            missing = REQUIRED_KEYS - cached.keys()
            log.info(f"\n  Existing JSON missing keys {missing} — re-running audits.")

    # ── Load dataset ───────────────────────────────────────────────────────────
    log.info(f"\n  Loading CelebA...")
    dataset = load_dataset(DATASET_NAME, split='train')
    N = len(dataset)
    log.info(f"  ✓ {N:,} samples loaded")

    # ── Load partition ─────────────────────────────────────────────────────────
    partition_fname = f'partition_alpha{ALPHA}_seed{PART_SEED}_{PARTITION_ATTR}.json'
    partition_path  = os.path.join(EXP_DIR, 'partitions', RUN_DATE, partition_fname)
    with open(partition_path) as f:
        partition_data = json.load(f)
    node_indices = [np.array(idx, dtype=np.int64)
                    for idx in partition_data['node_indices']]
    log.info(f"  ✓ Partition loaded from {partition_path}")

    # ── Load true DP gaps from Step 4 ─────────────────────────────────────────
    step4_path = os.path.join(RESULTS_DIR, f'step4_all_nodes_{PARTITION_ATTR}_results.json')
    with open(step4_path) as f:
        step4 = json.load(f)
    true_dp_gaps      = {r['node_id']: r['dp_gap_model'] for r in step4['nodes']}
    true_dp_gaps_data = {r['node_id']: r['dp_gap_data']  for r in step4['nodes']}
    node_stats        = {r['node_id']: r                 for r in step4['nodes']}

    log.info(f"\n  True DP gaps (model) from Step 4:")
    for node_id, dp in true_dp_gaps.items():
        log.info(f"    Node {node_id}: {dp:.4f}")

    # ── Load step 3 node stats ─────────────────────────────────────────────────
    step3_path = os.path.join(RESULTS_DIR, f'step3_{ALPHA}_{PARTITION_ATTR}_partition_stats.json')
    with open(step3_path) as f:
        step3 = json.load(f)
    node_partition_stats = step3['nodes']

    # ── Audit count summary ────────────────────────────────────────────────────
    n_others  = NUM_NODES - 1
    n_pairs   = len(list(combinations(range(n_others), 2)))
    n_triples = len(list(combinations(range(n_others), 3)))
    log.info(f"\n  Collaboration counts per target node:")
    log.info(f"    Single  : {n_others}")
    log.info(f"    Pairs   : {n_pairs}  (C({n_others},2))")
    log.info(f"    Triples : {n_triples}  (C({n_others},3))")

    # ── Launch parallel auditing ───────────────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    log.info(f"\n  GPUs available : {n_gpus}")
    log.info(f"\n{'─'*70}")
    log.info(f"  Launching {NUM_NODES} audit workers "
             f"(max {min(NUM_NODES, n_gpus)} parallel)")
    log.info(f"{'─'*70}\n")

    mp.set_start_method('spawn', force=True)
    result_queue   = mp.Queue()
    active_procs   = {}
    pending        = list(range(1, NUM_NODES + 1))
    all_results    = {}
    start_time     = time.time()
    gpu_ids        = [i % n_gpus for i in range(NUM_NODES)]

    def launch_target(target_id):
        gpu_id    = gpu_ids[target_id - 1]
        nfs_paths = {'ckpt_dir': CKPT_DIR, 'results_dir': RESULTS_DIR,
                     'plot_dir': PLOT_DIR}
        p = mp.Process(
            target=audit_target_node,
            args=(target_id, gpu_id, dataset, node_indices,
                  true_dp_gaps, true_dp_gaps_data, result_queue, cfg, nfs_paths),
            name=f'Target-{target_id}'
        )
        p.start()
        active_procs[target_id] = p
        log.info(f"  ▶ Launched Target {target_id} on GPU {gpu_id}  (pid={p.pid})")

    for _ in range(min(n_gpus, NUM_NODES)):
        if pending:
            launch_target(pending.pop(0))

    completed = 0
    while completed < NUM_NODES:
        status, target_id, payload = result_queue.get()
        active_procs.pop(target_id).join()
        completed += 1
        elapsed = time.time() - start_time
        if status == 'success':
            all_results[target_id] = payload
            log.info(f"\n  ✓ Target {target_id} finished  (elapsed={elapsed:.0f}s)")
        else:
            log.info(f"\n  ✗ Target {target_id} FAILED: {payload}")
        if pending:
            launch_target(pending.pop(0))

    total_time = time.time() - start_time
    log.info(f"\n  All targets finished in {total_time:.0f}s ({total_time/60:.1f} min)\n")

    # ── Flatten results ────────────────────────────────────────────────────────
    full_results                = []
    collab_pair_results         = []
    collab_triple_results       = []
    budget_results              = []
    collab_pair_budget_results  = []
    collab_triple_budget_results = []
    global_all_results          = []
    global_excl_results         = []
    global_excl_budget_results  = []

    for target_id, res in all_results.items():
        full_results.extend(res['full'])
        collab_pair_results.extend(res['collab_pairs'])
        collab_triple_results.extend(res['collab_triples'])
        budget_results.extend(res['budget'])
        collab_pair_budget_results.extend(res['collab_pairs_budget'])
        collab_triple_budget_results.extend(res['collab_triples_budget'])
        global_all_results.append(res['global_all'])
        global_excl_results.append(res['global_excl'])
        global_excl_budget_results.extend(res['global_excl_budget'])

    if not full_results:
        log.info("  ✗ No successful audits — exiting")
        return

    # ── Generate plots ─────────────────────────────────────────────────────────
    log.info(f"\n{'─'*70}")
    log.info("  Generating Plots")
    log.info(f"{'─'*70}\n")

    out = plot_full_estimated_vs_true(full_results, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_full_error_heatmaps(full_results, NUM_NODES, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_global_vs_local(full_results, global_all_results,
                                global_excl_results, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_collab_all_modes(full_results, collab_pair_results,
                                 collab_triple_results, PLOT_DIR)
    log.info(f"  ✓ {out}")
    out = plot_budget_collab_comparison(budget_results, collab_pair_budget_results,
                                        collab_triple_budget_results,
                                        global_excl_budget_results, PLOT_DIR)
    log.info(f"  ✓ {out}")

    # ── Save results JSON ──────────────────────────────────────────────────────
    abs_errs      = np.array([r['abs_error']           for r in full_results])
    rel_errs      = np.array([r['rel_error']           for r in full_results])
    abs_errs_data = np.array([r['abs_error_data']      for r in full_results])
    abs_errs_full = np.array([r['abs_error_model_full'] for r in full_results])

    true_dp_gaps_model_full = {r['target_id']: r['true_dp_gap_model_full']
                               for r in global_all_results}

    results = {
        'experiment'              : EXP_NAME,
        'partition_attr'          : PARTITION_ATTR,
        'alpha'                   : ALPHA,
        'num_nodes'               : NUM_NODES,
        'budget_sizes'            : BUDGET_SIZES,
        'num_repeats'             : NUM_REPEATS,
        'total_time_s'            : total_time,
        'true_dp_gaps_model_val'  : true_dp_gaps,
        'true_dp_gaps_data'       : true_dp_gaps_data,
        'true_dp_gaps_model_full' : true_dp_gaps_model_full,
        'full_results'                : full_results,
        'collab_pair_results'         : collab_pair_results,
        'collab_triple_results'       : collab_triple_results,
        'budget_results'              : budget_results,
        'collab_pair_budget_results'  : collab_pair_budget_results,
        'collab_triple_budget_results': collab_triple_budget_results,
        'global_all_results'          : global_all_results,
        'global_excl_results'         : global_excl_results,
        'global_excl_budget_results'  : global_excl_budget_results,
        'summary': {
            'full_mean_abs_error_model_val' : float(abs_errs.mean()),
            'full_std_abs_error_model_val'  : float(abs_errs.std()),
            'full_mean_rel_error_model_val' : float(rel_errs.mean()),
            'full_std_rel_error_model_val'  : float(rel_errs.std()),
            'full_mean_abs_error_data'      : float(abs_errs_data.mean()),
            'full_std_abs_error_data'       : float(abs_errs_data.std()),
            'full_mean_abs_error_model_full': float(abs_errs_full.mean()),
            'full_std_abs_error_model_full' : float(abs_errs_full.std()),
        }
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  ✓ Results saved → {results_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info("  Step 5c Complete")
    log.info(f"{'='*70}")
    log.info(f"  Total time : {total_time:.0f}s ({total_time/60:.1f} min)")
    log.info(f"  Full local abs err (model_val)  : {abs_errs.mean():.4f} ± {abs_errs.std():.4f}")
    log.info(f"  Full local abs err (data)       : {abs_errs_data.mean():.4f} ± {abs_errs_data.std():.4f}")
    log.info(f"  Full local abs err (model_full) : {abs_errs_full.mean():.4f} ± {abs_errs_full.std():.4f}")
    log.info(f"\n  Outputs saved to: {EXP_DIR}")
    log.info(f"    results/{RUN_DATE}/{RESULTS_FNAME}")
    log.info(f"    plots/{RUN_DATE}/step5c_collab_all_modes_{PARTITION_ATTR}.png")
    log.info(f"    plots/{RUN_DATE}/step5c_budget_collab_comparison_{PARTITION_ATTR}.png")
    log.info(f"    logs/{RUN_DATE}/step5c_{PARTITION_ATTR}.log")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
