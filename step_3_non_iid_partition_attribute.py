"""
Step 3: Non-IID Data Partitioning + Visualisation
===================================================
Sweeps over all 40 CelebA attributes × all configured Dirichlet alpha values
to help identify the best partition attribute and heterogeneity level.

Run with:
    python3 step_3_non_iid_partition_attribute.py --config config.yaml

    --attr  <ATTR>          run only this partition attribute
    --alpha <0.1 0.5 ...>   override alpha_values from config (space-separated)

Env vars:
    NFS_ROOT=/your/nfs/path   override nfs.root from config

Per-combination outputs (under {nfs_root}/experiments/{exp_name}/):
    plots/{date}/alpha{alpha}/{attr}/step3_*.png       (7 plots)
    partitions/{date}/partition_alpha{alpha}_seed{seed}_{attr}.json
    results/{date}/step3_{alpha}_{attr}_partition_stats.json

Summary outputs:
    plots/{date}/step3_sweep_summary.png   (heatmaps: attrs × alphas)
    results/{date}/step3_sweep_summary.json
    logs/{date}/step3.log
"""

import os
import sys
import json
import shutil
import logging
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from datasets import load_dataset
import yaml

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Step 3: Non-IID Partition + Visualisation')
parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--attr',   type=str, default=None,
                    help='Run only this partition attribute (default: sweep all 40)')
parser.add_argument('--alpha',  type=float, nargs='+', default=None,
                    help='Override alpha_values from config (e.g. --alpha 0.5 0.1)')
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

DATASET_NAME   = cfg['dataset']['name']
SENSITIVE_ATTR = cfg['dataset']['sensitive_attr']
TARGET_ATTR    = cfg['dataset']['target_attr']
NUM_NODES      = cfg['partition']['num_nodes']
MIN_SAMPLES    = cfg['partition']['min_samples']
PART_SEED      = cfg['partition']['seed']
RUN_DATE       = cfg['experiment'].get('run_date') or datetime.date.today().isoformat()

_alpha_single  = cfg['partition']['alpha']   # the alpha used by steps 4/5
EXP_NAME       = (cfg['experiment'].get('name')
                  or f"lenet_alpha{_alpha_single}_{NUM_NODES}nodes_seed{PART_SEED}")

ALPHA_VALUES   = args.alpha if args.alpha is not None \
                 else cfg['partition'].get('alpha_values', [cfg['partition']['alpha']])

# ── NFS paths ──────────────────────────────────────────────────────────────────
NFS_ROOT  = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR   = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
HF_CACHE  = os.environ.get('HF_DATASETS_CACHE',
                            os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE

PLOT_DIR      = os.path.join(EXP_DIR, 'plots',      RUN_DATE)
RESULTS_DIR   = os.path.join(EXP_DIR, 'results',    RUN_DATE)
PARTITION_DIR = os.path.join(EXP_DIR, 'partitions', RUN_DATE)
LOG_DIR       = os.path.join(EXP_DIR, 'logs',       RUN_DATE)

for d in [HF_CACHE, PLOT_DIR, RESULTS_DIR, PARTITION_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

shutil.copy(args.config, os.path.join(EXP_DIR, 'config.yaml'))

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, 'step3.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode='w')
    ]
)
log = logging.getLogger()

# ── All 40 CelebA attributes ───────────────────────────────────────────────────
CELEBA_ATTRS = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
    'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
    'Young'
]

NODE_COLORS = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']

# ── Determine sweep dimensions ────────────────────────────────────────────────
ATTRS_TO_RUN = [args.attr] if args.attr is not None else CELEBA_ATTRS
if args.attr is not None and args.attr not in CELEBA_ATTRS:
    raise ValueError(
        f"--attr '{args.attr}' not found in the 40 CelebA attributes.\n"
        f"Valid options: {CELEBA_ATTRS}"
    )

# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("  Step 3: Non-IID Partition Sweep")
log.info("=" * 70)
log.info(f"\n  Config          : {args.config}")
log.info(f"  Experiment      : {EXP_NAME}")
log.info(f"  Run date        : {RUN_DATE}")
log.info(f"  NFS root        : {NFS_ROOT}")
log.info(f"  Num nodes       : {NUM_NODES}")
log.info(f"  Alpha values    : {ALPHA_VALUES}")
log.info(f"  Min samples     : {MIN_SAMPLES}")
log.info(f"  Partition seed  : {PART_SEED}")
log.info(f"  Sensitive attr  : {SENSITIVE_ATTR}  (used for DP-gap)")
log.info(f"  Target attr     : {TARGET_ATTR}")
log.info(f"  Attributes      : {len(ATTRS_TO_RUN)}")
log.info(f"  Total combos    : {len(ALPHA_VALUES)} × {len(ATTRS_TO_RUN)} = "
         f"{len(ALPHA_VALUES)*len(ATTRS_TO_RUN)}")
log.info(f"  Log file        : {log_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load dataset + all attributes (once for the whole sweep)
# ─────────────────────────────────────────────────────────────────────────────
log.info("─" * 70)
log.info("  SECTION 1: Loading Dataset")
log.info("─" * 70)

log.info("\n  Loading CelebA (train split)...")
dataset = load_dataset(DATASET_NAME, split='train')
N = len(dataset)
log.info(f"  ✓ Loaded {N:,} samples")

gender  = np.array(dataset[SENSITIVE_ATTR], dtype=np.int64)
smiling = np.array(dataset[TARGET_ATTR],    dtype=np.int64)

global_pct_male       = float(gender.mean())
global_pct_smiling    = float(smiling.mean())
p_smile_male_global   = float(smiling[gender == 1].mean())
p_smile_female_global = float(smiling[gender == 0].mean())
dp_gap_global         = abs(p_smile_male_global - p_smile_female_global)

sens_pos_label = SENSITIVE_ATTR.replace('_', ' ')
sens_neg_label = f'Non-{sens_pos_label}'

log.info("  Loading all 40 attributes for distribution PCA...")
attr_matrix = np.zeros((N, len(CELEBA_ATTRS)), dtype=np.float32)
for j, attr in enumerate(CELEBA_ATTRS):
    attr_matrix[:, j] = np.array(dataset[attr], dtype=np.float32)
log.info(f"  ✓ Attribute matrix shape: {attr_matrix.shape}  (samples × attributes)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Dirichlet partition
# ─────────────────────────────────────────────────────────────────────────────
def dirichlet_partition(part_attr_vals, num_nodes, alpha, min_samples, seed):
    rng     = np.random.default_rng(seed)
    pos_idx = np.where(part_attr_vals == 1)[0]
    neg_idx = np.where(part_attr_vals == 0)[0]

    attempts = 0
    while True:
        attempts += 1
        pos_props = rng.dirichlet(alpha * np.ones(num_nodes))
        neg_props = rng.dirichlet(alpha * np.ones(num_nodes))

        pos_shuffled = rng.permutation(pos_idx)
        neg_shuffled = rng.permutation(neg_idx)

        pos_splits = (pos_props * len(pos_idx)).astype(int)
        neg_splits = (neg_props * len(neg_idx)).astype(int)

        pos_splits[-1] = len(pos_idx) - pos_splits[:-1].sum()
        neg_splits[-1] = len(neg_idx) - neg_splits[:-1].sum()

        node_indices = []
        p_start, n_start = 0, 0
        valid = True
        for i in range(num_nodes):
            p_end = p_start + pos_splits[i]
            n_end = n_start + neg_splits[i]
            node_idx = np.concatenate([
                pos_shuffled[p_start:p_end],
                neg_shuffled[n_start:n_end]
            ])
            if len(node_idx) < min_samples:
                valid = False
                break
            node_indices.append(node_idx)
            p_start = p_end
            n_start = n_end

        if valid:
            return node_indices, attempts


# ─────────────────────────────────────────────────────────────────────────────
# Per-combination processing function
# ─────────────────────────────────────────────────────────────────────────────
def process_partition_attr(PARTITION_ATTR, ALPHA):
    """Run partition + stats + plots for one (attr, alpha) combo. Returns summary dict."""
    _attr_display  = PARTITION_ATTR.replace('_', ' ')
    PART_POS_LABEL = _attr_display
    PART_NEG_LABEL = f'No {_attr_display}'

    alpha_str     = f'alpha{ALPHA}'
    attr_plot_dir = os.path.join(PLOT_DIR, alpha_str, PARTITION_ATTR)
    os.makedirs(attr_plot_dir, exist_ok=True)

    part_attr_vals = np.array(dataset[PARTITION_ATTR], dtype=np.int64)
    global_pct_pos = float(part_attr_vals.mean())

    # ── Partition ─────────────────────────────────────────────────────────────
    node_indices, attempts = dirichlet_partition(
        part_attr_vals, NUM_NODES, ALPHA, MIN_SAMPLES, PART_SEED
    )

    # ── Node statistics ───────────────────────────────────────────────────────
    node_labels = [f'Node {i+1}' for i in range(NUM_NODES)]

    node_stats = []
    for i, idx in enumerate(node_indices):
        node_gender    = gender[idx]
        node_smiling   = smiling[idx]
        node_part_attr = part_attr_vals[idx]
        n = len(idx)

        pct_male    = float(node_gender.mean())
        pct_smiling = float(node_smiling.mean())
        pct_pos     = float(node_part_attr.mean())

        n_sens_pos = (node_gender == 1).sum()
        n_sens_neg = (node_gender == 0).sum()
        p_sm = float(node_smiling[node_gender == 1].mean()) if n_sens_pos > 0 else 0.0
        p_sf = float(node_smiling[node_gender == 0].mean()) if n_sens_neg > 0 else 0.0
        dp_gap = abs(p_sm - p_sf)

        attr_props = attr_matrix[idx].mean(axis=0).tolist()

        node_stats.append({
            'node_id'          : i + 1,
            'n'                : int(n),
            'n_part_pos'       : int((node_part_attr == 1).sum()),
            'n_part_neg'       : int((node_part_attr == 0).sum()),
            'pct_part_pos'     : pct_pos,
            'pct_part_neg'     : 1.0 - pct_pos,
            'n_sens_pos'       : int(n_sens_pos),
            'n_sens_neg'       : int(n_sens_neg),
            'pct_male'         : pct_male,
            'pct_female'       : 1.0 - pct_male,
            'pct_smiling'      : pct_smiling,
            'p_smile_sens_pos' : p_sm,
            'p_smile_sens_neg' : p_sf,
            'dp_gap_data'      : dp_gap,
            'attr_props'       : attr_props,
        })

    pct_pos_per_node = [s['pct_part_pos'] for s in node_stats]
    dp_gaps          = [s['dp_gap_data']  for s in node_stats]
    pct_males        = [s['pct_male']     for s in node_stats]

    x = np.arange(NUM_NODES)
    all_labels  = ['Global'] + node_labels
    all_indices = [np.arange(N)] + node_indices

    # ── Plot 1: Node sizes ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes = [s['n'] for s in node_stats]
    bars  = ax.bar(node_labels, sizes, color=NODE_COLORS, edgecolor='white', linewidth=1.2)
    ax.axhline(N / NUM_NODES, color='gray', linestyle='--', alpha=0.7,
               label=f'Equal split ({N//NUM_NODES:,})')
    ax.set_title(f'Dataset Size per Node  (α={ALPHA}, partitioned by {PART_POS_LABEL})',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples')
    for bar, v in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{v:,}', ha='center', fontsize=9)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(attr_plot_dir, 'step3_node_sizes.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Partition attribute distribution per node ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.35
    ax.bar(x - width/2, [s['pct_part_pos']*100 for s in node_stats],
           width, label=PART_POS_LABEL, color='steelblue', edgecolor='white')
    ax.bar(x + width/2, [s['pct_part_neg']*100 for s in node_stats],
           width, label=PART_NEG_LABEL, color='salmon', edgecolor='white')
    ax.axhline(100*global_pct_pos, color='steelblue', linestyle='--', alpha=0.5,
               label=f'Global {PART_POS_LABEL} ({100*global_pct_pos:.1f}%)')
    ax.axhline(100*(1-global_pct_pos), color='salmon', linestyle='--', alpha=0.5,
               label=f'Global {PART_NEG_LABEL} ({100*(1-global_pct_pos):.1f}%)')
    ax.set_title(
        f'{PART_POS_LABEL} vs {PART_NEG_LABEL} Distribution per Node  (α={ALPHA})\n'
        f'This is the attribute driving the non-IID split',
        fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xticks(x); ax.set_xticklabels(node_labels)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(attr_plot_dir, f'step3_{PARTITION_ATTR}_per_node.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Smiling distribution per node ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(node_labels, [s['pct_smiling']*100 for s in node_stats],
                  color=NODE_COLORS, edgecolor='white', linewidth=1.2)
    ax.axhline(100*global_pct_smiling, color='gray', linestyle='--', alpha=0.7,
               label=f'Global ({100*global_pct_smiling:.1f}%)')
    ax.set_title(
        f'{TARGET_ATTR} Rate per Node  (α={ALPHA})\n'
        f'Partitioned by {PART_POS_LABEL}',
        fontsize=12, fontweight='bold')
    ax.set_ylabel(f'% {TARGET_ATTR}')
    ax.set_ylim(0, 100)
    for bar, s in zip(bars, node_stats):
        ax.text(bar.get_x() + bar.get_width()/2, s['pct_smiling']*100 + 0.5,
                f"{s['pct_smiling']*100:.1f}%", ha='center', fontsize=9)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(attr_plot_dir, 'step3_smiling_per_node.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 4: Stacked group proportions ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    group_colors = ['#2196F3', '#90CAF9', '#E91E63', '#F48FB1']
    group_labels_list = [
        f'{sens_pos_label} + {TARGET_ATTR}',
        f'{sens_pos_label} + Not {TARGET_ATTR}',
        f'{sens_neg_label} + {TARGET_ATTR}',
        f'{sens_neg_label} + Not {TARGET_ATTR}',
    ]
    bottoms    = np.zeros(len(all_labels))
    group_data = {g: [] for g in group_labels_list}
    for idx in all_indices:
        g = gender[idx]; s = smiling[idx]; n = len(idx)
        group_data[f'{sens_pos_label} + {TARGET_ATTR}'].append(     ((g==1)&(s==1)).sum() / n)
        group_data[f'{sens_pos_label} + Not {TARGET_ATTR}'].append( ((g==1)&(s==0)).sum() / n)
        group_data[f'{sens_neg_label} + {TARGET_ATTR}'].append(     ((g==0)&(s==1)).sum() / n)
        group_data[f'{sens_neg_label} + Not {TARGET_ATTR}'].append( ((g==0)&(s==0)).sum() / n)
    for gname, color in zip(group_labels_list, group_colors):
        vals = group_data[gname]
        ax.bar(all_labels, vals, bottom=bottoms,
               label=gname, color=color, edgecolor='white', linewidth=0.8)
        bottoms += np.array(vals)
    ax.set_title(
        f'{SENSITIVE_ATTR} × {TARGET_ATTR} Group Composition per Node  (α={ALPHA})\n'
        f'Non-IID split driven by {PART_POS_LABEL}',
        fontsize=11, fontweight='bold')
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    for j, (lbl, idx) in enumerate(zip(all_labels, all_indices)):
        pct_p = part_attr_vals[idx].mean()
        ax.text(j, 1.02, f'{100*pct_p:.0f}%\n{PARTITION_ATTR[:4]}',
                ha='center', fontsize=8, color='#1565C0')
    plt.tight_layout()
    plt.savefig(os.path.join(attr_plot_dir, 'step3_stacked_groups.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 5: Joint heatmaps ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, NUM_NODES + 1, figsize=(4*(NUM_NODES+1), 4))
    fig.suptitle(
        f'{SENSITIVE_ATTR} × {TARGET_ATTR} Joint Distribution per Node  (α={ALPHA})\n'
        f'Non-IID split driven by: {PART_POS_LABEL}',
        fontsize=12, fontweight='bold')
    for ax, idx, title in zip(axes, all_indices, all_labels):
        g = gender[idx]; s = smiling[idx]; n = len(idx)
        mat = np.array([
            [((g==0)&(s==0)).sum()/n, ((g==0)&(s==1)).sum()/n],
            [((g==1)&(s==0)).sum()/n, ((g==1)&(s==1)).sum()/n],
        ])
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=0.5, aspect='auto')
        ax.set_xticks([0,1])
        ax.set_xticklabels([f'No {TARGET_ATTR}', TARGET_ATTR], fontsize=8)
        ax.set_yticks([0,1])
        ax.set_yticklabels([sens_neg_label, sens_pos_label], fontsize=8)
        ax.set_title(title, fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                        fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=axes[-1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(attr_plot_dir, 'step3_joint_heatmaps.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 6: DP gap per node ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(node_labels, dp_gaps, color=NODE_COLORS, edgecolor='white', linewidth=1.2)
    ax.axhline(dp_gap_global, color='black', linestyle='--', linewidth=1.5,
               label=f'Global DP gap ({dp_gap_global:.4f})')
    ax.set_title(
        f'Ground-Truth DP Gap per Node  (α={ALPHA})\n'
        f'DP gap = |P({TARGET_ATTR}|{sens_pos_label}) − P({TARGET_ATTR}|{sens_neg_label})|\n'
        f'Non-IID split driven by: {PART_POS_LABEL}',
        fontsize=11, fontweight='bold')
    ax.set_ylabel('DP gap (data)')
    for bar, v in zip(bars, dp_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f'{v:.4f}', ha='center', fontsize=9)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(attr_plot_dir, 'step3_dp_gap_per_node.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Plot 7: Distribution-space PCA ────────────────────────────────────────
    prop_vectors = [attr_matrix.mean(axis=0)]
    for idx in node_indices:
        prop_vectors.append(attr_matrix[idx].mean(axis=0))
    prop_vectors = np.array(prop_vectors)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(prop_vectors[1:])
    coords    = pca.transform(prop_vectors)
    explained = pca.explained_variance_ratio_
    top_pc1   = [CELEBA_ATTRS[i] for i in np.argsort(np.abs(pca.components_[0]))[::-1][:5]]
    top_pc2   = [CELEBA_ATTRS[i] for i in np.argsort(np.abs(pca.components_[1]))[::-1][:5]]

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (xc, yc) in enumerate(coords[1:]):
        pct_p = node_stats[i]['pct_part_pos']
        ax.scatter(xc, yc, color=NODE_COLORS[i], s=200, zorder=5,
                   edgecolors='white', linewidth=1.5)
        ax.annotate(f'Node {i+1}\n({100*pct_p:.0f}% {PARTITION_ATTR[:8]})',
                    xy=(xc, yc), xytext=(xc + 0.0003, yc + 0.0003),
                    fontsize=10, fontweight='bold', color=NODE_COLORS[i])
    ax.scatter(coords[0, 0], coords[0, 1], color='black', s=300,
               marker='*', zorder=6, label='Global dataset')
    ax.annotate('Global', xy=(coords[0, 0], coords[0, 1]),
                xytext=(coords[0, 0] + 0.0003, coords[0, 1] + 0.0003),
                fontsize=10, fontweight='bold')
    for i, (xc, yc) in enumerate(coords[1:]):
        ax.plot([coords[0,0], xc], [coords[0,1], yc],
                color=NODE_COLORS[i], alpha=0.3, linestyle='--', linewidth=1)
    ax.set_xlabel(f'PC1 ({100*explained[0]:.1f}% var) — top: {", ".join(top_pc1[:3])}',
                  fontsize=9)
    ax.set_ylabel(f'PC2 ({100*explained[1]:.1f}% var) — top: {", ".join(top_pc2[:3])}',
                  fontsize=9)
    ax.set_title(
        f'Distribution-Space PCA  (α={ALPHA})\n'
        f'Non-IID split by: {PART_POS_LABEL}',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(attr_plot_dir, 'step3_distribution_pca.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Save partition JSON ───────────────────────────────────────────────────
    partition_fname = f'partition_alpha{ALPHA}_seed{PART_SEED}_{PARTITION_ATTR}.json'
    partition_path  = os.path.join(PARTITION_DIR, partition_fname)
    with open(partition_path, 'w') as f:
        json.dump({
            'alpha'          : ALPHA,
            'seed'           : PART_SEED,
            'num_nodes'      : NUM_NODES,
            'min_samples'    : MIN_SAMPLES,
            'partition_attr' : PARTITION_ATTR,
            'node_indices'   : [idx.tolist() for idx in node_indices],
        }, f)

    # ── Save per-combination results JSON ─────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, f'step3_{ALPHA}_{PARTITION_ATTR}_partition_stats.json')
    with open(results_path, 'w') as f:
        json.dump({
            'experiment'     : EXP_NAME,
            'alpha'          : ALPHA,
            'seed'           : PART_SEED,
            'num_nodes'      : NUM_NODES,
            'min_samples'    : MIN_SAMPLES,
            'partition_attr' : PARTITION_ATTR,
            'sensitive_attr' : SENSITIVE_ATTR,
            'target_attr'    : TARGET_ATTR,
            'partition_file' : partition_path,
            'global': {
                'n'                : N,
                'pct_part_pos'     : float(global_pct_pos),
                'pct_part_neg'     : float(1 - global_pct_pos),
                'pct_smiling'      : float(global_pct_smiling),
                'p_smile_sens_pos' : float(p_smile_male_global),
                'p_smile_sens_neg' : float(p_smile_female_global),
                'dp_gap_data'      : float(dp_gap_global),
            },
            'nodes': node_stats,
            'pca': {
                'explained_variance_pc1' : float(explained[0]),
                'explained_variance_pc2' : float(explained[1]),
                'top_pc1_attrs'          : top_pc1,
                'top_pc2_attrs'          : top_pc2,
                'coords'                 : coords.tolist(),
            }
        }, f, indent=2)

    return {
        'partition_attr'       : PARTITION_ATTR,
        'alpha'                : ALPHA,
        'global_pct_pos'       : global_pct_pos,
        'partition_skew_range' : max(pct_pos_per_node) - min(pct_pos_per_node),
        'dp_gap_range'         : max(dp_gaps) - min(dp_gaps),
        'dp_gap_std'           : float(np.std(dp_gaps)),
        'node_size_std'        : float(np.std([s['n'] for s in node_stats])),
        'dirichlet_attempts'   : attempts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Sweep over alpha × attribute
# ─────────────────────────────────────────────────────────────────────────────
log.info("─" * 70)
log.info(f"  SECTION 2: Sweep  "
         f"({len(ALPHA_VALUES)} alphas × {len(ATTRS_TO_RUN)} attrs = "
         f"{len(ALPHA_VALUES)*len(ATTRS_TO_RUN)} combos)")
log.info("─" * 70)

sweep_results = []
total = len(ALPHA_VALUES) * len(ATTRS_TO_RUN)
count = 0
for alpha in ALPHA_VALUES:
    log.info(f"\n  ── α = {alpha} {'─'*50}")
    for attr in ATTRS_TO_RUN:
        count += 1
        log.info(f"  [{count}/{total}]  α={alpha}  attr={attr}")
        summary = process_partition_attr(attr, alpha)
        sweep_results.append(summary)
        log.info(f"         skew_range={summary['partition_skew_range']:.3f}  "
                 f"dp_gap_range={summary['dp_gap_range']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Summary heatmap (only for full sweeps)
# ─────────────────────────────────────────────────────────────────────────────
if len(ATTRS_TO_RUN) > 1 and len(ALPHA_VALUES) > 1:
    log.info(f"\n{'─'*70}")
    log.info("  SECTION 3: Summary Heatmaps")
    log.info(f"{'─'*70}\n")

    # Build 2D matrices: rows = attrs (sorted by skew at largest alpha),
    #                    cols = alphas (sorted descending, i.e. most → least IID)
    alphas_sorted = sorted(ALPHA_VALUES, reverse=True)
    # Sort attrs by skew_range at the largest alpha for a stable, informative order
    ref_alpha   = max(ALPHA_VALUES)
    ref_skews   = {r['partition_attr']: r['partition_skew_range']
                   for r in sweep_results if r['alpha'] == ref_alpha}
    attrs_sorted = sorted(ATTRS_TO_RUN, key=lambda a: ref_skews.get(a, 0), reverse=True)

    def make_matrix(metric_key):
        mat = np.zeros((len(attrs_sorted), len(alphas_sorted)))
        for r in sweep_results:
            ri = attrs_sorted.index(r['partition_attr'])
            ci = alphas_sorted.index(r['alpha'])
            mat[ri, ci] = r[metric_key]
        return mat

    skew_mat    = make_matrix('partition_skew_range')
    dp_mat      = make_matrix('dp_gap_range')
    size_mat    = make_matrix('node_size_std')

    alpha_labels = [f'α={a}' for a in alphas_sorted]
    n_attrs = len(attrs_sorted)

    fig, axes = plt.subplots(1, 3, figsize=(6 * len(alphas_sorted) + 4, max(8, n_attrs * 0.35)))
    fig.suptitle(
        f'Partition Sweep Summary  ({NUM_NODES} nodes, seed={PART_SEED})\n'
        f'Rows sorted by partition skew range at α={ref_alpha} (↑ = more heterogeneous)',
        fontsize=13, fontweight='bold')

    def plot_heatmap(ax, mat, title, cmap, fmt='.2f'):
        im = ax.imshow(mat, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(alphas_sorted)))
        ax.set_xticklabels(alpha_labels, fontsize=9)
        ax.set_yticks(range(n_attrs))
        ax.set_yticklabels(attrs_sorted, fontsize=8)
        ax.set_title(title, fontweight='bold', fontsize=10)
        for i in range(n_attrs):
            for j in range(len(alphas_sorted)):
                ax.text(j, i, f'{mat[i,j]:{fmt}}',
                        ha='center', va='center', fontsize=7,
                        color='white' if mat[i,j] > mat.max()*0.6 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    plot_heatmap(axes[0], skew_mat,
                 'Partition Skew Range\n(max−min % positive across nodes)',
                 'Blues')
    plot_heatmap(axes[1], dp_mat,
                 f'DP Gap Range Across Nodes\n(|P({TARGET_ATTR}|{SENSITIVE_ATTR}=1) − P({TARGET_ATTR}|{SENSITIVE_ATTR}=0)|)',
                 'Reds')
    plot_heatmap(axes[2], size_mat / 1000,
                 'Node Size Std  (×1000 samples)\n(higher = more unequal node sizes)',
                 'Greens', fmt='.1f')

    plt.tight_layout()
    summary_plot_path = os.path.join(PLOT_DIR, 'step3_sweep_summary.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Summary heatmap → {summary_plot_path}")

    # Print top-5 per alpha
    for alpha in alphas_sorted:
        alpha_results = sorted(
            [r for r in sweep_results if r['alpha'] == alpha],
            key=lambda r: r['partition_skew_range'], reverse=True)
        log.info(f"\n  Top 5 attrs by skew range  (α={alpha}):")
        log.info(f"  {'Attribute':<25} {'Skew range':>12} {'DP gap range':>14}")
        log.info(f"  {'─'*25} {'─'*12} {'─'*14}")
        for r in alpha_results[:5]:
            log.info(f"  {r['partition_attr']:<25} "
                     f"{r['partition_skew_range']:>12.3f} "
                     f"{r['dp_gap_range']:>14.4f}")

elif len(ATTRS_TO_RUN) > 1:
    # Single alpha — bar chart summary (same as before)
    attrs          = [r['partition_attr']      for r in sweep_results]
    skew_ranges    = [r['partition_skew_range'] for r in sweep_results]
    dp_ranges      = [r['dp_gap_range']         for r in sweep_results]
    size_stds      = [r['node_size_std']        for r in sweep_results]

    order       = np.argsort(skew_ranges)[::-1]
    attrs_s     = [attrs[i]     for i in order]
    skew_s      = [skew_ranges[i] for i in order]
    dp_s        = [dp_ranges[i]   for i in order]
    size_s      = [size_stds[i]   for i in order]
    n_attrs     = len(attrs_s)
    y           = np.arange(n_attrs)

    fig, axes = plt.subplots(1, 3, figsize=(20, max(8, n_attrs * 0.35)))
    fig.suptitle(
        f'Partition Attribute Sweep  (α={ALPHA_VALUES[0]}, {NUM_NODES} nodes)\n'
        f'Sorted by partition skew range',
        fontsize=13, fontweight='bold')
    axes[0].barh(y, skew_s, color='steelblue', edgecolor='white')
    axes[0].set_yticks(y); axes[0].set_yticklabels(attrs_s, fontsize=8)
    axes[0].set_xlabel('Skew range (max − min % positive)')
    axes[0].set_title('Partition Attribute Skew Range')
    axes[0].spines[['top','right']].set_visible(False)
    axes[1].barh(y, dp_s, color='salmon', edgecolor='white')
    axes[1].set_yticks(y); axes[1].set_yticklabels(attrs_s, fontsize=8)
    axes[1].set_xlabel('DP gap range across nodes')
    axes[1].set_title('DP Gap Range Across Nodes')
    axes[1].spines[['top','right']].set_visible(False)
    axes[2].barh(y, size_s, color='mediumseagreen', edgecolor='white')
    axes[2].set_yticks(y); axes[2].set_yticklabels(attrs_s, fontsize=8)
    axes[2].set_xlabel('Std of node sizes (samples)')
    axes[2].set_title('Node Size Imbalance')
    axes[2].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    summary_plot_path = os.path.join(PLOT_DIR, 'step3_sweep_summary.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Summary plot → {summary_plot_path}")

# ── Save sweep summary JSON ───────────────────────────────────────────────────
if len(sweep_results) > 1:
    sweep_json_path = os.path.join(RESULTS_DIR, 'step3_sweep_summary.json')
    with open(sweep_json_path, 'w') as f:
        json.dump({
            'alpha_values' : ALPHA_VALUES,
            'num_nodes'    : NUM_NODES,
            'seed'         : PART_SEED,
            'results'      : sweep_results,
        }, f, indent=2)
    log.info(f"  ✓ Sweep summary JSON → {sweep_json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'='*70}")
log.info("  Step 3 Complete")
log.info(f"{'='*70}")
log.info(f"\n  Processed {len(sweep_results)} combination(s)  "
         f"({len(ALPHA_VALUES)} alpha(s) × {len(ATTRS_TO_RUN)} attr(s))")
log.info(f"  Outputs saved to: {EXP_DIR}")
log.info(f"    plots/{RUN_DATE}/alpha{{α}}/{{attr}}/   — per-combination plots (7 each)")
if len(sweep_results) > 1:
    log.info(f"    plots/{RUN_DATE}/step3_sweep_summary.png")
    log.info(f"    results/{RUN_DATE}/step3_sweep_summary.json")
log.info(f"    results/{RUN_DATE}/step3_{{alpha}}_{{attr}}_partition_stats.json")
log.info(f"    partitions/{RUN_DATE}/partition_alpha{{alpha}}_seed{{seed}}_{{attr}}.json")
log.info(f"    logs/{RUN_DATE}/step3.log")
log.info(f"\n  Next → python step_4_train_5_models_on_data_paritions.py --config {args.config}")
log.info(f"{'='*70}")