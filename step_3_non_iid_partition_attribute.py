"""
Step 3: Non-IID Data Partitioning + Visualisation
===================================================
Partitions CelebA across N nodes using a Dirichlet distribution to simulate
non-IID federated data heterogeneity, then produces comprehensive visualisations
showing how the partition differs from the global distribution.

The attribute driving the non-IID split is controlled by `partition.partition_attr`
in config.yaml (e.g. "High_Cheekbones", "Male", "Young").  The sensitive attribute
(for DP-gap computation) and target attribute remain as configured separately under
`dataset.sensitive_attr` and `dataset.target_attr`.

Run with:
    python3 step_3_non_iid_data_partition.py --config config.yaml

Env vars:
    NFS_ROOT=/your/nfs/path   override nfs.root from config

Produces (all saved to {nfs_root}/experiments/{exp_name}/):
    partitions/partition_alpha{α}_seed{seed}_{partition_attr}.json
    plots/step3_node_sizes.png
    plots/step3_{partition_attr}_per_node.png
    plots/step3_smiling_per_node.png
    plots/step3_stacked_groups.png
    plots/step3_joint_heatmaps.png
    plots/step3_dp_gap_per_node.png
    plots/step3_distribution_pca.png
    results/step3_partition_stats.json
    logs/step3.log
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
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

DATASET_NAME   = cfg['dataset']['name']
SENSITIVE_ATTR = cfg['dataset']['sensitive_attr']
TARGET_ATTR    = cfg['dataset']['target_attr']
EXP_NAME       = cfg['experiment']['name']
RUN_DATE       = cfg['experiment'].get('run_date') or datetime.date.today().isoformat()
NUM_NODES      = cfg['partition']['num_nodes']
ALPHA          = cfg['partition']['alpha']
MIN_SAMPLES    = cfg['partition']['min_samples']
PART_SEED      = cfg['partition']['seed']
PARTITION_ATTR = cfg['partition']['partition_attr']   # ← new: drives the split

# Human-readable labels for the partition attribute's two groups
# e.g. "High_Cheekbones" → ("High Cheekbones", "No High Cheekbones")
_attr_display  = PARTITION_ATTR.replace('_', ' ')
PART_POS_LABEL = _attr_display            # samples where attr == 1
PART_NEG_LABEL = f'No {_attr_display}'   # samples where attr == 0

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

# Validate partition attribute
if PARTITION_ATTR not in CELEBA_ATTRS:
    raise ValueError(
        f"partition_attr '{PARTITION_ATTR}' not found in the 40 CelebA attributes.\n"
        f"Valid options: {CELEBA_ATTRS}"
    )

NODE_COLORS = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']

# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("  Step 3: Non-IID Partition + Visualisation")
log.info("=" * 70)
log.info(f"\n  Config          : {args.config}")
log.info(f"  Experiment      : {EXP_NAME}")
log.info(f"  Run date        : {RUN_DATE}")
log.info(f"  NFS root        : {NFS_ROOT}")
log.info(f"  Num nodes       : {NUM_NODES}")
log.info(f"  Dirichlet α     : {ALPHA}")
log.info(f"  Min samples     : {MIN_SAMPLES}")
log.info(f"  Partition seed  : {PART_SEED}")
log.info(f"  Partition attr  : {PARTITION_ATTR}  (drives non-IID split)")
log.info(f"  Sensitive attr  : {SENSITIVE_ATTR}  (used for DP-gap computation)")
log.info(f"  Target attr     : {TARGET_ATTR}")
log.info(f"  Log file        : {log_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load dataset + all attributes
# ─────────────────────────────────────────────────────────────────────────────
log.info("─" * 70)
log.info("  SECTION 1: Loading Dataset")
log.info("─" * 70)

log.info("\n  Loading CelebA (train split)...")
dataset = load_dataset(DATASET_NAME, split='train')
N = len(dataset)
log.info(f"  ✓ Loaded {N:,} samples")

# Sensitive attribute (for DP gap) and target attribute
gender  = np.array(dataset[SENSITIVE_ATTR], dtype=np.int64)
smiling = np.array(dataset[TARGET_ATTR],    dtype=np.int64)

# Partition-driving attribute
part_attr_vals = np.array(dataset[PARTITION_ATTR], dtype=np.int64)
global_pct_pos = float(part_attr_vals.mean())
log.info(f"  {PARTITION_ATTR}: {100*global_pct_pos:.1f}% positive  "
         f"({(part_attr_vals==1).sum():,} samples with attr=1, "
         f"{(part_attr_vals==0).sum():,} with attr=0)")

# Load all 40 attributes into a matrix — used for distribution-space PCA
log.info("  Loading all 40 attributes for distribution PCA...")
attr_matrix = np.zeros((N, len(CELEBA_ATTRS)), dtype=np.float32)
for j, attr in enumerate(CELEBA_ATTRS):
    attr_matrix[:, j] = np.array(dataset[attr], dtype=np.float32)
log.info(f"  ✓ Attribute matrix shape: {attr_matrix.shape}  (samples × attributes)\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Dirichlet partition
# ─────────────────────────────────────────────────────────────────────────────
log.info("─" * 70)
log.info("  SECTION 2: Dirichlet Non-IID Partitioning")
log.info("─" * 70)
log.info(f"""
  Partition attribute : {PARTITION_ATTR}
  Method: Dirichlet(α={ALPHA}) applied separately to
          '{PART_POS_LABEL}' and '{PART_NEG_LABEL}' indices.

  How it works:
    1. Split all indices into two groups based on {PARTITION_ATTR}
       (attr=1 → '{PART_POS_LABEL}',  attr=0 → '{PART_NEG_LABEL}')
    2. Sample a Dirichlet(α) proportion vector for each group
       → this gives each node a different fraction of the two groups
    3. Assign indices to nodes according to those proportions
    4. Each node gets a non-overlapping subset of the full dataset

  Effect of α:
    α → 0    : extreme heterogeneity — each node gets almost all of one group
    α = 0.5  : moderate heterogeneity (your current setting)
    α = 1.0  : mild heterogeneity
    α → ∞   : IID — all nodes get the same {PARTITION_ATTR} distribution

  Note: Sensitive attr ({SENSITIVE_ATTR}) and DP-gap computation are unchanged.
        The skew introduced is in {PARTITION_ATTR}, not {SENSITIVE_ATTR}.

  Min samples constraint ({MIN_SAMPLES:,}):
    Re-samples the Dirichlet until all nodes meet the minimum.
""")


def dirichlet_partition(part_attr_vals, N, num_nodes, alpha, min_samples, seed):
    """
    Partition dataset indices across nodes using Dirichlet distribution.
    Applied separately to positive (attr=1) and negative (attr=0) indices
    of the chosen partition attribute to control distributional skew.
    Repeats sampling until all nodes meet min_samples constraint.

    Returns:
        node_indices: list of numpy arrays, one per node
        attempts    : number of Dirichlet samples needed
    """
    rng      = np.random.default_rng(seed)
    pos_idx  = np.where(part_attr_vals == 1)[0]
    neg_idx  = np.where(part_attr_vals == 0)[0]

    attempts = 0
    while True:
        attempts += 1
        # Sample Dirichlet proportions independently for each group
        pos_props = rng.dirichlet(alpha * np.ones(num_nodes))
        neg_props = rng.dirichlet(alpha * np.ones(num_nodes))

        # Shuffle indices before splitting
        pos_shuffled = rng.permutation(pos_idx)
        neg_shuffled = rng.permutation(neg_idx)

        # Split indices according to proportions
        pos_splits = (pos_props * len(pos_idx)).astype(int)
        neg_splits = (neg_props * len(neg_idx)).astype(int)

        # Fix rounding so splits sum exactly to group size
        pos_splits[-1] = len(pos_idx) - pos_splits[:-1].sum()
        neg_splits[-1] = len(neg_idx) - neg_splits[:-1].sum()

        # Assign indices to each node
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


log.info("  Running Dirichlet partition...")
node_indices, attempts = dirichlet_partition(
    part_attr_vals, N, NUM_NODES, ALPHA, MIN_SAMPLES, PART_SEED
)
log.info(f"  ✓ Partition found after {attempts} attempt(s)\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Partition statistics
# ─────────────────────────────────────────────────────────────────────────────
log.info("─" * 70)
log.info("  SECTION 3: Partition Statistics")
log.info("─" * 70)

# Global stats
global_pct_male    = float(gender.mean())
global_pct_smiling = float(smiling.mean())
p_smile_male_global   = float(smiling[gender == 1].mean())
p_smile_female_global = float(smiling[gender == 0].mean())
dp_gap_global = abs(p_smile_male_global - p_smile_female_global)

# Sensitive attr labels (for display)
sens_pos_label = SENSITIVE_ATTR.replace('_', ' ')          # e.g. "Male"
sens_neg_label = f'Non-{sens_pos_label}'                   # e.g. "Non-Male"

node_stats = []

# ── Table 1: original sensitive-attr view (matches original script exactly) ──
log.info(f"\n  {'':8} {'N':>7} {'% Male':>8} {'% Female':>9} "
         f"{'% Smiling':>10} {'DP gap':>8} {'P(Smile|M)':>11} {'P(Smile|F)':>11}")
log.info(f"  {'Global':<8} {N:>7,} {100*global_pct_male:>7.1f}% "
         f"{100*(1-global_pct_male):>8.1f}% {100*global_pct_smiling:>9.1f}% "
         f"{dp_gap_global:>8.4f} {p_smile_male_global:>11.4f} "
         f"{p_smile_female_global:>11.4f}")
log.info("  " + "-" * 82)

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

    # 40-attribute proportion vector for PCA
    attr_props = attr_matrix[idx].mean(axis=0).tolist()

    st = {
        'node_id'           : i + 1,
        'n'                 : int(n),
        'n_part_pos'        : int((node_part_attr == 1).sum()),
        'n_part_neg'        : int((node_part_attr == 0).sum()),
        'pct_part_pos'      : pct_pos,
        'pct_part_neg'      : 1.0 - pct_pos,
        'n_sens_pos'        : int(n_sens_pos),
        'n_sens_neg'        : int(n_sens_neg),
        'pct_male'          : pct_male,
        'pct_female'        : 1.0 - pct_male,
        'pct_smiling'       : pct_smiling,
        'p_smile_sens_pos'  : p_sm,
        'p_smile_sens_neg'  : p_sf,
        'dp_gap_data'       : dp_gap,
        'attr_props'        : attr_props,
    }
    node_stats.append(st)

    log.info(f"  Node {i+1:<4} {n:>7,} {100*pct_male:>7.1f}% "
             f"{100*(1-pct_male):>8.1f}% {100*pct_smiling:>9.1f}% "
             f"{dp_gap:>8.4f} {p_sm:>11.4f} {p_sf:>11.4f}")

# ── Table 2: partition attribute breakdown (new) ──────────────────────────────
log.info(f"\n  Partition attribute breakdown  ({PARTITION_ATTR}):")
log.info(f"  {'':8} {'N':>7} {'% '+PART_POS_LABEL:>22} {'% '+PART_NEG_LABEL:>24}")
log.info(f"  {'Global':<8} {N:>7,} "
         f"{100*global_pct_pos:>21.1f}% "
         f"{100*(1-global_pct_pos):>23.1f}%")
log.info("  " + "-" * 62)
for s in node_stats:
    log.info(f"  Node {s['node_id']:<4} {s['n']:>7,} "
             f"{100*s['pct_part_pos']:>21.1f}% "
             f"{100*s['pct_part_neg']:>23.1f}%")

pct_pos_per_node = [s['pct_part_pos'] for s in node_stats]
dp_gaps          = [s['dp_gap_data']  for s in node_stats]
pct_males        = [s['pct_male']     for s in node_stats]

log.info(f"\n  Key observations:")
log.info(f"  {PARTITION_ATTR} skew range : "
         f"{100*min(pct_pos_per_node):.1f}% – {100*max(pct_pos_per_node):.1f}% "
         f"positive across nodes")
log.info(f"  Gender skew range  : {100*min(pct_males):.1f}% – {100*max(pct_males):.1f}% male across nodes")
log.info(f"  DP gap range       : {min(dp_gaps):.4f} – {max(dp_gaps):.4f} across nodes")
log.info(f"  Global DP gap      : {dp_gap_global:.4f}")
log.info(f"  Node with most {PARTITION_ATTR[:8]}  : Node {np.argmax(pct_pos_per_node)+1} "
         f"({100*max(pct_pos_per_node):.1f}%)")
log.info(f"  Node with least {PARTITION_ATTR[:8]} : Node {np.argmin(pct_pos_per_node)+1} "
         f"({100*min(pct_pos_per_node):.1f}%)")
log.info(f"  Node with most male   : Node {np.argmax(pct_males)+1} ({100*max(pct_males):.1f}%)")
log.info(f"  Node with least male  : Node {np.argmin(pct_males)+1} ({100*min(pct_males):.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Save partition to NFS
# ─────────────────────────────────────────────────────────────────────────────
partition_fname = f'partition_alpha{ALPHA}_seed{PART_SEED}_{PARTITION_ATTR}.json'
partition_path  = os.path.join(PARTITION_DIR, partition_fname)

partition_data = {
    'alpha'          : ALPHA,
    'seed'           : PART_SEED,
    'num_nodes'      : NUM_NODES,
    'min_samples'    : MIN_SAMPLES,
    'partition_attr' : PARTITION_ATTR,
    'node_indices'   : [idx.tolist() for idx in node_indices],
}
with open(partition_path, 'w') as f:
    json.dump(partition_data, f)
log.info(f"\n  ✓ Partition saved → {partition_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Visualisations
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'─'*70}")
log.info("  SECTION 5: Generating Visualisations")
log.info(f"{'─'*70}\n")

node_labels = [f'Node {i+1}' for i in range(NUM_NODES)]
x = np.arange(NUM_NODES)

# ── Plot 1: Node sizes ─────────────────────────────────────────────────────────
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
out = os.path.join(PLOT_DIR, 'step3_node_sizes.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 2: Partition attribute distribution per node ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
pct_pos_vals = [s['pct_part_pos'] for s in node_stats]
pct_neg_vals = [s['pct_part_neg'] for s in node_stats]
width = 0.35
bars_pos = ax.bar(x - width/2, [p*100 for p in pct_pos_vals],
                  width, label=PART_POS_LABEL,   color='steelblue', edgecolor='white')
bars_neg = ax.bar(x + width/2, [p*100 for p in pct_neg_vals],
                  width, label=PART_NEG_LABEL, color='salmon',    edgecolor='white')
ax.axhline(100*global_pct_pos, color='steelblue', linestyle='--',
           alpha=0.5, label=f'Global {PART_POS_LABEL} ({100*global_pct_pos:.1f}%)')
ax.axhline(100*(1-global_pct_pos), color='salmon', linestyle='--',
           alpha=0.5, label=f'Global {PART_NEG_LABEL} ({100*(1-global_pct_pos):.1f}%)')
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
out = os.path.join(PLOT_DIR, f'step3_{PARTITION_ATTR}_per_node.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 3: Smiling distribution per node ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
pct_smile_per_node = [s['pct_smiling'] for s in node_stats]
bars = ax.bar(node_labels, [p*100 for p in pct_smile_per_node],
              color=NODE_COLORS, edgecolor='white', linewidth=1.2)
ax.axhline(100*global_pct_smiling, color='gray', linestyle='--',
           alpha=0.7, label=f'Global ({100*global_pct_smiling:.1f}%)')
ax.set_title(
    f'{TARGET_ATTR} Rate per Node  (α={ALPHA})\n'
    f'Partitioned by {PART_POS_LABEL}  |  '
    f'Indirect skew driven by {PARTITION_ATTR}–{TARGET_ATTR} correlation',
    fontsize=12, fontweight='bold')
ax.set_ylabel(f'% {TARGET_ATTR}')
ax.set_ylim(0, 100)
for bar, v in zip(bars, pct_smile_per_node):
    ax.text(bar.get_x() + bar.get_width()/2, v*100 + 0.5,
            f'{100*v:.1f}%', ha='center', fontsize=9)
ax.legend()
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step3_smiling_per_node.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 4: Stacked group proportions ─────────────────────────────────────────
# Groups are defined by sensitive_attr × target_attr (unchanged from original).
# Partitioning axis is noted in the title for clarity.
fig, ax = plt.subplots(figsize=(10, 5))

_sp = sens_pos_label   # e.g. "Male"
_sn = sens_neg_label   # e.g. "Non-Male"
group_colors = ['#2196F3', '#90CAF9', '#E91E63', '#F48FB1']
group_labels = [
    f'{_sp} + {TARGET_ATTR}',
    f'{_sp} + Not {TARGET_ATTR}',
    f'{_sn} + {TARGET_ATTR}',
    f'{_sn} + Not {TARGET_ATTR}',
]

all_labels  = ['Global'] + node_labels
all_indices = [np.arange(N)] + node_indices

bottoms    = np.zeros(len(all_labels))
group_data = {g: [] for g in group_labels}

for idx in all_indices:
    g = gender[idx]; s = smiling[idx]; n = len(idx)
    group_data[f'{_sp} + {TARGET_ATTR}'].append(        ((g==1)&(s==1)).sum() / n)
    group_data[f'{_sp} + Not {TARGET_ATTR}'].append(    ((g==1)&(s==0)).sum() / n)
    group_data[f'{_sn} + {TARGET_ATTR}'].append(        ((g==0)&(s==1)).sum() / n)
    group_data[f'{_sn} + Not {TARGET_ATTR}'].append(    ((g==0)&(s==0)).sum() / n)

for gname, color in zip(group_labels, group_colors):
    vals = group_data[gname]
    ax.bar(all_labels, vals, bottom=bottoms,
           label=gname, color=color, edgecolor='white', linewidth=0.8)
    bottoms += np.array(vals)

ax.set_title(
    f'{SENSITIVE_ATTR} × {TARGET_ATTR} Group Composition per Node  (α={ALPHA})\n'
    f'Non-IID split driven by {PART_POS_LABEL}  |  '
    f'Sensitive attr: {SENSITIVE_ATTR}  |  Target attr: {TARGET_ATTR}',
    fontsize=11, fontweight='bold')
ax.set_ylabel('Proportion')
ax.set_ylim(0, 1)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.spines[['top','right']].set_visible(False)

# Annotate % partition-positive on each bar
for j, (lbl, idx) in enumerate(zip(all_labels, all_indices)):
    pct_p = part_attr_vals[idx].mean()
    ax.text(j, 1.02, f'{100*pct_p:.0f}%\n{PARTITION_ATTR[:4]}',
            ha='center', fontsize=8, color='#1565C0')

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step3_stacked_groups.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 5: Joint heatmaps — sensitive_attr × target_attr per node ────────────
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
out = os.path.join(PLOT_DIR, 'step3_joint_heatmaps.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 6: DP gap per node vs global ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
dp_gaps_nodes = [s['dp_gap_data'] for s in node_stats]
bars = ax.bar(node_labels, dp_gaps_nodes,
              color=NODE_COLORS, edgecolor='white', linewidth=1.2)
ax.axhline(dp_gap_global, color='black', linestyle='--', linewidth=1.5,
           label=f'Global DP gap ({dp_gap_global:.4f})')
ax.set_title(
    f'Ground-Truth DP Gap per Node  (α={ALPHA})\n'
    f'DP gap = |P({TARGET_ATTR}|{sens_pos_label}) − P({TARGET_ATTR}|{sens_neg_label})|\n'
    f'Non-IID split driven by: {PART_POS_LABEL}  '
    f'(sensitive attr: {SENSITIVE_ATTR})',
    fontsize=11, fontweight='bold')
ax.set_ylabel('DP gap (data)')
for bar, v in zip(bars, dp_gaps_nodes):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
            f'{v:.4f}', ha='center', fontsize=9)
ax.legend()
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step3_dp_gap_per_node.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 7: Distribution-space PCA ────────────────────────────────────────────
log.info("  Computing distribution-space PCA...")
log.info(f"""
  Each node (and the global dataset) is represented as a 40-dimensional
  vector of attribute prevalences — e.g. [% {PARTITION_ATTR}, % {TARGET_ATTR}, % Young, ...].
  PCA projects these {NUM_NODES+1} points ({NUM_NODES} nodes + global) into 2D.

  What this shows:
    - Nodes close together have similar attribute distributions
    - Nodes far from 'Global' are most distributionally shifted
    - The axes capture the directions of greatest variance across nodes
    - Spread is expected along the {PARTITION_ATTR} axis given how partitioning works
  """)

prop_vectors = []
prop_vectors.append(attr_matrix.mean(axis=0))        # global
for idx in node_indices:
    prop_vectors.append(attr_matrix[idx].mean(axis=0))

prop_vectors = np.array(prop_vectors)   # (NUM_NODES+1, 40)

pca = PCA(n_components=2, random_state=42)
pca.fit(prop_vectors[1:])               # fit on nodes only
coords = pca.transform(prop_vectors)    # transform all including global

explained = pca.explained_variance_ratio_
log.info(f"  PCA explained variance: PC1={100*explained[0]:.1f}%,  "
         f"PC2={100*explained[1]:.1f}%  "
         f"(total={100*sum(explained):.1f}%)")

pc1_loadings = np.abs(pca.components_[0])
pc2_loadings = np.abs(pca.components_[1])
top_pc1 = [CELEBA_ATTRS[i] for i in np.argsort(pc1_loadings)[::-1][:5]]
top_pc2 = [CELEBA_ATTRS[i] for i in np.argsort(pc2_loadings)[::-1][:5]]
log.info(f"  Top PC1 attributes: {top_pc1}")
log.info(f"  Top PC2 attributes: {top_pc2}")

fig, ax = plt.subplots(figsize=(9, 7))

for i, (x_coord, y_coord) in enumerate(coords[1:]):
    pct_p = node_stats[i]['pct_part_pos']
    ax.scatter(x_coord, y_coord, color=NODE_COLORS[i], s=200, zorder=5,
               edgecolors='white', linewidth=1.5)
    ax.annotate(
        f'Node {i+1}\n({100*pct_p:.0f}% {PARTITION_ATTR[:8]})',
        xy=(x_coord, y_coord),
        xytext=(x_coord + 0.0003, y_coord + 0.0003),
        fontsize=10, fontweight='bold', color=NODE_COLORS[i])

ax.scatter(coords[0, 0], coords[0, 1], color='black', s=300,
           marker='*', zorder=6, label='Global dataset')
ax.annotate('Global', xy=(coords[0, 0], coords[0, 1]),
            xytext=(coords[0, 0] + 0.0003, coords[0, 1] + 0.0003),
            fontsize=10, fontweight='bold')

for i, (x_coord, y_coord) in enumerate(coords[1:]):
    ax.plot([coords[0,0], x_coord], [coords[0,1], y_coord],
            color=NODE_COLORS[i], alpha=0.3, linestyle='--', linewidth=1)

ax.set_xlabel(f'PC1 ({100*explained[0]:.1f}% variance)\n'
              f'Top attrs: {", ".join(top_pc1[:3])}', fontsize=9)
ax.set_ylabel(f'PC2 ({100*explained[1]:.1f}% variance)\n'
              f'Top attrs: {", ".join(top_pc2[:3])}', fontsize=9)
ax.set_title(
    f'Distribution-Space PCA  (α={ALPHA})\n'
    f'Each point = one node\'s 40-attribute prevalence vector\n'
    f'Non-IID split by: {PART_POS_LABEL}  |  '
    f'Distance from Global = degree of distributional shift',
    fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step3_distribution_pca.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Save results JSON
# ─────────────────────────────────────────────────────────────────────────────
results = {
    'experiment'      : EXP_NAME,
    'alpha'           : ALPHA,
    'seed'            : PART_SEED,
    'num_nodes'       : NUM_NODES,
    'min_samples'     : MIN_SAMPLES,
    'partition_attr'  : PARTITION_ATTR,
    'sensitive_attr'  : SENSITIVE_ATTR,
    'target_attr'     : TARGET_ATTR,
    'partition_file'  : partition_path,
    'global': {
        'n'                 : N,
        'pct_part_pos'      : float(global_pct_pos),
        'pct_part_neg'      : float(1 - global_pct_pos),
        'pct_smiling'       : float(global_pct_smiling),
        'p_smile_sens_pos'  : float(p_smile_male_global),
        'p_smile_sens_neg'  : float(p_smile_female_global),
        'dp_gap_data'       : float(dp_gap_global),
    },
    'nodes'           : node_stats,
    'pca': {
        'explained_variance_pc1' : float(explained[0]),
        'explained_variance_pc2' : float(explained[1]),
        'top_pc1_attrs'          : top_pc1,
        'top_pc2_attrs'          : top_pc2,
        'coords'                 : coords.tolist(),
    }
}

results_path = os.path.join(RESULTS_DIR, 'step3_partition_stats.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
log.info(f"\n  ✓ Results saved → {results_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'='*70}")
log.info("  Step 3 Complete — Summary")
log.info(f"{'='*70}")
log.info(f"\n  Partition attr  : {PARTITION_ATTR}  (drives non-IID split)")
log.info(f"  Sensitive attr  : {SENSITIVE_ATTR}  (used for DP-gap)")
log.info(f"  Target attr     : {TARGET_ATTR}")
log.info(f"  α={ALPHA}, seed={PART_SEED}, {NUM_NODES} nodes")
log.info(f"  {PARTITION_ATTR} skew : "
         f"{100*min(pct_pos_per_node):.1f}% – {100*max(pct_pos_per_node):.1f}% "
         f"positive across nodes")
log.info(f"  Gender skew     : {100*min(pct_males):.1f}% – {100*max(pct_males):.1f}% male across nodes")
log.info(f"  DP gap range    : {min(dp_gaps):.4f} – {max(dp_gaps):.4f}  "
         f"(global: {dp_gap_global:.4f})")
log.info(f"\n  Outputs saved to: {EXP_DIR}")
log.info(f"    partitions/{partition_fname}")
log.info(f"    plots/step3_node_sizes.png")
log.info(f"    plots/step3_{PARTITION_ATTR}_per_node.png")
log.info(f"    plots/step3_smiling_per_node.png")
log.info(f"    plots/step3_stacked_groups.png")
log.info(f"    plots/step3_joint_heatmaps.png")
log.info(f"    plots/step3_dp_gap_per_node.png")
log.info(f"    plots/step3_distribution_pca.png")
log.info(f"    results/step3_partition_stats.json")
log.info(f"    logs/step3.log")
log.info(f"\n  Next → python step4_train_nodes.py --config {args.config}")
log.info(f"{'='*70}")