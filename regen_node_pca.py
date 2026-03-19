"""
Node Distribution PCA using all 40 CelebA attributes.

For each node, computes the mean of every binary CelebA attribute
(fraction of samples with attribute=1), giving a 40-dim feature vector.
PCA projects these onto 2 components.

Produces two plots:
  step5b_node_pca_40attr_{PARTITION_ATTR}.png
    - Left:  node scatter in PC1/PC2 space
    - Right: top-N feature loadings for PC1 and PC2 (bar chart)

Usage:
    python3 regen_node_pca.py --config config.yaml
"""

import os
import json
import argparse
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--config',    type=str, default='config.yaml')
parser.add_argument('--top-n',     type=int, default=15,
                    help='Number of top features to show in loading bar chart')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

ALPHA          = cfg['partition']['alpha']
PART_SEED      = cfg['partition']['seed']
NUM_NODES      = cfg['partition']['num_nodes']
PARTITION_ATTR = cfg['partition']['partition_attr']
RUN_DATE       = cfg['experiment'].get('run_date') or datetime.date.today().isoformat()
EXP_NAME       = (cfg['experiment'].get('name')
                  or f"lenet_alpha{ALPHA}_{NUM_NODES}nodes_seed{PART_SEED}")
NFS_ROOT       = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR        = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
PLOT_DIR       = os.path.join(EXP_DIR, 'plots', RUN_DATE)
os.makedirs(PLOT_DIR, exist_ok=True)

NODE_COLORS = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']

# ── Load partition indices ─────────────────────────────────────────────────
partition_fname = f'partition_alpha{ALPHA}_seed{PART_SEED}_{PARTITION_ATTR}.json'
partition_path  = os.path.join(EXP_DIR, 'partitions', RUN_DATE, partition_fname)
with open(partition_path) as f:
    partition_data = json.load(f)
node_indices = [np.array(idx, dtype=np.int64)
                for idx in partition_data['node_indices']]

# ── Load dataset — attributes only, no images ─────────────────────────────
HF_CACHE = os.environ.get('HF_DATASETS_CACHE',
                           os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE
print("Loading CelebA attributes...")
dataset = load_dataset(cfg['dataset']['name'], split='train')

# Identify all binary attribute columns (exclude 'image' and index columns)
sample = dataset[0]
attr_names = sorted([
    k for k, v in sample.items()
    if k != 'image' and isinstance(v, (bool, int)) and v in (0, 1, True, False)
])
print(f"Found {len(attr_names)} binary attributes: {attr_names}")

# Build attribute matrix: shape (N_samples, N_attrs)
print("Extracting attribute columns...")
attr_matrix = np.column_stack([
    np.array(dataset[a], dtype=np.float32) for a in attr_names
])  # shape: (N, 40)

# ── Per-node mean of each attribute ───────────────────────────────────────
# X shape: (NUM_NODES, 40)
X = np.array([attr_matrix[idx].mean(axis=0) for idx in node_indices])
print(f"Feature matrix: {X.shape}  ({NUM_NODES} nodes × {len(attr_names)} attrs)")

# ── PCA ───────────────────────────────────────────────────────────────────
X_scaled = StandardScaler().fit_transform(X)
n_components = min(NUM_NODES, len(attr_names), 2)
pca    = PCA(n_components=n_components)
coords = pca.fit_transform(X_scaled)
var    = pca.explained_variance_ratio_
loadings = pca.components_   # shape: (n_components, 40)

print(f"PC1 explains {var[0]:.1%} variance")
if len(var) > 1:
    print(f"PC2 explains {var[1]:.1%} variance")

# ── Plot 1: Node scatter in PC space ──────────────────────────────────────
top_n = min(args.top_n, len(attr_names))

fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle(f'Node Distribution Space — PCA over all {len(attr_names)} CelebA attributes\n'
             f'Each point = one node (mean attribute prevalence as feature vector)',
             fontsize=11, fontweight='bold')
for i in range(NUM_NODES):
    x = coords[i, 0]
    y = coords[i, 1] if coords.shape[1] > 1 else 0.0
    ax.scatter(x, y, color=NODE_COLORS[i], s=250, zorder=3)
    ax.annotate(f'Node {i+1}', xy=(x, y),
                xytext=(6, 4), textcoords='offset points', fontsize=10)
ax.set_xlabel(f'PC1  ({var[0]:.1%} var)')
ax.set_ylabel(f'PC2  ({var[1]:.1%} var)' if len(var) > 1 else 'PC2  (0.0% var)')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
out = os.path.join(PLOT_DIR, f'step5b_node_pca_scatter_{PARTITION_ATTR}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

# ── Plot 2: PC1 loadings ───────────────────────────────────────────────────
order1  = np.argsort(np.abs(loadings[0]))[::-1][:top_n]
names1  = [attr_names[i] for i in order1]
vals1   = loadings[0][order1]
colors1 = ['steelblue' if v >= 0 else 'tomato' for v in vals1]

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(range(top_n), vals1[::-1], color=colors1[::-1])
ax.set_yticks(range(top_n))
ax.set_yticklabels(names1[::-1], fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Loading')
ax.set_title(f'PC1 ({var[0]:.1%} variance) — top {top_n} feature loadings\n'
             f'blue = positive, red = negative',
             fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
out = os.path.join(PLOT_DIR, f'step5b_node_pca_pc1_loadings_{PARTITION_ATTR}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")

# ── Plot 3: PC2 loadings ───────────────────────────────────────────────────
if len(var) > 1:
    order2  = np.argsort(np.abs(loadings[1]))[::-1][:top_n]
    names2  = [attr_names[i] for i in order2]
    vals2   = loadings[1][order2]
    colors2 = ['steelblue' if v >= 0 else 'tomato' for v in vals2]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(top_n), vals2[::-1], color=colors2[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names2[::-1], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Loading')
    ax.set_title(f'PC2 ({var[1]:.1%} variance) — top {top_n} feature loadings\n'
                 f'blue = positive, red = negative',
                 fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5b_node_pca_pc2_loadings_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")
