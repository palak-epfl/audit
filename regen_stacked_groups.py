"""
Regenerate step3_stacked_groups.png with a neutral color scheme.
Replaces pink/magenta for non-male with orange tones.

Usage:
    python3 regen_stacked_groups.py --config config.yaml
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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

ALPHA          = cfg['partition']['alpha']
PART_SEED      = cfg['partition']['seed']
NUM_NODES      = cfg['partition']['num_nodes']
PARTITION_ATTR = cfg['partition']['partition_attr']
SENSITIVE_ATTR = cfg['dataset']['sensitive_attr']
TARGET_ATTR    = cfg['dataset']['target_attr']
RUN_DATE       = cfg['experiment'].get('run_date') or datetime.date.today().isoformat()
EXP_NAME       = (cfg['experiment'].get('name')
                  or f"lenet_alpha{ALPHA}_{NUM_NODES}nodes_seed{PART_SEED}")
NFS_ROOT       = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR        = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
PLOT_DIR       = os.path.join(EXP_DIR, 'plots', RUN_DATE, PARTITION_ATTR)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load partition indices ─────────────────────────────────────────────────
partition_fname = f'partition_alpha{ALPHA}_seed{PART_SEED}_{PARTITION_ATTR}.json'
partition_path  = os.path.join(EXP_DIR, 'partitions', RUN_DATE, partition_fname)
with open(partition_path) as f:
    partition_data = json.load(f)
node_indices = [np.array(idx, dtype=np.int64)
                for idx in partition_data['node_indices']]

# ── Load only attribute columns (no images) ────────────────────────────────
print("Loading dataset attributes...")
HF_CACHE = os.environ.get('HF_DATASETS_CACHE',
                           os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE
dataset = load_dataset(cfg['dataset']['name'], split='train')

gender       = np.array(dataset[SENSITIVE_ATTR], dtype=np.int64)
smiling      = np.array(dataset[TARGET_ATTR],    dtype=np.int64)
part_attr    = np.array(dataset[PARTITION_ATTR], dtype=np.int64)

# Full dataset as the last "node" for reference
all_indices = node_indices + [np.arange(len(dataset), dtype=np.int64)]
all_labels  = [f'Node {i+1}' for i in range(NUM_NODES)] + ['All']

sens_pos_label = SENSITIVE_ATTR           # e.g. "Male"
sens_neg_label = f'Non-{SENSITIVE_ATTR}'  # e.g. "Non-Male"

# ── Neutral color scheme: blue for Male, orange for Non-Male ───────────────
group_colors = ['#1565C0', '#90CAF9', '#E65100', '#FFCC80']
# Previously: ['#2196F3', '#90CAF9', '#E91E63', '#F48FB1']
# (last two were pink/magenta — replaced with dark/light orange)

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

fig, ax = plt.subplots(figsize=(10, 5))
for gname, color in zip(group_labels_list, group_colors):
    vals = group_data[gname]
    ax.bar(all_labels, vals, bottom=bottoms,
           label=gname, color=color, edgecolor='white', linewidth=0.8)
    bottoms += np.array(vals)

ax.set_title(
    f'{SENSITIVE_ATTR} × {TARGET_ATTR} Group Composition per Node  (α={ALPHA})\n'
    f'Non-IID split driven by {PARTITION_ATTR}',
    fontsize=11, fontweight='bold')
ax.set_ylabel('Proportion')
ax.set_ylim(0, 1)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

for j, idx in enumerate(all_indices):
    pct_p = part_attr[idx].mean()
    ax.text(j, 1.02, f'{100*pct_p:.0f}%\n{PARTITION_ATTR[:4]}',
            ha='center', fontsize=8, color='#1565C0')

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step3_stacked_groups.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {out}")
