"""
Regenerate error heatmaps for all audit modes.

Reads from step5_audit_results_{attr}.json (full local, global) and
step5c_audit_results_{attr}.json (collab pairs, collab triples) if present.

Produces one heatmap per mode × ground truth:
  step5_heatmap_full_local_{attr}_{gt}.png
  step5_heatmap_global_{attr}_{gt}.png
  step5_heatmap_collab_pairs_{attr}_{gt}.png
  step5_heatmap_collab_triples_{attr}_{gt}.png

Each heatmap: rows = auditor / coalition, cols = target node.
Cell value = absolute error, annotated in each cell.

Usage:
    python3 regen_error_heatmaps.py --config config.yaml
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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')
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
PLOT_DIR       = os.path.join(EXP_DIR, 'plots',   RUN_DATE)
RESULTS_DIR    = os.path.join(EXP_DIR, 'results', RUN_DATE)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load results ───────────────────────────────────────────────────────────
step5_path = os.path.join(RESULTS_DIR, f'step5_audit_results_{PARTITION_ATTR}.json')
with open(step5_path) as f:
    data5 = json.load(f)

full_results        = data5['full_results']
global_all_results  = data5['global_all_results']
global_excl_results = data5['global_excl_results']
collab_pair_results = data5.get('collab_pair_results', [])

step5c_path = os.path.join(RESULTS_DIR, f'step5c_audit_results_{PARTITION_ATTR}.json')
collab_triple_results = []
if os.path.exists(step5c_path):
    with open(step5c_path) as f:
        data5c = json.load(f)
    collab_triple_results = data5c.get('collab_triple_results', [])
    if not collab_pair_results:
        collab_pair_results = data5c.get('collab_pair_results', [])
    print(f"Loaded step5c — triples: {len(collab_triple_results)}")

target_ids = sorted(set(r['target_id'] for r in full_results))

# Ground truth error field mapping
GT_FIELDS = {
    'model_val' : ('abs_error',            'abs_error_model_val',  'GT: Model Val'),
    'data'      : ('abs_error_data',        'abs_error_data',       'GT: Data Labels'),
    'model_full': ('abs_error_model_full',  'abs_error_model_full', 'GT: Model Full'),
}


def save_heatmap(matrix, row_labels, col_labels, title, out_path):
    n_rows, n_cols = matrix.shape
    fig_w = max(6, n_cols * 1.2)
    fig_h = max(3, n_rows * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f'Target {c}' for c in col_labels], fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Auditor / Coalition')
    ax.set_title(title, fontsize=11, fontweight='bold')
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            txt = f'{val:.3f}' if not np.isnan(val) else '—'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='Absolute Error')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")


for gt_key, (err_field_full, err_field_collab, gt_label) in GT_FIELDS.items():

    # ── Full local heatmap ─────────────────────────────────────────────────
    auditor_ids = sorted(set(r['auditor_id'] for r in full_results))
    mat = np.full((len(auditor_ids), len(target_ids)), np.nan)
    for r in full_results:
        i = auditor_ids.index(r['auditor_id'])
        j = target_ids.index(r['target_id'])
        mat[i, j] = r.get(err_field_full, np.nan)
    row_labels = [f'Node {a}' for a in auditor_ids]
    save_heatmap(
        mat, row_labels, target_ids,
        f'Full Local Audit — Absolute Error  ({gt_label})',
        os.path.join(PLOT_DIR, f'step5_heatmap_full_local_{PARTITION_ATTR}_{gt_key}.png')
    )

    # ── Global heatmap (global_all + global_excl as two rows) ─────────────
    mat = np.full((2, len(target_ids)), np.nan)
    for r in global_all_results:
        j = target_ids.index(r['target_id'])
        mat[0, j] = r.get(err_field_full, np.nan)
    for r in global_excl_results:
        j = target_ids.index(r['target_id'])
        mat[1, j] = r.get(err_field_full, np.nan)
    save_heatmap(
        mat, ['Global all', 'Global excl'], target_ids,
        f'Global Audit — Absolute Error  ({gt_label})',
        os.path.join(PLOT_DIR, f'step5_heatmap_global_{PARTITION_ATTR}_{gt_key}.png')
    )

    # ── Collab pairs heatmap ───────────────────────────────────────────────
    if collab_pair_results:
        coalition_labels = sorted(set(r['auditor_id'] for r in collab_pair_results))
        mat = np.full((len(coalition_labels), len(target_ids)), np.nan)
        for r in collab_pair_results:
            i = coalition_labels.index(r['auditor_id'])
            j = target_ids.index(r['target_id'])
            mat[i, j] = r.get(err_field_collab, np.nan)
        save_heatmap(
            mat, coalition_labels, target_ids,
            f'Pairwise Collaborative Audit — Absolute Error  ({gt_label})',
            os.path.join(PLOT_DIR, f'step5_heatmap_collab_pairs_{PARTITION_ATTR}_{gt_key}.png')
        )

    # ── Collab triples heatmap ─────────────────────────────────────────────
    if collab_triple_results:
        coalition_labels = sorted(set(r['auditor_id'] for r in collab_triple_results))
        mat = np.full((len(coalition_labels), len(target_ids)), np.nan)
        for r in collab_triple_results:
            i = coalition_labels.index(r['auditor_id'])
            j = target_ids.index(r['target_id'])
            mat[i, j] = r.get(err_field_collab, np.nan)
        save_heatmap(
            mat, coalition_labels, target_ids,
            f'Triplet Collaborative Audit — Absolute Error  ({gt_label})',
            os.path.join(PLOT_DIR, f'step5_heatmap_collab_triples_{PARTITION_ATTR}_{gt_key}.png')
        )
