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

# Ground truth error field mapping: (abs_field_full, abs_field_collab, rel_field_full, rel_field_collab, label)
GT_FIELDS = {
    'model_val' : ('abs_error',           'abs_error_model_val',  'rel_error',           'rel_error_model_val',  'GT: Model Val'),
    'data'      : ('abs_error_data',      'abs_error_data',       'rel_error_data',       'rel_error_data',       'GT: Data Labels'),
    'model_full': ('abs_error_model_full','abs_error_model_full', 'rel_error_model_full', 'rel_error_model_full', 'GT: Model Full'),
}


def save_heatmap(abs_mat, rel_mat, row_labels, col_labels, title, out_path):
    n_rows, n_cols = abs_mat.shape
    cell_w = max(1.1, 5 / n_cols)
    fig_w  = cell_w * n_cols * 2 + 2   # two panels side by side
    fig_h  = max(3, n_rows * 0.65)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    for ax, mat, panel_title, fmt, cbar_label in [
        (axes[0], abs_mat, 'Absolute Error', '{:.3f}',  'Abs Error'),
        (axes[1], rel_mat, 'Relative Error', '{:.0%}',  'Rel Error'),
    ]:
        im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f'Target {c}' for c in col_labels], fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Auditor / Coalition')
        ax.set_title(panel_title)
        for i in range(n_rows):
            for j in range(n_cols):
                val = mat[i, j]
                txt = fmt.format(val) if not np.isnan(val) else '—'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, label=cbar_label)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")


for gt_key, (abs_field_full, abs_field_collab,
             rel_field_full, rel_field_collab, gt_label) in GT_FIELDS.items():

    # ── Full local heatmap ─────────────────────────────────────────────────
    auditor_ids = sorted(set(r['auditor_id'] for r in full_results))
    abs_mat = np.full((len(auditor_ids), len(target_ids)), np.nan)
    rel_mat = np.full((len(auditor_ids), len(target_ids)), np.nan)
    for r in full_results:
        i = auditor_ids.index(r['auditor_id'])
        j = target_ids.index(r['target_id'])
        abs_mat[i, j] = r.get(abs_field_full, np.nan)
        rel_mat[i, j] = r.get(rel_field_full, np.nan)
    row_labels = [f'Node {a}' for a in auditor_ids]
    save_heatmap(
        abs_mat, rel_mat, row_labels, target_ids,
        f'Full Local Audit  ({gt_label})',
        os.path.join(PLOT_DIR, f'step5_heatmap_full_local_{PARTITION_ATTR}_{gt_key}.png')
    )

    # ── Global heatmap (global_all + global_excl as two rows) ─────────────
    abs_mat = np.full((2, len(target_ids)), np.nan)
    rel_mat = np.full((2, len(target_ids)), np.nan)
    for r in global_all_results:
        j = target_ids.index(r['target_id'])
        abs_mat[0, j] = r.get(abs_field_full, np.nan)
        rel_mat[0, j] = r.get(rel_field_full, np.nan)
    for r in global_excl_results:
        j = target_ids.index(r['target_id'])
        abs_mat[1, j] = r.get(abs_field_full, np.nan)
        rel_mat[1, j] = r.get(rel_field_full, np.nan)
    save_heatmap(
        abs_mat, rel_mat, ['Global all', 'Global excl'], target_ids,
        f'Global Audit  ({gt_label})',
        os.path.join(PLOT_DIR, f'step5_heatmap_global_{PARTITION_ATTR}_{gt_key}.png')
    )

    # ── Collab pairs heatmap ───────────────────────────────────────────────
    if collab_pair_results:
        coalition_labels = sorted(set(r['auditor_id'] for r in collab_pair_results))
        abs_mat = np.full((len(coalition_labels), len(target_ids)), np.nan)
        rel_mat = np.full((len(coalition_labels), len(target_ids)), np.nan)
        for r in collab_pair_results:
            i = coalition_labels.index(r['auditor_id'])
            j = target_ids.index(r['target_id'])
            abs_mat[i, j] = r.get(abs_field_collab, np.nan)
            rel_mat[i, j] = r.get(rel_field_collab, np.nan)
        save_heatmap(
            abs_mat, rel_mat, coalition_labels, target_ids,
            f'Pairwise Collaborative Audit  ({gt_label})',
            os.path.join(PLOT_DIR, f'step5_heatmap_collab_pairs_{PARTITION_ATTR}_{gt_key}.png')
        )

    # ── Collab triples heatmap ─────────────────────────────────────────────
    if collab_triple_results:
        coalition_labels = sorted(set(r['auditor_id'] for r in collab_triple_results))
        abs_mat = np.full((len(coalition_labels), len(target_ids)), np.nan)
        rel_mat = np.full((len(coalition_labels), len(target_ids)), np.nan)
        for r in collab_triple_results:
            i = coalition_labels.index(r['auditor_id'])
            j = target_ids.index(r['target_id'])
            abs_mat[i, j] = r.get(abs_field_collab, np.nan)
            rel_mat[i, j] = r.get(rel_field_collab, np.nan)
        save_heatmap(
            abs_mat, rel_mat, coalition_labels, target_ids,
            f'Triplet Collaborative Audit  ({gt_label})',
            os.path.join(PLOT_DIR, f'step5_heatmap_collab_triples_{PARTITION_ATTR}_{gt_key}.png')
        )

    # ── Combined: single + pairs + triples in one heatmap ─────────────────
    # Rows: single auditors, then pair coalitions, then triple coalitions
    # Divider rows (NaN) inserted between groups for visual separation
    auditor_ids      = sorted(set(r['auditor_id'] for r in full_results))
    pair_labels      = sorted(set(r['auditor_id'] for r in collab_pair_results))   if collab_pair_results   else []
    triple_labels    = sorted(set(r['auditor_id'] for r in collab_triple_results)) if collab_triple_results else []

    # Build row label list with blank separator rows between groups
    row_labels_combined = (
        [f'N{a} (single)' for a in auditor_ids]
        + (['─── pairs ───'] if pair_labels   else [])
        + pair_labels
        + (['─ triples ───'] if triple_labels else [])
        + triple_labels
    )
    n_rows = len(row_labels_combined)
    abs_combined = np.full((n_rows, len(target_ids)), np.nan)
    rel_combined = np.full((n_rows, len(target_ids)), np.nan)

    # Fill single rows
    for r in full_results:
        i = [f'N{a} (single)' for a in auditor_ids].index(f'N{r["auditor_id"]} (single)')
        j = target_ids.index(r['target_id'])
        abs_combined[i, j] = r.get(abs_field_full, np.nan)
        rel_combined[i, j] = r.get(rel_field_full, np.nan)

    # Fill pair rows (offset past singles + 1 separator)
    pair_offset = len(auditor_ids) + (1 if pair_labels else 0)
    for r in collab_pair_results:
        i = pair_offset + pair_labels.index(r['auditor_id'])
        j = target_ids.index(r['target_id'])
        abs_combined[i, j] = r.get(abs_field_collab, np.nan)
        rel_combined[i, j] = r.get(rel_field_collab, np.nan)

    # Fill triple rows (offset past singles + pairs + 2 separators)
    triple_offset = len(auditor_ids) + (1 if pair_labels else 0) + len(pair_labels) + (1 if triple_labels else 0)
    for r in collab_triple_results:
        i = triple_offset + triple_labels.index(r['auditor_id'])
        j = target_ids.index(r['target_id'])
        abs_combined[i, j] = r.get(abs_field_collab, np.nan)
        rel_combined[i, j] = r.get(rel_field_collab, np.nan)

    save_heatmap(
        abs_combined, rel_combined, row_labels_combined, target_ids,
        f'All Auditors Combined: Single | Pairs | Triples  ({gt_label})',
        os.path.join(PLOT_DIR, f'step5_heatmap_combined_{PARTITION_ATTR}_{gt_key}.png')
    )
