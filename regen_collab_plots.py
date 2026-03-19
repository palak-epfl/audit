"""
Collaboration-focused audit plots.
Ground truth throughout: dp_gap_data (raw label DP gap per node).

Reads from:
  step5_audit_results_{attr}.json   — full_local, collab_pairs
  step5c_audit_results_{attr}.json  — collab_triples, budgeted collab, global_excl_budget

Produces:
  step5_collab_heatmap_{attr}.png              (1) coalition × target error heatmap
  step5_collab_bars_{attr}.png                 (2) est DP gap bars per target, true_dp_data ref
  step5_collab_gain_{attr}.png                 (3) mean abs error by coalition size per target
  step5_collab_budget_{attr}.png               (4) mean abs error vs budget (all coalition sizes)
  step5_collab_scatter_{attr}.png              (5) estimated vs true DP scatter, colored by size

Usage:
    python3 regen_collab_plots.py --config config.yaml
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

NODE_COLORS  = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']
SIZE_COLORS  = {'single': 'steelblue', 'pairs': '#b0b0b0', 'triples': '#606060'}
SIZE_HATCHES = {'single': '', 'pairs': '//', 'triples': 'xx'}

# ── Load results ───────────────────────────────────────────────────────────
step5_path = os.path.join(RESULTS_DIR, f'step5_audit_results_{PARTITION_ATTR}.json')
with open(step5_path) as f:
    data5 = json.load(f)

full_results        = data5['full_results']
global_all_results  = data5['global_all_results']
global_excl_results = data5['global_excl_results']
collab_pair_results = data5.get('collab_pair_results', [])
budget_results      = data5.get('budget_results', [])

step5c_path = os.path.join(RESULTS_DIR, f'step5c_audit_results_{PARTITION_ATTR}.json')
collab_triple_results        = []
collab_pair_budget_results   = []
collab_triple_budget_results = []
global_excl_budget_results   = []
if os.path.exists(step5c_path):
    with open(step5c_path) as f:
        data5c = json.load(f)
    collab_triple_results        = data5c.get('collab_triple_results',        [])
    collab_pair_budget_results   = data5c.get('collab_pair_budget_results',   [])
    collab_triple_budget_results = data5c.get('collab_triple_budget_results', [])
    global_excl_budget_results   = data5c.get('global_excl_budget_results',   [])
    if not collab_pair_results:
        collab_pair_results = data5c.get('collab_pair_results', [])
    print(f"Loaded step5c — triples: {len(collab_triple_results)}, "
          f"pair_budget: {len(collab_pair_budget_results)}, "
          f"triple_budget: {len(collab_triple_budget_results)}")

target_ids = sorted(set(r['target_id'] for r in full_results))
n_targets  = len(target_ids)

print(f"Targets: {target_ids}")
print(f"Single: {len(full_results)}  Pairs: {len(collab_pair_results)}  "
      f"Triples: {len(collab_triple_results)}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Coalition × target error heatmap (abs + rel, data GT)
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlot 1: Heatmap...")

auditor_ids   = sorted(set(r['auditor_id'] for r in full_results))
pair_labels   = sorted(set(r['auditor_id'] for r in collab_pair_results))   if collab_pair_results   else []
triple_labels = sorted(set(r['auditor_id'] for r in collab_triple_results)) if collab_triple_results else []

row_labels = (
    [f'N{a} (single)' for a in auditor_ids]
    + (['─── pairs ───']   if pair_labels   else [])
    + pair_labels
    + (['─ triples ───']   if triple_labels else [])
    + triple_labels
)
n_rows   = len(row_labels)
n_cols   = len(target_ids)
abs_mat  = np.full((n_rows, n_cols), np.nan)
rel_mat  = np.full((n_rows, n_cols), np.nan)

single_row_labels = [f'N{a} (single)' for a in auditor_ids]
for r in full_results:
    i = single_row_labels.index(f'N{r["auditor_id"]} (single)')
    j = target_ids.index(r['target_id'])
    abs_mat[i, j] = r.get('abs_error_data', np.nan)
    rel_mat[i, j] = r.get('rel_error_data', np.nan)

pair_offset = len(auditor_ids) + (1 if pair_labels else 0)
for r in collab_pair_results:
    i = pair_offset + pair_labels.index(r['auditor_id'])
    j = target_ids.index(r['target_id'])
    abs_mat[i, j] = r.get('abs_error_data', np.nan)
    rel_mat[i, j] = r.get('rel_error_data', np.nan)

triple_offset = pair_offset + len(pair_labels) + (1 if triple_labels else 0)
for r in collab_triple_results:
    i = triple_offset + triple_labels.index(r['auditor_id'])
    j = target_ids.index(r['target_id'])
    abs_mat[i, j] = r.get('abs_error_data', np.nan)
    rel_mat[i, j] = r.get('rel_error_data', np.nan)

cell_w = max(1.1, 5 / n_cols)
fig_w  = cell_w * n_cols * 2 + 2
fig_h  = max(3, n_rows * 0.65)
fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
fig.suptitle(f'Collaborative Audit Error Heatmap — GT: Data Labels  (α={ALPHA})',
             fontsize=11, fontweight='bold')

for ax, mat, panel_title, fmt, cbar_label in [
    (axes[0], abs_mat, 'Absolute Error', '{:.3f}', 'Abs Error'),
    (axes[1], rel_mat, 'Relative Error', '{:.0%}',  'Rel Error'),
]:
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f'Target {t}' for t in target_ids], fontsize=9)
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
out = os.path.join(PLOT_DIR, f'step5_collab_heatmap_{PARTITION_ATTR}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Estimated DP gap bars per target — single / pairs / triples
#         Reference line = true_dp_gap_data
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 2: Bar chart per target...")

fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), sharey=True)
if n_targets == 1:
    axes = [axes]
fig.suptitle('Estimated DP Gap per Coalition — GT: Data Labels\n'
             '(dashed = true DP from data labels)',
             fontsize=12, fontweight='bold')

for ax, target_id in zip(axes, target_ids):
    singles = sorted([r for r in full_results        if r['target_id'] == target_id],
                     key=lambda r: r['auditor_id'])
    pairs   = sorted([r for r in collab_pair_results   if r['target_id'] == target_id],
                     key=lambda r: r['auditor_id'])
    triples = sorted([r for r in collab_triple_results if r['target_id'] == target_id],
                     key=lambda r: r['auditor_id'])

    true_dp_data = singles[0]['true_dp_gap_data'] if singles else 0.0

    groups = [
        (singles, [NODE_COLORS[r['auditor_id'] - 1] for r in singles], '',   'single'),
        (pairs,   [SIZE_COLORS['pairs']]   * len(pairs),               '//', 'pairs'),
        (triples, [SIZE_COLORS['triples']] * len(triples),             'xx', 'triples'),
    ]
    groups = [(res, cols, hatch, lbl) for res, cols, hatch, lbl in groups if res]

    x_pos = 0
    dividers = []
    group_spans = []
    for res, cols, hatch, lbl in groups:
        start = x_pos
        for r, col in zip(res, cols):
            val = r['est_dp_gap']
            ax.bar(x_pos, val, color=col, edgecolor='white', linewidth=0.8,
                   width=0.7, hatch=hatch, zorder=3)
            ax.text(x_pos, val + 0.003, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=6, rotation=90)
            x_pos += 1
        group_spans.append((start, x_pos - 1, lbl))

    all_vals = [r['est_dp_gap'] for res, _, _, _ in groups for r in res]
    y_top = max(all_vals + [true_dp_data]) * 1.35

    for i, (start, end, lbl) in enumerate(group_spans):
        if i > 0:
            ax.axvline(start - 0.5, color='gray', linestyle=':', linewidth=1)
        ax.text((start + end) / 2, y_top * 0.97, lbl,
                ha='center', va='top', fontsize=7, color='gray')

    ax.axhline(true_dp_data, color='black', linestyle='--', linewidth=1.5,
               label=f'True DP data ({true_dp_data:.3f})', zorder=2)

    all_labels = [f"N{r['auditor_id']}" for res, _, _, _ in groups for r in res]
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, fontsize=6, rotation=45, ha='right')
    ax.set_ylim(0, y_top)
    ax.set_title(f'Target: Node {target_id}', fontweight='bold')
    ax.legend(fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('Estimated DP Gap')
plt.tight_layout()
out = os.path.join(PLOT_DIR, f'step5_collab_bars_{PARTITION_ATTR}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Collaboration gain — mean abs error (data GT) by coalition size
#         One grouped bar cluster per target, bars = single / pairs / triples
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 3: Collaboration gain...")

size_groups = [
    ('single',  full_results,          'Single',  SIZE_COLORS['single'],  ''),
    ('pairs',   collab_pair_results,   'Pairs',   SIZE_COLORS['pairs'],   '//'),
    ('triples', collab_triple_results, 'Triples', SIZE_COLORS['triples'], 'xx'),
]
size_groups = [(k, res, lbl, col, hat) for k, res, lbl, col, hat in size_groups if res]

fig, ax = plt.subplots(figsize=(max(7, n_targets * 2), 5))
fig.suptitle('Mean Absolute Error by Coalition Size — GT: Data Labels\n'
             '(lower = better; error bars = ±1 std across coalitions)',
             fontsize=12, fontweight='bold')

n_sizes  = len(size_groups)
bar_w    = 0.7 / n_sizes
x        = np.arange(n_targets)

for idx, (key, res, lbl, col, hatch) in enumerate(size_groups):
    means, stds = [], []
    for tgt in target_ids:
        vals = [r['abs_error_data'] for r in res if r['target_id'] == tgt]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals)   if vals else 0.0)
    offset = (idx - (n_sizes - 1) / 2) * bar_w
    bars = ax.bar(x + offset, means, bar_w * 0.9,
                  label=lbl, color=col, hatch=hatch,
                  edgecolor='white', linewidth=0.6, zorder=3)
    ax.errorbar(x + offset, means, yerr=stds,
                fmt='none', color='black', capsize=3, linewidth=1, zorder=4)
    for bar, val in zip(bars, means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.1 + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([f'Target {t}' for t in target_ids])
ax.set_ylabel('Mean Absolute Error (data GT)')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
out = os.path.join(PLOT_DIR, f'step5_collab_gain_{PARTITION_ATTR}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Mean abs error vs budget — single / pairs / triples / global_excl
#         (data GT), one panel per target
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 4: Budget curves...")

budget_groups = [
    (budget_results,              'Single',       'steelblue',  '-',  'mean_abs_error_data'),
    (collab_pair_budget_results,  'Pair collab',  '#b0b0b0',    '--', 'mean_abs_error_data'),
    (collab_triple_budget_results,'Triple collab','#606060',    '-.', 'mean_abs_error_data'),
    (global_excl_budget_results,  'Global excl',  'darkorange', ':',  'mean_abs_error_data'),
]
budget_groups = [(res, lbl, col, ls, fld) for res, lbl, col, ls, fld in budget_groups if res]

if budget_groups:
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5), sharey=True)
    if n_targets == 1:
        axes = [axes]
    fig.suptitle('Mean Absolute Error vs Query Budget — GT: Data Labels',
                 fontsize=12, fontweight='bold')

    for ax, target_id in zip(axes, target_ids):
        budget_sizes = sorted(set(r['budget'] for r in budget_results
                                  if r['target_id'] == target_id))
        for res, lbl, col, ls, fld in budget_groups:
            means, stds = [], []
            for b in budget_sizes:
                vals = [r[fld] for r in res
                        if r['target_id'] == target_id and r['budget'] == b]
                means.append(np.mean(vals) if vals else np.nan)
                stds.append(np.std(vals)   if vals else np.nan)
            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)
            ax.plot(budget_sizes, means, color=col, linestyle=ls,
                    marker='o', linewidth=1.8, markersize=5, label=lbl)
            ax.fill_between(budget_sizes, means - stds, means + stds,
                            alpha=0.15, color=col)

        ax.set_xscale('log')
        ax.set_xlabel('Query budget')
        ax.set_title(f'Target: Node {target_id}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Mean Abs Error (data GT)')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5_collab_budget_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")
else:
    print("  Skipped (no budget results)")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Estimated vs true DP scatter — all coalition types, colored by size
# ─────────────────────────────────────────────────────────────────────────────
print("Plot 5: Estimated vs true DP scatter...")

scatter_groups = [
    (full_results,          'Single',  SIZE_COLORS['single'],  'o', 80),
    (collab_pair_results,   'Pairs',   SIZE_COLORS['pairs'],   's', 80),
    (collab_triple_results, 'Triples', SIZE_COLORS['triples'], '^', 80),
]
scatter_groups = [(res, lbl, col, mk, sz) for res, lbl, col, mk, sz in scatter_groups if res]

fig, ax = plt.subplots(figsize=(6, 6))
all_true, all_est = [], []
for res, lbl, col, mk, sz in scatter_groups:
    true_vals = [r['true_dp_gap_data'] for r in res]
    est_vals  = [r['est_dp_gap']       for r in res]
    ax.scatter(true_vals, est_vals, color=col, marker=mk, s=sz,
               alpha=0.7, label=lbl, zorder=3)
    all_true.extend(true_vals)
    all_est.extend(est_vals)

lim = [0, max(max(all_true), max(all_est)) * 1.2]
ax.plot(lim, lim, 'k--', alpha=0.4, linewidth=1, label='Perfect estimate')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('True DP Gap (data labels)')
ax.set_ylabel('Estimated DP Gap')
ax.set_title('Estimated vs True DP Gap — All Coalition Sizes\n'
             'GT: Data Labels',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
out = os.path.join(PLOT_DIR, f'step5_collab_scatter_{PARTITION_ATTR}.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → {out}")

print("\nDone.")
