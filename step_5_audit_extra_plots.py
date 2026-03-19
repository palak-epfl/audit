"""
Step 5 Extra Plots
===================
Reads from already-generated step5_audit_results.json and produces
additional visualisations. No re-running of audits needed.

Run with:
    python step5_extra_plots.py
    python3 step_5_audit_extra_plots.py --config config.yaml

Produces:
    plots/step5b_budget_labelled.png           (1) fixed budget labels
    plots/step5b_per_auditee_budget.png        (2) per-auditee budget curves
    plots/step5b_confidence_intervals.png      (3) CI plot per pair per budget
    plots/step5b_bias_variance.png             (4) bias vs variance decomposition
    plots/step5b_combined_all_modes.png        (5) global + local + budgeted together
    plots/step5b_auditor_consistency.png       (6) per-auditor consistency box plots
"""

import os
import sys
import json
import argparse
import datetime
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yaml

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Step 5 Extra Plots')
parser.add_argument('--config', type=str, default='config.yaml')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

NUM_NODES      = cfg['partition']['num_nodes']
ALPHA          = cfg['partition']['alpha']
PART_SEED      = cfg['partition']['seed']
PARTITION_ATTR = cfg['partition']['partition_attr']
RUN_DATE       = cfg['experiment'].get('run_date') or datetime.date.today().isoformat()
EXP_NAME       = (cfg['experiment'].get('name')
                  or f"lenet_alpha{ALPHA}_{NUM_NODES}nodes_seed{PART_SEED}")
NFS_ROOT       = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR        = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
PLOT_DIR       = os.path.join(EXP_DIR, 'plots',   RUN_DATE)
RESULTS_DIR    = os.path.join(EXP_DIR, 'results', RUN_DATE)
LOG_DIR        = os.path.join(EXP_DIR, 'logs',    RUN_DATE)

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, f'step5b_{PARTITION_ATTR}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode='w')
    ]
)
log = logging.getLogger()

# ── Style constants ────────────────────────────────────────────────────────────
NODE_COLORS   = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']
LINE_STYLES   = ['-', '--', '-.', ':', (0,(3,1,1,1))]   # one per target node
MARKER_STYLES = ['o', 's', '^', 'D', 'v']

# ── Load results ───────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("  Step 5b: Extra Plots")
log.info("=" * 70)

results_path = os.path.join(RESULTS_DIR, f'step5_audit_results_{PARTITION_ATTR}.json')
log.info(f"\n  Loading results from {results_path}...")
with open(results_path) as f:
    data = json.load(f)

full_results        = data['full_results']
budget_results      = data['budget_results']
global_all_results  = data['global_all_results']
global_excl_results = data['global_excl_results']
collab_pair_results = data.get('collab_pair_results', [])

# Also load triple (and any larger coalition) results from step5c JSON if present
step5c_path = os.path.join(RESULTS_DIR, f'step5c_audit_results_{PARTITION_ATTR}.json')
collab_triple_results = []
if os.path.exists(step5c_path):
    with open(step5c_path) as _f:
        _step5c = json.load(_f)
    collab_triple_results = _step5c.get('collab_triple_results', [])
    # step5c may have richer pair results too; only override if step5 JSON had none
    if not collab_pair_results:
        collab_pair_results = _step5c.get('collab_pair_results', [])
    log.info(f"  ✓ step5c JSON loaded — triples: {len(collab_triple_results)}")
budget_sizes        = sorted(set(r['budget'] for r in budget_results))
auditor_ids         = sorted(set(r['auditor_id'] for r in full_results))

# All three ground truth definitions
all_true_dp_gaps = {
    'data'       : {int(k): v for k, v in data['true_dp_gaps_data'].items()},
    'model_val'  : {int(k): v for k, v in data['true_dp_gaps_model_val'].items()},
    'model_full' : (
        {int(k): v for k, v in data['true_dp_gaps_model_full'].items()}
        if 'true_dp_gaps_model_full' in data else
        {r['target_id']: r['true_dp_gap_model_full'] for r in global_all_results}
    ),
}
# Primary (model_val) — used in helpers that need a single true_dp_gaps dict
true_dp_gaps = all_true_dp_gaps['model_val']
target_ids   = sorted(true_dp_gaps.keys())

log.info(f"  ✓ Loaded")
log.info(f"    Full results        : {len(full_results)}")
log.info(f"    Budget results      : {len(budget_results)}")
log.info(f"    Global all results  : {len(global_all_results)}")
log.info(f"    Global excl results : {len(global_excl_results)}")
log.info(f"    Budget sizes        : {budget_sizes}")
log.info(f"    Nodes               : {NUM_NODES}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get budget-aggregated result for a specific (auditor, target, budget)
# ─────────────────────────────────────────────────────────────────────────────
def get_budget_agg(auditor_id, target_id, budget):
    matches = [r for r in budget_results
               if r['auditor_id'] == auditor_id
               and r['target_id'] == target_id
               and r['budget'] == budget]
    return matches[0] if matches else None


def get_full(auditor_id, target_id):
    matches = [r for r in full_results
               if r['auditor_id'] == auditor_id
               and r['target_id'] == target_id]
    return matches[0] if matches else None


def get_global_all(target_id):
    matches = [r for r in global_all_results if r['target_id'] == target_id]
    return matches[0] if matches else None

def get_global_excl(target_id):
    matches = [r for r in global_excl_results if r['target_id'] == target_id]
    return matches[0] if matches else None


def bootstrap_ci(values, n_boot=1000, ci=95, seed=42):
    """Return (mean, lo, hi) bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    values = np.array(values)
    boots = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(np.mean(values)), float(lo), float(hi)


# ─────────────────────────────────────────────────────────────────────────────
# Generate all plots for each ground truth definition
# ─────────────────────────────────────────────────────────────────────────────
for gt_name, gt_label in [
    ('data',       'GT: Data Labels'),
    ('model_val',  'GT: Model Val'),
    ('model_full', 'GT: Model Full'),
]:
    log.info(f"\n{'='*70}")
    log.info(f"  Ground Truth: {gt_label}")
    log.info(f"{'='*70}\n")

    # Re-alias error fields in all results to point to current GT
    # so all plot code can use r['abs_error'] / r['mean_abs_error'] unchanged
    # For model_val the primary fields (abs_error, rel_error, mean_abs_error)
    # are the model_val values; use them as fallback for older JSONs that lack
    # the explicit _model_val suffixed keys.
    true_dp_gaps = all_true_dp_gaps[gt_name]
    for r in full_results:
        r['abs_error']   = r.get(f'abs_error_{gt_name}',   r['abs_error'])
        r['rel_error']   = r.get(f'rel_error_{gt_name}',   r['rel_error'])
        r['true_dp_gap'] = r[f'true_dp_gap_{gt_name}']
    for r in budget_results:
        r['mean_abs_error'] = r.get(f'mean_abs_error_{gt_name}', r['mean_abs_error'])
        r['true_dp_gap']    = r[f'true_dp_gap_{gt_name}']
        for rep in r['repeats']:
            rep['abs_error'] = rep.get(f'abs_error_{gt_name}', rep['abs_error'])
            rep['rel_error'] = rep.get(f'rel_error_{gt_name}', rep['rel_error'])
    for r in global_all_results + global_excl_results:
        r['abs_error']   = r.get(f'abs_error_{gt_name}',   r['abs_error'])
        r['rel_error']   = r.get(f'rel_error_{gt_name}',   r['rel_error'])
        r['true_dp_gap'] = r[f'true_dp_gap_{gt_name}']

    for shading_mode in ['std', 'ci']:
        shading_label = '±1 std' if shading_mode == 'std' else '95% CI'

        # ─────────────────────────────────────────────────────────────────────────────
        # Plot 1: Budget curves with proper N1→N3 labels
        # ─────────────────────────────────────────────────────────────────────────────
        log.info(f"  [{gt_label}] Generating Plot 1: Budget curves with N_aud→N_tgt labels ({shading_mode})...")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Budgeted Audit — Sample Efficiency\n'
                     'Label format: Nauditor→Ntarget',
                     fontsize=13, fontweight='bold')

        # Left panel: abs error vs budget, one line per pair
        ax = axes[0]
        for aud_id in auditor_ids:
            for tgt_id in target_ids:
                if aud_id == tgt_id:
                    continue
                pair_data = sorted(
                    [r for r in budget_results
                     if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
                    key=lambda r: r['budget']
                )
                if not pair_data:
                    continue
                budgets   = [r['budget']         for r in pair_data]
                mean_errs = [r['mean_abs_error'] for r in pair_data]
                # std of abs errors across repeats — matches the y-axis (mean abs error)
                std_errs  = [
                    float(np.std([rep['abs_error'] for rep in r['repeats']]))
                    for r in pair_data
                ]

                color     = NODE_COLORS[aud_id - 1]
                linestyle = LINE_STYLES[(tgt_id - 1) % len(LINE_STYLES)]
                label     = f'N{aud_id}→N{tgt_id}'

                ax.plot(budgets, mean_errs, color=color, linestyle=linestyle,
                        linewidth=1.5, marker=MARKER_STYLES[tgt_id-1],
                        markersize=4, label=label)

        ax.set_xlabel('Query Budget')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('All Pairs (colour=auditor, style=target)')
        ax.set_xscale('log')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                  fontsize=7, ncol=1)
        ax.spines[['top','right']].set_visible(False)

        # Right panel: averaged across all pairs with conditional shading
        ax = axes[1]
        mean_per_budget = []
        band_vals = []
        for budget in budget_sizes:
            vals = [rep['abs_error']
                    for r in budget_results if r['budget'] == budget
                    for rep in r['repeats']]
            mean_per_budget.append(np.mean([r['mean_abs_error'] for r in budget_results
                                            if r['budget'] == budget]))
            band_vals.append(vals)

        m = np.array(mean_per_budget)
        if shading_mode == 'std':
            lo = np.array([m[i] - np.std(v) for i, v in enumerate(band_vals)])
            hi = np.array([m[i] + np.std(v) for i, v in enumerate(band_vals)])
        else:
            lo = np.array([bootstrap_ci(v)[1] for v in band_vals])
            hi = np.array([bootstrap_ci(v)[2] for v in band_vals])
        ax.plot(budget_sizes, m, color='steelblue', linewidth=2,
                marker='o', label='Mean ± std across all pairs')
        ax.fill_between(budget_sizes, lo, hi, alpha=0.2, color='steelblue')

        # Mark each budget size
        for b, mv in zip(budget_sizes, m):
            ax.annotate(f'{b}', xy=(b, mv), xytext=(0, 8),
                        textcoords='offset points', ha='center',
                        fontsize=8, color='steelblue')

        ax.set_xlabel('Query Budget')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'Averaged Across All Pairs\n(shading = {shading_label})')
        ax.set_xscale('log')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)

        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f'step5b_budget_labelled_{PARTITION_ATTR}_{gt_name}_{shading_mode}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved → {out}")


        # ─────────────────────────────────────────────────────────────────────────────
        # Plot 2: Per-auditee budget curves
        # One subplot per target node. Each line = one auditor.
        # ─────────────────────────────────────────────────────────────────────────────
        log.info(f"  [{gt_label}] Generating Plot 2: Per-auditee budget curves ({shading_mode})...")

        ncols = min(3, len(target_ids))
        nrows = (len(target_ids) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4 * nrows),
                                  squeeze=False)
        fig.suptitle('Per-Auditee Budget Curves\n'
                     'How does each auditor\'s accuracy improve with more queries?',
                     fontsize=13, fontweight='bold')

        for idx, tgt_id in enumerate(target_ids):
            ax       = axes[idx // ncols][idx % ncols]
            true_dp  = true_dp_gaps[tgt_id]

            for aud_id in auditor_ids:
                if aud_id == tgt_id:
                    continue
                pair_data = sorted(
                    [r for r in budget_results
                     if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
                    key=lambda r: r['budget']
                )
                if not pair_data:
                    continue
                budgets   = [r['budget']         for r in pair_data]
                mean_errs = [r['mean_abs_error'] for r in pair_data]
                # std of absolute errors across repeats — correct uncertainty for
                # a mean abs error plot. std_est_dp would be wrong here because
                # it measures spread of raw DP estimates, not spread of abs errors.

                color = NODE_COLORS[aud_id - 1]
                ax.plot(budgets, mean_errs, color=color, linewidth=1.8,
                        marker='o', markersize=4, label=f'Auditor N{aud_id}')
                if shading_mode == 'std':
                    lo = np.maximum(
                        np.array(mean_errs) - np.array([np.std([rep['abs_error'] for rep in r['repeats']]) for r in pair_data]),
                        0)
                    hi = np.array(mean_errs) + np.array([np.std([rep['abs_error'] for rep in r['repeats']]) for r in pair_data])
                else:
                    lo = np.array([bootstrap_ci([rep['abs_error'] for rep in r['repeats']])[1] for r in pair_data])
                    hi = np.array([bootstrap_ci([rep['abs_error'] for rep in r['repeats']])[2] for r in pair_data])
                ax.fill_between(budgets, lo, hi, alpha=0.1, color=color)

            # Full local reference lines — one dashed line per auditor
            # coloured to match its budgeted curve, labelled at right edge
            for aud_id in auditor_ids:
                if aud_id == tgt_id:
                    continue
                full = get_full(aud_id, tgt_id)
                if full:
                    ax.axhline(full['abs_error'], color=NODE_COLORS[aud_id-1],
                               linestyle='--', alpha=0.5, linewidth=1)
                    # Label at right edge so it is clear which auditor it belongs to
                    ax.text(budget_sizes[-1] * 1.05, full['abs_error'],
                            f'N{aud_id} full', fontsize=6,
                            color=NODE_COLORS[aud_id-1], va='center', alpha=0.7)

            # Global auditor reference lines — all data (red) and excl own data (orange)
            glob_all  = get_global_all(tgt_id)
            glob_excl = get_global_excl(tgt_id)
            if glob_all:
                ax.axhline(glob_all['abs_error'], color='red',
                           linestyle='-.', linewidth=1.5,
                           label=f'Global all (err={glob_all["abs_error"]:.3f})')
                ax.text(budget_sizes[-1] * 1.05, glob_all['abs_error'],
                        'g_all', fontsize=6, color='red', va='center')
            if glob_excl:
                ax.axhline(glob_excl['abs_error'], color='orange',
                           linestyle=':', linewidth=1.5,
                           label=f'Global excl (err={glob_excl["abs_error"]:.3f})')
                ax.text(budget_sizes[-1] * 1.05, glob_excl['abs_error'],
                        'g_excl', fontsize=6, color='orange', va='center')

            ax.set_title(f'Target: Node {tgt_id}  (true DP={true_dp:.3f})',
                         fontweight='bold')
            ax.set_xlabel('Query Budget')
            ax.set_ylabel('Mean Absolute Error')
            ax.set_xscale('log')
            ax.legend(fontsize=7)
            ax.spines[['top','right']].set_visible(False)

            # Legend note
            ax.text(0.98, 0.98,
                    'Dashed = full local per auditor\n'
                    'Dash-dot (red) = global all\n'
                    'Dotted (orange) = global excl own',
                    transform=ax.transAxes, fontsize=6,
                    ha='right', va='top', color='gray')

        # Hide unused subplots
        for idx in range(len(target_ids), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f'step5b_per_auditee_budget_{PARTITION_ATTR}_{gt_name}_{shading_mode}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved → {out}")


    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 2b: Per-auditee budget BAR chart — mean abs error per auditor at each
    #          budget level, values written on top of bars for easy reading.
    # ─────────────────────────────────────────────────────────────────────────────
    log.info(f"  [{gt_label}] Generating Plot 2b: Per-auditee budget bar chart...")

    ncols_b = min(3, len(target_ids))
    nrows_b = (len(target_ids) + ncols_b - 1) // ncols_b
    fig, axes = plt.subplots(nrows_b, ncols_b,
                              figsize=(6 * ncols_b, 4 * nrows_b),
                              squeeze=False)
    fig.suptitle('Per-Auditee Budget — Mean Abs Error per Auditor at Each Budget\n'
                 '(values labelled on bars)',
                 fontsize=13, fontweight='bold')

    for idx, tgt_id in enumerate(target_ids):
        ax = axes[idx // ncols_b][idx % ncols_b]
        aud_ids_here = [a for a in auditor_ids if a != tgt_id]
        n_auds    = len(aud_ids_here)
        n_bud     = len(budget_sizes)
        bar_w     = 0.8 / n_auds
        x_centers = np.arange(n_bud)

        for a_idx, aud_id in enumerate(aud_ids_here):
            offset = (a_idx - n_auds / 2 + 0.5) * bar_w
            color  = NODE_COLORS[aud_id - 1]
            for b_idx, budget in enumerate(budget_sizes):
                rows = [r for r in budget_results
                        if r['auditor_id'] == aud_id and r['target_id'] == tgt_id
                        and r['budget'] == budget]
                if not rows:
                    continue
                val = rows[0]['mean_abs_error']
                x   = x_centers[b_idx] + offset
                ax.bar(x, val, bar_w * 0.9, color=color, edgecolor='white',
                       linewidth=0.5,
                       label=f'N{aud_id}' if b_idx == 0 else '')
                ax.text(x, val + 0.002, f'{val:.3f}', ha='center', va='bottom',
                        fontsize=5, color=color, rotation=90)

        glob_all  = get_global_all(tgt_id)
        glob_excl = get_global_excl(tgt_id)
        if glob_all:
            ax.axhline(glob_all['abs_error'], color='red', linestyle='-.', linewidth=1.2,
                       label=f'Global all ({glob_all["abs_error"]:.3f})')
        if glob_excl:
            ax.axhline(glob_excl['abs_error'], color='orange', linestyle=':', linewidth=1.2,
                       label=f'Global excl ({glob_excl["abs_error"]:.3f})')

        ax.set_xticks(x_centers)
        ax.set_xticklabels([str(b) for b in budget_sizes], fontsize=8)
        ax.set_xlabel('Query Budget')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title(f'Target: Node {tgt_id}', fontweight='bold')
        ax.legend(fontsize=7)
        ax.spines[['top','right']].set_visible(False)

    for idx in range(len(target_ids), nrows_b * ncols_b):
        axes[idx // ncols_b][idx % ncols_b].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5b_per_auditee_budget_bars_{PARTITION_ATTR}_{gt_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved → {out}")


    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 3: Confidence intervals per (auditor, target) pair per budget
    # ─────────────────────────────────────────────────────────────────────────────
    log.info(f"  [{gt_label}] Generating Plot 3: Confidence interval plot...")

    n_budgets = len(budget_sizes)
    ncols     = min(3, n_budgets)
    nrows     = (n_budgets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle('Bootstrap 95% Confidence Intervals on Estimated DP Gap\n'
                 'per Auditor→Target Pair at Each Budget',
                 fontsize=13, fontweight='bold')

    for idx, budget in enumerate(budget_sizes):
        ax      = axes[idx // ncols][idx % ncols]
        pairs   = [(aud, tgt)
                   for aud in auditor_ids for tgt in target_ids
                   if aud != tgt]
        x_ticks = []
        x_labels= []

        for x_pos, (aud_id, tgt_id) in enumerate(pairs):
            r = get_budget_agg(aud_id, tgt_id, budget)
            if r is None:
                continue

            mean_dp = r['mean_est_dp']
            ci_lo   = r['ci_lower']
            ci_hi   = r['ci_upper']
            true_dp = true_dp_gaps[tgt_id]
            color   = NODE_COLORS[aud_id - 1]

            # CI bar
            ax.plot([x_pos, x_pos], [ci_lo, ci_hi],
                    color=color, linewidth=2, solid_capstyle='round')
            # Mean point
            ax.scatter(x_pos, mean_dp, color=color, s=40, zorder=4)
            # True DP tick
            ax.scatter(x_pos, true_dp, color='black',
                       marker='_', s=80, zorder=5, linewidths=2)

            x_ticks.append(x_pos)
            x_labels.append(f'N{aud_id}→N{tgt_id}')

        ax.set_title(f'Budget = {budget}', fontweight='bold')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Estimated DP Gap')
        ax.spines[['top','right']].set_visible(False)

        # Legend
        ci_line   = mlines.Line2D([], [], color='gray', linewidth=2,
                                   label='95% CI')
        true_mark = mlines.Line2D([], [], color='black', marker='_',
                                   markersize=8, linewidth=0,
                                   label='True DP gap')
        ax.legend(handles=[ci_line, true_mark], fontsize=7)

    # Hide unused
    for idx in range(n_budgets, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5b_confidence_intervals_{PARTITION_ATTR}_{gt_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved → {out}")

    for shading_mode in ['std', 'ci']:
        shading_label = '±1 std' if shading_mode == 'std' else '95% CI'

        # ─────────────────────────────────────────────────────────────────────────────
        # Plot 4: Bias vs variance decomposition
        # bias  = mean(estimated DP) - true DP     (systematic over/underestimation)
        # variance = std(estimated DP)²            (sensitivity to query sample)
        # ─────────────────────────────────────────────────────────────────────────────
        log.info(f"  [{gt_label}] Generating Plot 4: Bias vs variance decomposition ({shading_mode})...")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Bias vs Variance Decomposition of Audit Error\n'
                     'bias = mean(est DP) − true DP  |  '
                     'variance = std(est DP)²  |  '
                     'MSE = bias² + variance',
                     fontsize=12, fontweight='bold')

        # Collect bias and variance per pair per budget
        bv_data = {}   # (aud, tgt, budget) → {bias, variance, mse}
        for r in budget_results:
            aud_id  = r['auditor_id']
            tgt_id  = r['target_id']
            budget  = r['budget']
            true_dp = true_dp_gaps[tgt_id]

            # Extract per-repeat estimates
            repeat_ests = [rep['est_dp_gap'] for rep in r['repeats']]
            mean_est    = np.mean(repeat_ests)
            std_est     = np.std(repeat_ests)

            bias     = float(mean_est - true_dp)   # signed: positive = overestimate
            variance = float(std_est ** 2)
            mse      = float(bias ** 2 + variance)

            bv_data[(aud_id, tgt_id, budget)] = {
                'bias': bias, 'variance': variance, 'mse': mse,
                'abs_bias': abs(bias)
            }

        # Panel 1: |bias| vs budget, averaged across pairs
        ax = axes[0]
        mean_bias_per_budget = []
        std_bias_per_budget  = []
        for budget in budget_sizes:
            abs_biases = [bv_data[(a, t, budget)]['abs_bias']
                          for a in auditor_ids for t in target_ids
                          if a != t and (a, t, budget) in bv_data]
            mean_bias_per_budget.append(np.mean(abs_biases))
            std_bias_per_budget.append(np.std(abs_biases))

        m = np.array(mean_bias_per_budget)
        ax.plot(budget_sizes, m, color='tomato', linewidth=2,
                marker='o', label='Mean |bias|')
        if shading_mode == 'std':
            s = np.array(std_bias_per_budget)
            ax.fill_between(budget_sizes, m - s, m + s, alpha=0.2, color='tomato')
        else:
            bias_lo = np.array([bootstrap_ci([bv_data[(a,t,b)]['abs_bias']
                                 for a in auditor_ids for t in target_ids
                                 if a != t and (a,t,b) in bv_data])[1] for b in budget_sizes])
            bias_hi = np.array([bootstrap_ci([bv_data[(a,t,b)]['abs_bias']
                                 for a in auditor_ids for t in target_ids
                                 if a != t and (a,t,b) in bv_data])[2] for b in budget_sizes])
            ax.fill_between(budget_sizes, bias_lo, bias_hi, alpha=0.2, color='tomato')
        ax.set_xlabel('Query Budget'); ax.set_ylabel('|Bias|')
        ax.set_title(f'|Bias| vs Budget\n(shading = {shading_label})')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        # Panel 2: variance vs budget, averaged across pairs
        ax = axes[1]
        mean_var_per_budget = []
        std_var_per_budget  = []
        for budget in budget_sizes:
            variances = [bv_data[(a, t, budget)]['variance']
                         for a in auditor_ids for t in target_ids
                         if a != t and (a, t, budget) in bv_data]
            mean_var_per_budget.append(np.mean(variances))
            std_var_per_budget.append(np.std(variances))

        m = np.array(mean_var_per_budget)
        ax.plot(budget_sizes, m, color='steelblue', linewidth=2,
                marker='o', label='Mean variance')
        if shading_mode == 'std':
            s = np.array(std_var_per_budget)
            ax.fill_between(budget_sizes, np.maximum(m - s, 0), m + s,
                            alpha=0.2, color='steelblue')
        else:
            var_lo = np.maximum(np.array([bootstrap_ci([bv_data[(a,t,b)]['variance']
                                  for a in auditor_ids for t in target_ids
                                  if a != t and (a,t,b) in bv_data])[1] for b in budget_sizes]), 0)
            var_hi = np.array([bootstrap_ci([bv_data[(a,t,b)]['variance']
                                for a in auditor_ids for t in target_ids
                                if a != t and (a,t,b) in bv_data])[2] for b in budget_sizes])
            ax.fill_between(budget_sizes, var_lo, var_hi, alpha=0.2, color='steelblue')
        ax.set_xlabel('Query Budget'); ax.set_ylabel('Variance')
        ax.set_title(f'Variance vs Budget\n(shading = {shading_label})')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        # Panel 3: stacked bias² + variance = MSE at each budget
        ax = axes[2]
        mean_bias2 = [np.mean([bv_data[(a,t,b)]['bias']**2
                                for a in auditor_ids for t in target_ids
                                if a != t and (a,t,b) in bv_data])
                      for b in budget_sizes]
        mean_var   = [np.mean([bv_data[(a,t,b)]['variance']
                                for a in auditor_ids for t in target_ids
                                if a != t and (a,t,b) in bv_data])
                      for b in budget_sizes]

        x = np.arange(len(budget_sizes))
        w = 0.5
        ax.bar(x, mean_bias2, w, label='Bias²',     color='tomato',    alpha=0.85)
        ax.bar(x, mean_var,   w, bottom=mean_bias2, label='Variance',
               color='steelblue', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in budget_sizes])
        ax.set_xlabel('Query Budget'); ax.set_ylabel('MSE')
        ax.set_title('MSE = Bias² + Variance\n(what drives the error at each budget?)')
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)

        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f'step5b_bias_variance_{PARTITION_ATTR}_{gt_name}_{shading_mode}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved → {out}")


        # ─────────────────────────────────────────────────────────────────────────────
        # Plot 5: Global + full local + budgeted on one axis per target node
        # ─────────────────────────────────────────────────────────────────────────────
        log.info(f"  [{gt_label}] Generating Plot 5: Combined all modes per target ({shading_mode})...")

        ncols = min(3, len(target_ids))
        nrows = (len(target_ids) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(6 * ncols, 5 * nrows),
                                  squeeze=False)
        fig.suptitle('All Audit Modes Combined per Target Node\n'
                     'Global  |  Full Local  |  Budgeted (with CI)',
                     fontsize=13, fontweight='bold')

        for idx, tgt_id in enumerate(target_ids):
            ax      = axes[idx // ncols][idx % ncols]
            true_dp = true_dp_gaps[tgt_id]

            # True DP reference line
            ax.axhline(true_dp, color='black', linestyle='--',
                       linewidth=1.5, label=f'True DP ({true_dp:.3f})', zorder=1)

            # Global estimates — all data (red) and excl own data (orange)
            glob_all  = get_global_all(tgt_id)
            glob_excl = get_global_excl(tgt_id)
            if glob_all:
                ax.axhline(glob_all['est_dp_gap'], color='red', linestyle='-.',
                           linewidth=1.5,
                           label=f'Global all ({glob_all["est_dp_gap"]:.3f})', zorder=2)
            if glob_excl:
                ax.axhline(glob_excl['est_dp_gap'], color='orange', linestyle=':',
                           linewidth=1.5,
                           label=f'Global excl ({glob_excl["est_dp_gap"]:.3f})', zorder=2)

            # For each auditor: full local + budgeted curve
            for aud_id in auditor_ids:
                if aud_id == tgt_id:
                    continue
                color = NODE_COLORS[aud_id - 1]

                # Full local — single point at x = max_budget * 1.3 (right side)
                full = get_full(aud_id, tgt_id)

                # Budgeted curve
                pair_budget = sorted(
                    [r for r in budget_results
                     if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
                    key=lambda r: r['budget']
                )
                if pair_budget:
                    budgets   = [r['budget']     for r in pair_budget]
                    mean_ests = [r['mean_est_dp'] for r in pair_budget]

                    ax.plot(budgets, mean_ests, color=color, linewidth=1.8,
                            marker='o', markersize=4,
                            label=f'N{aud_id} budgeted', zorder=3)
                    if shading_mode == 'std':
                        std_ests = [float(np.std([rep['est_dp_gap'] for rep in r['repeats']])) for r in pair_budget]
                        mean_arr = np.array(mean_ests)
                        std_arr  = np.array(std_ests)
                        band_lo  = np.maximum(mean_arr - std_arr, 0)
                        band_hi  = mean_arr + std_arr
                    else:
                        band_lo = np.array([r['ci_lower'] for r in pair_budget])
                        band_hi = np.array([r['ci_upper'] for r in pair_budget])
                    ax.fill_between(budgets, band_lo, band_hi, alpha=0.15, color=color)

                # Full local — plotted as a star at x just beyond max budget
                if full:
                    x_full = budget_sizes[-1] * 1.5
                    ax.scatter(x_full, full['est_dp_gap'],
                               color=color, marker='*', s=180, zorder=5)

            # Add a text label for the star markers
            ax.text(0.97, 0.04, '★ = full local',
                    transform=ax.transAxes, fontsize=7,
                    ha='right', va='bottom', color='gray')

            ax.set_title(f'Target: Node {tgt_id}', fontweight='bold')
            ax.set_xlabel('Query Budget (★ = full local data)')
            ax.set_ylabel('Estimated DP Gap')
            ax.set_xscale('log')
            ax.legend(fontsize=7, loc='upper right')
            ax.spines[['top','right']].set_visible(False)

        # Hide unused subplots
        for idx in range(len(target_ids), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f'step5b_combined_all_modes_{PARTITION_ATTR}_{gt_name}_{shading_mode}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved → {out}")

    # end for shading_mode


    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 5b: Combined all modes BAR chart — estimated DP gap for full-local,
    #          global-all and global-excl per target, values on top of bars.
    # ─────────────────────────────────────────────────────────────────────────────
    log.info(f"  [{gt_label}] Generating Plot 5b: Combined all modes bar chart...")

    ncols_b = min(3, len(target_ids))
    nrows_b = (len(target_ids) + ncols_b - 1) // ncols_b
    fig, axes = plt.subplots(nrows_b, ncols_b,
                              figsize=(7 * ncols_b, 5 * nrows_b),
                              squeeze=False)
    fig.suptitle('All Audit Modes — Estimated DP Gap (Bar Chart)\n'
                 'Full Local | Global All | Global Excl — values on bars',
                 fontsize=13, fontweight='bold')

    for idx, tgt_id in enumerate(target_ids):
        ax      = axes[idx // ncols_b][idx % ncols_b]
        true_dp = true_dp_gaps[tgt_id]

        bar_labels = []
        bar_vals   = []
        bar_cols   = []

        for aud_id in auditor_ids:
            if aud_id == tgt_id:
                continue
            full = get_full(aud_id, tgt_id)
            if full:
                bar_labels.append(f'N{aud_id}\nfull')
                bar_vals.append(full['est_dp_gap'])
                bar_cols.append(NODE_COLORS[aud_id - 1])

        glob_all  = get_global_all(tgt_id)
        glob_excl = get_global_excl(tgt_id)
        if glob_all:
            bar_labels.append('Global\nall')
            bar_vals.append(glob_all['est_dp_gap'])
            bar_cols.append('red')
        if glob_excl:
            bar_labels.append('Global\nexcl')
            bar_vals.append(glob_excl['est_dp_gap'])
            bar_cols.append('orange')

        x_pos = np.arange(len(bar_labels))
        rects = ax.bar(x_pos, bar_vals, color=bar_cols, edgecolor='white',
                       linewidth=0.8, width=0.7, zorder=3)

        for rect, val in zip(rects, bar_vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.003, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.axhline(true_dp, color='black', linestyle='--', linewidth=1.5,
                   label=f'True DP ({true_dp:.3f})', zorder=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel('Estimated DP Gap')
        ax.set_ylim(0, max(bar_vals + [true_dp]) * 1.25)
        ax.set_title(f'Target: Node {tgt_id}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)

    for idx in range(len(target_ids), nrows_b * ncols_b):
        axes[idx // ncols_b][idx % ncols_b].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5b_combined_all_modes_bars_{PARTITION_ATTR}_{gt_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved → {out}")


    # ─────────────────────────────────────────────────────────────────────────────
    # Plot 6: Per-auditor consistency box plots
    # For each auditor: distribution of absolute errors across all targets
    # and all budget sizes. Shows which auditor is most/least consistent.
    # ─────────────────────────────────────────────────────────────────────────────
    log.info(f"  [{gt_label}] Generating Plot 6: Per-auditor consistency box plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Per-Auditor Consistency\n'
                 'How much does error vary across different targets?',
                 fontsize=13, fontweight='bold')

    # Panel 1: full local — box per auditor, one point per target
    ax = axes[0]
    full_errors_per_auditor = []
    labels = []
    for aud_id in auditor_ids:
        errs = [r['abs_error'] for r in full_results
                if r['auditor_id'] == aud_id]
        full_errors_per_auditor.append(errs)
        labels.append(f'Node {aud_id}\n(n={len(errs)})')

    bp = ax.boxplot(full_errors_per_auditor, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], NODE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for x_pos, (errs, aud_id) in enumerate(
            zip(full_errors_per_auditor, auditor_ids), start=1):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(errs))
        ax.scatter(x_pos + jitter, errs,
                   color=NODE_COLORS[aud_id-1], alpha=0.6, s=40, zorder=3)

    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Absolute Error')
    ax.set_title('Full Local Audit\nError distribution across targets per auditor')
    ax.spines[['top','right']].set_visible(False)

    # Panel 2: budgeted — box per auditor at each budget,
    # grouped by auditor with budget as hue via offset x positions
    ax = axes[1]
    n_auditors = len(auditor_ids)
    n_budgets  = len(budget_sizes)
    group_width = 0.8
    bar_width   = group_width / n_budgets
    budget_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_budgets))

    for b_idx, budget in enumerate(budget_sizes):
        for a_idx, aud_id in enumerate(auditor_ids):
            errs = [r['mean_abs_error'] for r in budget_results
                    if r['auditor_id'] == aud_id and r['budget'] == budget]
            if not errs:
                continue
            x = a_idx + (b_idx - n_budgets/2 + 0.5) * bar_width
            ax.bar(x, np.mean(errs), bar_width * 0.9,
                   color=budget_colors[b_idx], edgecolor='white',
                   linewidth=0.5,
                   label=f'Budget {budget}' if a_idx == 0 else '')

    ax.set_xticks(range(n_auditors))
    ax.set_xticklabels([f'Node {i}' for i in auditor_ids], fontsize=9)
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Budgeted Audit\nMean error per auditor at each budget\n'
                 '(darker = larger budget)')
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5b_auditor_consistency_{PARTITION_ATTR}_{gt_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved → {out}")




# ─────────────────────────────────────────────────────────────────────────────
# PCA of node distribution space (runs once, not per GT variant)
# Features: pct_male, dp_gap_data, dp_gap_model, accuracy,
#           p_smile_true_male, p_smile_true_female  (all from step3/step4 JSONs)
# ─────────────────────────────────────────────────────────────────────────────
log.info("  Generating PCA plot of node distribution space...")

step4_path = os.path.join(RESULTS_DIR, f'step4_all_nodes_{PARTITION_ATTR}_results.json')
step3_path = os.path.join(RESULTS_DIR, f'step3_{ALPHA}_{PARTITION_ATTR}_partition_stats.json')

try:
    with open(step4_path) as f:
        step4_data = json.load(f)
    with open(step3_path) as f:
        step3_data = json.load(f)

    step4_nodes = {r['node_id']: r for r in step4_data['nodes']}
    step3_nodes = {r['node_id']: r for r in step3_data['nodes']}
    node_ids_pca = sorted(step4_nodes.keys())

    feature_names = [
        'pct_male',
        'dp_gap_data',
        'dp_gap_model',
        'accuracy',
        'p_smile_true_male',
        'p_smile_true_female',
    ]

    X = np.array([
        [
            step3_nodes[nid]['pct_male'],
            step4_nodes[nid]['dp_gap_data'],
            step4_nodes[nid]['dp_gap_model'],
            step4_nodes[nid]['accuracy'],
            step4_nodes[nid]['p_smile_true_male'],
            step4_nodes[nid]['p_smile_true_female'],
        ]
        for nid in node_ids_pca
    ])

    X_scaled = StandardScaler().fit_transform(X)
    pca      = PCA(n_components=min(2, X_scaled.shape[1]))
    coords   = pca.fit_transform(X_scaled)
    var      = pca.explained_variance_ratio_

    loadings = pca.components_   # shape (n_components, n_features)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Node Distribution Space — PCA\n'
                 f'Features: {", ".join(feature_names)}',
                 fontsize=12, fontweight='bold')

    # Panel 1: scatter of nodes in PC space
    ax = axes[0]
    for i, nid in enumerate(node_ids_pca):
        x, y = coords[i, 0], coords[i, 1] if coords.shape[1] > 1 else 0.0
        ax.scatter(x, y, color=NODE_COLORS[nid - 1], s=200, zorder=3)
        ax.annotate(f'Node {nid}', xy=(x, y),
                    xytext=(6, 4), textcoords='offset points', fontsize=9)

    ax.set_xlabel(f'PC1  ({var[0]:.1%} var)')
    ax.set_ylabel(f'PC2  ({var[1]:.1%} var)' if len(var) > 1 else 'PC2  (0.0% var)')
    ax.set_title('Node positions in PC space')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: loading bar chart — contribution of each feature to PC1 and PC2
    ax = axes[1]
    x_pos = np.arange(len(feature_names))
    bar_w = 0.35
    ax.bar(x_pos - bar_w / 2, loadings[0], bar_w,
           label=f'PC1 ({var[0]:.1%})', color='steelblue', alpha=0.85)
    if len(var) > 1:
        ax.bar(x_pos + bar_w / 2, loadings[1], bar_w,
               label=f'PC2 ({var[1]:.1%})', color='salmon', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Loading')
    ax.set_title('Feature loadings per PC\n(which features drive each component?)')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5b_node_pca_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved → {out}")

except FileNotFoundError as e:
    log.warning(f"  ✗ PCA plot skipped — missing file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: Single vs collaborative auditors (any coalition size available)
#         Two true DP reference lines: model/val and data labels.
#         Groups: single | pairs (if present) | triples (if present) | ...
# ─────────────────────────────────────────────────────────────────────────────
# Build ordered list of (group_label, results_list, bar_color, hatch)
_collab_groups = [
    ('single',  full_results,           NODE_COLORS,  ''),
    ('pairs',   collab_pair_results,    ['#b0b0b0'],  '//'),
    ('triples', collab_triple_results,  ['#606060'],  'xx'),
]
# Keep only groups that have data
_collab_groups = [(lbl, res, col, hat)
                  for lbl, res, col, hat in _collab_groups if res]

if _collab_groups:
    log.info("  Generating Plot 7: Single vs collaborative auditors (all sizes)...")

    target_ids_collab = sorted(set(r['target_id'] for r in full_results))
    n_targets  = len(target_ids_collab)

    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), sharey=True)
    if n_targets == 1:
        axes = [axes]
    group_names = ' vs '.join(lbl for lbl, _, _, _ in _collab_groups)
    fig.suptitle(f'Collaborative Auditors per Target: {group_names}\n'
                 '(bars = estimated DP gap; dashed = true DP model/val, dotted = true DP data)',
                 fontsize=13, fontweight='bold')

    for ax, target_id in zip(axes, target_ids_collab):
        all_labels, all_values, all_colors, all_hatches = [], [], [], []
        group_spans = []   # (start_x, end_x, label) for dividers + group text

        for lbl, res, col_spec, hatch in _collab_groups:
            group_rows = sorted([r for r in res if r['target_id'] == target_id],
                                key=lambda r: str(r['auditor_id']))
            start_x = len(all_labels)
            for r in group_rows:
                all_labels.append(f"N{r['auditor_id']}")
                all_values.append(r['est_dp_gap'])
                # single uses per-node color; collab groups use fixed color
                if lbl == 'single':
                    all_colors.append(NODE_COLORS[r['auditor_id'] - 1])
                else:
                    all_colors.append(col_spec[0])
                all_hatches.append(hatch)
            group_spans.append((start_x, len(all_labels) - 1, lbl))

        singles_ref = [r for r in full_results if r['target_id'] == target_id]
        true_dp_mv   = singles_ref[0]['true_dp_gap_model_val']           if singles_ref else 0.0
        true_dp_data = singles_ref[0].get('true_dp_gap_data', true_dp_mv) if singles_ref else 0.0

        x_pos = np.arange(len(all_labels))
        for x, val, col, hatch in zip(x_pos, all_values, all_colors, all_hatches):
            ax.bar(x, val, color=col, edgecolor='white', linewidth=0.8,
                   width=0.7, hatch=hatch, zorder=3)
            ax.text(x, val + 0.003, f'{val:.3f}', ha='center', va='bottom',
                    fontsize=6, rotation=90)

        y_top = max(all_values + [true_dp_mv, true_dp_data]) * 1.3
        for i, (start_x, end_x, lbl) in enumerate(group_spans):
            if i > 0:
                ax.axvline(start_x - 0.5, color='gray', linestyle=':', linewidth=1)
            cx = (start_x + end_x) / 2
            ax.text(cx, y_top * 0.97, lbl, ha='center', va='top',
                    fontsize=7, color='gray')

        ax.axhline(true_dp_mv,   color='black', linestyle='--', linewidth=1.5,
                   label=f'True DP model/val ({true_dp_mv:.3f})', zorder=2)
        ax.axhline(true_dp_data, color='gray',  linestyle=':',  linewidth=1.5,
                   label=f'True DP data ({true_dp_data:.3f})', zorder=2)

        ax.set_title(f'Target: Node {target_id}', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_labels, fontsize=6, rotation=45, ha='right')
        ax.set_ylim(0, y_top)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Estimated DP Gap')
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, f'step5b_collab_vs_single_{PARTITION_ATTR}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"  ✓ Saved → {out}")
else:
    log.warning("  ✗ Plot 7 skipped — no collab results found")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'='*70}")
log.info("  Step 5b Complete")
log.info(f"{'='*70}")
log.info(f"\n  All plots saved to: {PLOT_DIR}")
log.info(f"  Each plot generated 3x — one per ground truth definition:")
for gt_name, gt_label in [('data','Data Labels'), ('model_val','Model Val'), ('model_full','Model Full')]:
    log.info(f"\n  [{gt_label}]")
    for name in ['per_auditee_budget_bars', 'confidence_intervals',
                 'combined_all_modes_bars', 'auditor_consistency']:
        log.info(f"    step5b_{name}_{PARTITION_ATTR}_{gt_name}.png")
    for shading_mode in ['std', 'ci']:
        for name in ['budget_labelled', 'per_auditee_budget', 'bias_variance', 'combined_all_modes']:
            log.info(f"    step5b_{name}_{PARTITION_ATTR}_{gt_name}_{shading_mode}.png")
log.info(f"\n  logs/step5b_{PARTITION_ATTR}.log")
log.info(f"{'='*70}")






# """
# Step 5 Extra Plots
# ===================
# Reads from already-generated step5_audit_results.json and produces
# additional visualisations. No re-running of audits needed.

# Run with:
#     python step5_extra_plots.py
#     python3 step_5_audit_extra_plots.py --config config.yaml

# Produces:
#     plots/step5b_budget_labelled.png           (1) fixed budget labels
#     plots/step5b_per_auditee_budget.png        (2) per-auditee budget curves
#     plots/step5b_confidence_intervals.png      (3) CI plot per pair per budget
#     plots/step5b_bias_variance.png             (4) bias vs variance decomposition
#     plots/step5b_combined_all_modes.png        (5) global + local + budgeted together
#     plots/step5b_auditor_consistency.png       (6) per-auditor consistency box plots
# """

# import os
# import sys
# import json
# import argparse
# import logging
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# from scipy import stats as scipy_stats
# import yaml

# # ── Argument parsing ───────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser(description='Step 5 Extra Plots')
# parser.add_argument('--config', type=str, default='config.yaml')
# args = parser.parse_args()

# with open(args.config) as f:
#     cfg = yaml.safe_load(f)

# EXP_NAME  = cfg['experiment']['name']
# NUM_NODES = cfg['partition']['num_nodes']
# NFS_ROOT  = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
# EXP_DIR   = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
# PLOT_DIR  = os.path.join(EXP_DIR, 'plots')
# LOG_DIR   = os.path.join(EXP_DIR, 'logs')

# os.makedirs(PLOT_DIR, exist_ok=True)

# # ── Logging ────────────────────────────────────────────────────────────────────
# log_path = os.path.join(LOG_DIR, 'step5b.log')
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler(log_path, mode='w')
#     ]
# )
# log = logging.getLogger()

# # ── Style constants ────────────────────────────────────────────────────────────
# NODE_COLORS   = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']
# LINE_STYLES   = ['-', '--', '-.', ':', (0,(3,1,1,1))]   # one per target node
# MARKER_STYLES = ['o', 's', '^', 'D', 'v']

# # ── Load results ───────────────────────────────────────────────────────────────
# log.info("=" * 70)
# log.info("  Step 5b: Extra Plots")
# log.info("=" * 70)

# results_path = os.path.join(EXP_DIR, 'results', 'step5_audit_results.json')
# log.info(f"\n  Loading results from {results_path}...")
# with open(results_path) as f:
#     data = json.load(f)

# full_results   = data['full_results']
# budget_results = data['budget_results']
# global_results = data['global_results']
# true_dp_gaps   = {int(k): v for k, v in data['true_dp_gaps'].items()}
# budget_sizes   = sorted(set(r['budget'] for r in budget_results))
# auditor_ids    = sorted(set(r['auditor_id'] for r in full_results))
# target_ids     = sorted(true_dp_gaps.keys())

# log.info(f"  ✓ Loaded")
# log.info(f"    Full results   : {len(full_results)}")
# log.info(f"    Budget results : {len(budget_results)}")
# log.info(f"    Global results : {len(global_results)}")
# log.info(f"    Budget sizes   : {budget_sizes}")
# log.info(f"    Nodes          : {NUM_NODES}\n")


# # ─────────────────────────────────────────────────────────────────────────────
# # Helper: get budget-aggregated result for a specific (auditor, target, budget)
# # ─────────────────────────────────────────────────────────────────────────────
# def get_budget_agg(auditor_id, target_id, budget):
#     matches = [r for r in budget_results
#                if r['auditor_id'] == auditor_id
#                and r['target_id'] == target_id
#                and r['budget'] == budget]
#     return matches[0] if matches else None


# def get_full(auditor_id, target_id):
#     matches = [r for r in full_results
#                if r['auditor_id'] == auditor_id
#                and r['target_id'] == target_id]
#     return matches[0] if matches else None


# def get_global(target_id):
#     matches = [r for r in global_results if r['target_id'] == target_id]
#     return matches[0] if matches else None


# # ─────────────────────────────────────────────────────────────────────────────
# # Plot 1: Budget curves with proper N1→N3 labels
# # ─────────────────────────────────────────────────────────────────────────────
# log.info("  Generating Plot 1: Budget curves with N_aud→N_tgt labels...")

# fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# fig.suptitle('Budgeted Audit — Sample Efficiency\n'
#              'Label format: Nauditor→Ntarget',
#              fontsize=13, fontweight='bold')

# # Left panel: abs error vs budget, one line per pair
# ax = axes[0]
# for aud_id in auditor_ids:
#     for tgt_id in target_ids:
#         if aud_id == tgt_id:
#             continue
#         pair_data = sorted(
#             [r for r in budget_results
#              if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
#             key=lambda r: r['budget']
#         )
#         if not pair_data:
#             continue
#         budgets   = [r['budget']         for r in pair_data]
#         mean_errs = [r['mean_abs_error'] for r in pair_data]
#         # std of abs errors across repeats — matches the y-axis (mean abs error)
#         std_errs  = [
#             float(np.std([rep['abs_error'] for rep in r['repeats']]))
#             for r in pair_data
#         ]

#         color     = NODE_COLORS[aud_id - 1]
#         linestyle = LINE_STYLES[(tgt_id - 1) % len(LINE_STYLES)]
#         label     = f'N{aud_id}→N{tgt_id}'

#         ax.plot(budgets, mean_errs, color=color, linestyle=linestyle,
#                 linewidth=1.5, marker=MARKER_STYLES[tgt_id-1],
#                 markersize=4, label=label)

# ax.set_xlabel('Query Budget')
# ax.set_ylabel('Mean Absolute Error')
# ax.set_title('All Pairs (colour=auditor, style=target)')
# ax.set_xscale('log')
# ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
#           fontsize=7, ncol=1)
# ax.spines[['top','right']].set_visible(False)

# # Right panel: averaged across all pairs with std shading
# ax = axes[1]
# mean_per_budget = []
# std_per_budget  = []
# for budget in budget_sizes:
#     # Collect all per-repeat abs errors across all pairs at this budget,
#     # then take std — this is the uncertainty on the mean abs error curve
#     all_abs_errs = [
#         rep['abs_error']
#         for r in budget_results if r['budget'] == budget
#         for rep in r['repeats']
#     ]
#     mean_per_budget.append(np.mean([r['mean_abs_error'] for r in budget_results
#                                     if r['budget'] == budget]))
#     std_per_budget.append(np.std(all_abs_errs))

# m = np.array(mean_per_budget)
# s = np.array(std_per_budget)
# ax.plot(budget_sizes, m, color='steelblue', linewidth=2,
#         marker='o', label='Mean ± std across all pairs')
# ax.fill_between(budget_sizes, m - s, m + s,
#                 alpha=0.2, color='steelblue')

# # Mark each budget size
# for b, mv in zip(budget_sizes, m):
#     ax.annotate(f'{b}', xy=(b, mv), xytext=(0, 8),
#                 textcoords='offset points', ha='center',
#                 fontsize=8, color='steelblue')

# ax.set_xlabel('Query Budget')
# ax.set_ylabel('Mean Absolute Error')
# ax.set_title('Averaged Across All Pairs\n(shading = ±1 std)')
# ax.set_xscale('log')
# ax.legend(fontsize=9)
# ax.spines[['top','right']].set_visible(False)

# plt.tight_layout()
# out = os.path.join(PLOT_DIR, 'step5b_budget_labelled.png')
# plt.savefig(out, dpi=150, bbox_inches='tight')
# plt.close()
# log.info(f"  ✓ Saved → {out}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Plot 2: Per-auditee budget curves
# # One subplot per target node. Each line = one auditor.
# # ─────────────────────────────────────────────────────────────────────────────
# log.info("  Generating Plot 2: Per-auditee budget curves...")

# ncols = min(3, len(target_ids))
# nrows = (len(target_ids) + ncols - 1) // ncols
# fig, axes = plt.subplots(nrows, ncols,
#                           figsize=(5 * ncols, 4 * nrows),
#                           squeeze=False)
# fig.suptitle('Per-Auditee Budget Curves\n'
#              'How does each auditor\'s accuracy improve with more queries?',
#              fontsize=13, fontweight='bold')

# for idx, tgt_id in enumerate(target_ids):
#     ax       = axes[idx // ncols][idx % ncols]
#     true_dp  = true_dp_gaps[tgt_id]

#     for aud_id in auditor_ids:
#         if aud_id == tgt_id:
#             continue
#         pair_data = sorted(
#             [r for r in budget_results
#              if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
#             key=lambda r: r['budget']
#         )
#         if not pair_data:
#             continue
#         budgets   = [r['budget']         for r in pair_data]
#         mean_errs = [r['mean_abs_error'] for r in pair_data]
#         # std of absolute errors across repeats — correct uncertainty for
#         # a mean abs error plot. std_est_dp would be wrong here because
#         # it measures spread of raw DP estimates, not spread of abs errors.
#         std_abs_errs = [
#             float(np.std([rep['abs_error'] for rep in r['repeats']]))
#             for r in pair_data
#         ]

#         color = NODE_COLORS[aud_id - 1]
#         ax.plot(budgets, mean_errs, color=color, linewidth=1.8,
#                 marker='o', markersize=4, label=f'Auditor N{aud_id}')
#         ax.fill_between(budgets,
#                         np.maximum(np.array(mean_errs) - np.array(std_abs_errs), 0),
#                         np.array(mean_errs) + np.array(std_abs_errs),
#                         alpha=0.1, color=color)

#     # Full local reference lines — one dashed line per auditor
#     # coloured to match its budgeted curve, labelled at right edge
#     for aud_id in auditor_ids:
#         if aud_id == tgt_id:
#             continue
#         full = get_full(aud_id, tgt_id)
#         if full:
#             ax.axhline(full['abs_error'], color=NODE_COLORS[aud_id-1],
#                        linestyle='--', alpha=0.5, linewidth=1)
#             # Label at right edge so it is clear which auditor it belongs to
#             ax.text(budget_sizes[-1] * 1.05, full['abs_error'],
#                     f'N{aud_id} full', fontsize=6,
#                     color=NODE_COLORS[aud_id-1], va='center', alpha=0.7)

#     # Global auditor reference line — single red dash-dot, same for all auditors
#     glob = get_global(tgt_id)
#     if glob:
#         ax.axhline(glob['abs_error'], color='red',
#                    linestyle='-.', linewidth=1.5,
#                    label=f'Global (err={glob["abs_error"]:.3f})')
#         ax.text(budget_sizes[-1] * 1.05, glob['abs_error'],
#                 'global', fontsize=6, color='red', va='center')

#     ax.set_title(f'Target: Node {tgt_id}  (true DP={true_dp:.3f})',
#                  fontweight='bold')
#     ax.set_xlabel('Query Budget')
#     ax.set_ylabel('Mean Absolute Error')
#     ax.set_xscale('log')
#     ax.legend(fontsize=7)
#     ax.spines[['top','right']].set_visible(False)

#     # Legend note
#     ax.text(0.98, 0.98,
#             'Dashed = full local per auditor\nDash-dot = global auditor',
#             transform=ax.transAxes, fontsize=6,
#             ha='right', va='top', color='gray')

# # Hide unused subplots
# for idx in range(len(target_ids), nrows * ncols):
#     axes[idx // ncols][idx % ncols].set_visible(False)

# plt.tight_layout()
# out = os.path.join(PLOT_DIR, 'step5b_per_auditee_budget.png')
# plt.savefig(out, dpi=150, bbox_inches='tight')
# plt.close()
# log.info(f"  ✓ Saved → {out}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Plot 3: Confidence intervals per (auditor, target) pair per budget
# # ─────────────────────────────────────────────────────────────────────────────
# log.info("  Generating Plot 3: Confidence interval plot...")

# n_budgets = len(budget_sizes)
# ncols     = min(3, n_budgets)
# nrows     = (n_budgets + ncols - 1) // ncols

# fig, axes = plt.subplots(nrows, ncols,
#                           figsize=(6 * ncols, 4 * nrows),
#                           squeeze=False)
# fig.suptitle('Bootstrap 95% Confidence Intervals on Estimated DP Gap\n'
#              'per Auditor→Target Pair at Each Budget',
#              fontsize=13, fontweight='bold')

# for idx, budget in enumerate(budget_sizes):
#     ax      = axes[idx // ncols][idx % ncols]
#     pairs   = [(aud, tgt)
#                for aud in auditor_ids for tgt in target_ids
#                if aud != tgt]
#     x_ticks = []
#     x_labels= []

#     for x_pos, (aud_id, tgt_id) in enumerate(pairs):
#         r = get_budget_agg(aud_id, tgt_id, budget)
#         if r is None:
#             continue

#         mean_dp = r['mean_est_dp']
#         ci_lo   = r['ci_lower']
#         ci_hi   = r['ci_upper']
#         true_dp = true_dp_gaps[tgt_id]
#         color   = NODE_COLORS[aud_id - 1]

#         # CI bar
#         ax.plot([x_pos, x_pos], [ci_lo, ci_hi],
#                 color=color, linewidth=2, solid_capstyle='round')
#         # Mean point
#         ax.scatter(x_pos, mean_dp, color=color, s=40, zorder=4)
#         # True DP tick
#         ax.scatter(x_pos, true_dp, color='black',
#                    marker='_', s=80, zorder=5, linewidths=2)

#         x_ticks.append(x_pos)
#         x_labels.append(f'N{aud_id}→N{tgt_id}')

#     ax.set_title(f'Budget = {budget}', fontweight='bold')
#     ax.set_xticks(x_ticks)
#     ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
#     ax.set_ylabel('Estimated DP Gap')
#     ax.spines[['top','right']].set_visible(False)

#     # Legend
#     ci_line   = mlines.Line2D([], [], color='gray', linewidth=2,
#                                label='95% CI')
#     true_mark = mlines.Line2D([], [], color='black', marker='_',
#                                markersize=8, linewidth=0,
#                                label='True DP gap')
#     ax.legend(handles=[ci_line, true_mark], fontsize=7)

# # Hide unused
# for idx in range(n_budgets, nrows * ncols):
#     axes[idx // ncols][idx % ncols].set_visible(False)

# plt.tight_layout()
# out = os.path.join(PLOT_DIR, 'step5b_confidence_intervals.png')
# plt.savefig(out, dpi=150, bbox_inches='tight')
# plt.close()
# log.info(f"  ✓ Saved → {out}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Plot 4: Bias vs variance decomposition
# # bias  = mean(estimated DP) - true DP     (systematic over/underestimation)
# # variance = std(estimated DP)²            (sensitivity to query sample)
# # ─────────────────────────────────────────────────────────────────────────────
# log.info("  Generating Plot 4: Bias vs variance decomposition...")

# fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# fig.suptitle('Bias vs Variance Decomposition of Audit Error\n'
#              'bias = mean(est DP) − true DP  |  '
#              'variance = std(est DP)²  |  '
#              'MSE = bias² + variance',
#              fontsize=12, fontweight='bold')

# # Collect bias and variance per pair per budget
# bv_data = {}   # (aud, tgt, budget) → {bias, variance, mse}
# for r in budget_results:
#     aud_id  = r['auditor_id']
#     tgt_id  = r['target_id']
#     budget  = r['budget']
#     true_dp = true_dp_gaps[tgt_id]

#     # Extract per-repeat estimates
#     repeat_ests = [rep['est_dp_gap'] for rep in r['repeats']]
#     mean_est    = np.mean(repeat_ests)
#     std_est     = np.std(repeat_ests)

#     bias     = float(mean_est - true_dp)   # signed: positive = overestimate
#     variance = float(std_est ** 2)
#     mse      = float(bias ** 2 + variance)

#     bv_data[(aud_id, tgt_id, budget)] = {
#         'bias': bias, 'variance': variance, 'mse': mse,
#         'abs_bias': abs(bias)
#     }

# # Panel 1: |bias| vs budget, averaged across pairs
# ax = axes[0]
# mean_bias_per_budget = []
# std_bias_per_budget  = []
# for budget in budget_sizes:
#     abs_biases = [bv_data[(a, t, budget)]['abs_bias']
#                   for a in auditor_ids for t in target_ids
#                   if a != t and (a, t, budget) in bv_data]
#     mean_bias_per_budget.append(np.mean(abs_biases))
#     std_bias_per_budget.append(np.std(abs_biases))

# m = np.array(mean_bias_per_budget)
# s = np.array(std_bias_per_budget)
# ax.plot(budget_sizes, m, color='tomato', linewidth=2,
#         marker='o', label='Mean |bias|')
# ax.fill_between(budget_sizes, m - s, m + s, alpha=0.2, color='tomato')
# ax.set_xlabel('Query Budget'); ax.set_ylabel('|Bias|')
# ax.set_title('|Bias| vs Budget\n(does more data reduce systematic error?)')
# ax.set_xscale('log')
# ax.legend(fontsize=8)
# ax.spines[['top','right']].set_visible(False)

# # Panel 2: variance vs budget, averaged across pairs
# ax = axes[1]
# mean_var_per_budget = []
# std_var_per_budget  = []
# for budget in budget_sizes:
#     variances = [bv_data[(a, t, budget)]['variance']
#                  for a in auditor_ids for t in target_ids
#                  if a != t and (a, t, budget) in bv_data]
#     mean_var_per_budget.append(np.mean(variances))
#     std_var_per_budget.append(np.std(variances))

# m = np.array(mean_var_per_budget)
# s = np.array(std_var_per_budget)
# ax.plot(budget_sizes, m, color='steelblue', linewidth=2,
#         marker='o', label='Mean variance')
# ax.fill_between(budget_sizes, np.maximum(m - s, 0), m + s,
#                 alpha=0.2, color='steelblue')
# ax.set_xlabel('Query Budget'); ax.set_ylabel('Variance')
# ax.set_title('Variance vs Budget\n(does more data reduce noise?)')
# ax.set_xscale('log')
# ax.legend(fontsize=8)
# ax.spines[['top','right']].set_visible(False)

# # Panel 3: stacked bias² + variance = MSE at each budget
# ax = axes[2]
# mean_bias2 = [np.mean([bv_data[(a,t,b)]['bias']**2
#                         for a in auditor_ids for t in target_ids
#                         if a != t and (a,t,b) in bv_data])
#               for b in budget_sizes]
# mean_var   = [np.mean([bv_data[(a,t,b)]['variance']
#                         for a in auditor_ids for t in target_ids
#                         if a != t and (a,t,b) in bv_data])
#               for b in budget_sizes]

# x = np.arange(len(budget_sizes))
# w = 0.5
# ax.bar(x, mean_bias2, w, label='Bias²',     color='tomato',    alpha=0.85)
# ax.bar(x, mean_var,   w, bottom=mean_bias2, label='Variance',
#        color='steelblue', alpha=0.85)
# ax.set_xticks(x)
# ax.set_xticklabels([str(b) for b in budget_sizes])
# ax.set_xlabel('Query Budget'); ax.set_ylabel('MSE')
# ax.set_title('MSE = Bias² + Variance\n(what drives the error at each budget?)')
# ax.legend(fontsize=8)
# ax.spines[['top','right']].set_visible(False)

# plt.tight_layout()
# out = os.path.join(PLOT_DIR, 'step5b_bias_variance.png')
# plt.savefig(out, dpi=150, bbox_inches='tight')
# plt.close()
# log.info(f"  ✓ Saved → {out}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Plot 5: Global + full local + budgeted on one axis per target node
# # ─────────────────────────────────────────────────────────────────────────────
# log.info("  Generating Plot 5: Combined all modes per target...")

# ncols = min(3, len(target_ids))
# nrows = (len(target_ids) + ncols - 1) // ncols
# fig, axes = plt.subplots(nrows, ncols,
#                           figsize=(6 * ncols, 5 * nrows),
#                           squeeze=False)
# fig.suptitle('All Audit Modes Combined per Target Node\n'
#              'Global  |  Full Local  |  Budgeted (with CI)',
#              fontsize=13, fontweight='bold')

# for idx, tgt_id in enumerate(target_ids):
#     ax      = axes[idx // ncols][idx % ncols]
#     true_dp = true_dp_gaps[tgt_id]

#     # True DP reference line
#     ax.axhline(true_dp, color='black', linestyle='--',
#                linewidth=1.5, label=f'True DP ({true_dp:.3f})', zorder=1)

#     # Global estimate — horizontal band
#     glob = get_global(tgt_id)
#     if glob:
#         ax.axhline(glob['est_dp_gap'], color='red', linestyle='-.',
#                    linewidth=1.5,
#                    label=f'Global ({glob["est_dp_gap"]:.3f})', zorder=2)

#     # For each auditor: full local + budgeted curve
#     for aud_id in auditor_ids:
#         if aud_id == tgt_id:
#             continue
#         color = NODE_COLORS[aud_id - 1]

#         # Full local — single point at x = max_budget * 1.3 (right side)
#         full = get_full(aud_id, tgt_id)

#         # Budgeted curve
#         pair_budget = sorted(
#             [r for r in budget_results
#              if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
#             key=lambda r: r['budget']
#         )
#         if pair_budget:
#             budgets   = [r['budget']     for r in pair_budget]
#             mean_ests = [r['mean_est_dp'] for r in pair_budget]
#             ci_los    = [r['ci_lower']    for r in pair_budget]
#             ci_his    = [r['ci_upper']    for r in pair_budget]

#             ax.plot(budgets, mean_ests, color=color, linewidth=1.8,
#                     marker='o', markersize=4,
#                     label=f'N{aud_id} budgeted', zorder=3)
#             ax.fill_between(budgets, ci_los, ci_his,
#                             alpha=0.15, color=color)

#         # Full local — plotted as a star at x just beyond max budget
#         if full:
#             x_full = budget_sizes[-1] * 1.5
#             ax.scatter(x_full, full['est_dp_gap'],
#                        color=color, marker='*', s=180, zorder=5)

#     # Add a text label for the star markers
#     ax.text(0.97, 0.04, '★ = full local',
#             transform=ax.transAxes, fontsize=7,
#             ha='right', va='bottom', color='gray')

#     ax.set_title(f'Target: Node {tgt_id}', fontweight='bold')
#     ax.set_xlabel('Query Budget (★ = full local data)')
#     ax.set_ylabel('Estimated DP Gap')
#     ax.set_xscale('log')
#     ax.legend(fontsize=7, loc='upper right')
#     ax.spines[['top','right']].set_visible(False)

# # Hide unused subplots
# for idx in range(len(target_ids), nrows * ncols):
#     axes[idx // ncols][idx % ncols].set_visible(False)

# plt.tight_layout()
# out = os.path.join(PLOT_DIR, 'step5b_combined_all_modes.png')
# plt.savefig(out, dpi=150, bbox_inches='tight')
# plt.close()
# log.info(f"  ✓ Saved → {out}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Plot 6: Per-auditor consistency box plots
# # For each auditor: distribution of absolute errors across all targets
# # and all budget sizes. Shows which auditor is most/least consistent.
# # ─────────────────────────────────────────────────────────────────────────────
# log.info("  Generating Plot 6: Per-auditor consistency box plots...")

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle('Per-Auditor Consistency\n'
#              'How much does error vary across different targets?',
#              fontsize=13, fontweight='bold')

# # Panel 1: full local — box per auditor, one point per target
# ax = axes[0]
# full_errors_per_auditor = []
# labels = []
# for aud_id in auditor_ids:
#     errs = [r['abs_error'] for r in full_results
#             if r['auditor_id'] == aud_id]
#     full_errors_per_auditor.append(errs)
#     labels.append(f'Node {aud_id}\n(n={len(errs)})')

# bp = ax.boxplot(full_errors_per_auditor, patch_artist=True,
#                 medianprops=dict(color='black', linewidth=2))
# for patch, color in zip(bp['boxes'], NODE_COLORS):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.7)

# # Overlay individual points
# for x_pos, (errs, aud_id) in enumerate(
#         zip(full_errors_per_auditor, auditor_ids), start=1):
#     jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(errs))
#     ax.scatter(x_pos + jitter, errs,
#                color=NODE_COLORS[aud_id-1], alpha=0.6, s=40, zorder=3)

# ax.set_xticklabels(labels, fontsize=9)
# ax.set_ylabel('Absolute Error')
# ax.set_title('Full Local Audit\nError distribution across targets per auditor')
# ax.spines[['top','right']].set_visible(False)

# # Panel 2: budgeted — box per auditor at each budget,
# # grouped by auditor with budget as hue via offset x positions
# ax = axes[1]
# n_auditors = len(auditor_ids)
# n_budgets  = len(budget_sizes)
# group_width = 0.8
# bar_width   = group_width / n_budgets
# budget_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_budgets))

# for b_idx, budget in enumerate(budget_sizes):
#     for a_idx, aud_id in enumerate(auditor_ids):
#         errs = [r['mean_abs_error'] for r in budget_results
#                 if r['auditor_id'] == aud_id and r['budget'] == budget]
#         if not errs:
#             continue
#         x = a_idx + (b_idx - n_budgets/2 + 0.5) * bar_width
#         ax.bar(x, np.mean(errs), bar_width * 0.9,
#                color=budget_colors[b_idx], edgecolor='white',
#                linewidth=0.5,
#                label=f'Budget {budget}' if a_idx == 0 else '')

# ax.set_xticks(range(n_auditors))
# ax.set_xticklabels([f'Node {i}' for i in auditor_ids], fontsize=9)
# ax.set_ylabel('Mean Absolute Error')
# ax.set_title('Budgeted Audit\nMean error per auditor at each budget\n'
#              '(darker = larger budget)')
# ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
# ax.spines[['top','right']].set_visible(False)

# plt.tight_layout()
# out = os.path.join(PLOT_DIR, 'step5b_auditor_consistency.png')
# plt.savefig(out, dpi=150, bbox_inches='tight')
# plt.close()
# log.info(f"  ✓ Saved → {out}")


# # ─────────────────────────────────────────────────────────────────────────────
# # Summary
# # ─────────────────────────────────────────────────────────────────────────────
# log.info(f"\n{'='*70}")
# log.info("  Step 5b Complete")
# log.info(f"{'='*70}")
# log.info(f"\n  All plots saved to: {PLOT_DIR}")
# log.info(f"    step5b_budget_labelled.png")
# log.info(f"    step5b_per_auditee_budget.png")
# log.info(f"    step5b_confidence_intervals.png")
# log.info(f"    step5b_bias_variance.png")
# log.info(f"    step5b_combined_all_modes.png")
# log.info(f"    step5b_auditor_consistency.png")
# log.info(f"    logs/step5b.log")
# log.info(f"{'='*70}")





# # """
# # Step 5 Extra Plots
# # ===================
# # Reads from already-generated step5_audit_results.json and produces
# # additional visualisations. No re-running of audits needed.

# # Run with:
# #     python step5_extra_plots.py
# #     python3 step_5_audit_extra_plots.py --config config.yaml

# # Produces:
# #     plots/step5b_budget_labelled.png           (1) fixed budget labels
# #     plots/step5b_per_auditee_budget.png        (2) per-auditee budget curves
# #     plots/step5b_confidence_intervals.png      (3) CI plot per pair per budget
# #     plots/step5b_bias_variance.png             (4) bias vs variance decomposition
# #     plots/step5b_combined_all_modes.png        (5) global + local + budgeted together
# #     plots/step5b_auditor_consistency.png       (6) per-auditor consistency box plots
# # """

# # import os
# # import sys
# # import json
# # import argparse
# # import logging
# # import numpy as np
# # import matplotlib
# # matplotlib.use('Agg')
# # import matplotlib.pyplot as plt
# # import matplotlib.lines as mlines
# # from scipy import stats as scipy_stats
# # import yaml

# # # ── Argument parsing ───────────────────────────────────────────────────────────
# # parser = argparse.ArgumentParser(description='Step 5 Extra Plots')
# # parser.add_argument('--config', type=str, default='config.yaml')
# # args = parser.parse_args()

# # with open(args.config) as f:
# #     cfg = yaml.safe_load(f)

# # EXP_NAME  = cfg['experiment']['name']
# # NUM_NODES = cfg['partition']['num_nodes']
# # NFS_ROOT  = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
# # EXP_DIR   = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
# # PLOT_DIR  = os.path.join(EXP_DIR, 'plots')
# # LOG_DIR   = os.path.join(EXP_DIR, 'logs')

# # os.makedirs(PLOT_DIR, exist_ok=True)

# # # ── Logging ────────────────────────────────────────────────────────────────────
# # log_path = os.path.join(LOG_DIR, 'step5b.log')
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(message)s',
# #     handlers=[
# #         logging.StreamHandler(sys.stdout),
# #         logging.FileHandler(log_path, mode='w')
# #     ]
# # )
# # log = logging.getLogger()

# # # ── Style constants ────────────────────────────────────────────────────────────
# # NODE_COLORS   = ['steelblue', 'salmon', 'mediumseagreen', 'mediumpurple', 'sandybrown']
# # LINE_STYLES   = ['-', '--', '-.', ':', (0,(3,1,1,1))]   # one per target node
# # MARKER_STYLES = ['o', 's', '^', 'D', 'v']

# # # ── Load results ───────────────────────────────────────────────────────────────
# # log.info("=" * 70)
# # log.info("  Step 5b: Extra Plots")
# # log.info("=" * 70)

# # results_path = os.path.join(EXP_DIR, 'results', 'step5_audit_results.json')
# # log.info(f"\n  Loading results from {results_path}...")
# # with open(results_path) as f:
# #     data = json.load(f)

# # full_results   = data['full_results']
# # budget_results = data['budget_results']
# # global_results = data['global_results']
# # true_dp_gaps   = {int(k): v for k, v in data['true_dp_gaps'].items()}
# # budget_sizes   = sorted(set(r['budget'] for r in budget_results))
# # auditor_ids    = sorted(set(r['auditor_id'] for r in full_results))
# # target_ids     = sorted(true_dp_gaps.keys())

# # log.info(f"  ✓ Loaded")
# # log.info(f"    Full results   : {len(full_results)}")
# # log.info(f"    Budget results : {len(budget_results)}")
# # log.info(f"    Global results : {len(global_results)}")
# # log.info(f"    Budget sizes   : {budget_sizes}")
# # log.info(f"    Nodes          : {NUM_NODES}\n")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Helper: get budget-aggregated result for a specific (auditor, target, budget)
# # # ─────────────────────────────────────────────────────────────────────────────
# # def get_budget_agg(auditor_id, target_id, budget):
# #     matches = [r for r in budget_results
# #                if r['auditor_id'] == auditor_id
# #                and r['target_id'] == target_id
# #                and r['budget'] == budget]
# #     return matches[0] if matches else None


# # def get_full(auditor_id, target_id):
# #     matches = [r for r in full_results
# #                if r['auditor_id'] == auditor_id
# #                and r['target_id'] == target_id]
# #     return matches[0] if matches else None


# # def get_global(target_id):
# #     matches = [r for r in global_results if r['target_id'] == target_id]
# #     return matches[0] if matches else None


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Plot 1: Budget curves with proper N1→N3 labels
# # # ─────────────────────────────────────────────────────────────────────────────
# # log.info("  Generating Plot 1: Budget curves with N_aud→N_tgt labels...")

# # fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# # fig.suptitle('Budgeted Audit — Sample Efficiency\n'
# #              'Label format: Nauditor→Ntarget',
# #              fontsize=13, fontweight='bold')

# # # Left panel: abs error vs budget, one line per pair
# # ax = axes[0]
# # for aud_id in auditor_ids:
# #     for tgt_id in target_ids:
# #         if aud_id == tgt_id:
# #             continue
# #         pair_data = sorted(
# #             [r for r in budget_results
# #              if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
# #             key=lambda r: r['budget']
# #         )
# #         if not pair_data:
# #             continue
# #         budgets   = [r['budget']         for r in pair_data]
# #         mean_errs = [r['mean_abs_error'] for r in pair_data]
# #         std_errs  = [r['std_est_dp']     for r in pair_data]

# #         color     = NODE_COLORS[aud_id - 1]
# #         linestyle = LINE_STYLES[(tgt_id - 1) % len(LINE_STYLES)]
# #         label     = f'N{aud_id}→N{tgt_id}'

# #         ax.plot(budgets, mean_errs, color=color, linestyle=linestyle,
# #                 linewidth=1.5, marker=MARKER_STYLES[tgt_id-1],
# #                 markersize=4, label=label)

# # ax.set_xlabel('Query Budget')
# # ax.set_ylabel('Mean Absolute Error')
# # ax.set_title('All Pairs (colour=auditor, style=target)')
# # ax.set_xscale('log')
# # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
# #           fontsize=7, ncol=1)
# # ax.spines[['top','right']].set_visible(False)

# # # Right panel: averaged across all pairs with std shading
# # ax = axes[1]
# # mean_per_budget = []
# # std_per_budget  = []
# # for budget in budget_sizes:
# #     errs = [r['mean_abs_error'] for r in budget_results
# #             if r['budget'] == budget]
# #     mean_per_budget.append(np.mean(errs))
# #     std_per_budget.append(np.std(errs))

# # m = np.array(mean_per_budget)
# # s = np.array(std_per_budget)
# # ax.plot(budget_sizes, m, color='steelblue', linewidth=2,
# #         marker='o', label='Mean ± std across all pairs')
# # ax.fill_between(budget_sizes, m - s, m + s,
# #                 alpha=0.2, color='steelblue')

# # # Mark each budget size
# # for b, mv in zip(budget_sizes, m):
# #     ax.annotate(f'{b}', xy=(b, mv), xytext=(0, 8),
# #                 textcoords='offset points', ha='center',
# #                 fontsize=8, color='steelblue')

# # ax.set_xlabel('Query Budget')
# # ax.set_ylabel('Mean Absolute Error')
# # ax.set_title('Averaged Across All Pairs\n(shading = ±1 std)')
# # ax.set_xscale('log')
# # ax.legend(fontsize=9)
# # ax.spines[['top','right']].set_visible(False)

# # plt.tight_layout()
# # out = os.path.join(PLOT_DIR, 'step5b_budget_labelled.png')
# # plt.savefig(out, dpi=150, bbox_inches='tight')
# # plt.close()
# # log.info(f"  ✓ Saved → {out}")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Plot 2: Per-auditee budget curves
# # # One subplot per target node. Each line = one auditor.
# # # ─────────────────────────────────────────────────────────────────────────────
# # log.info("  Generating Plot 2: Per-auditee budget curves...")

# # ncols = min(3, len(target_ids))
# # nrows = (len(target_ids) + ncols - 1) // ncols
# # fig, axes = plt.subplots(nrows, ncols,
# #                           figsize=(5 * ncols, 4 * nrows),
# #                           squeeze=False)
# # fig.suptitle('Per-Auditee Budget Curves\n'
# #              'How does each auditor\'s accuracy improve with more queries?',
# #              fontsize=13, fontweight='bold')

# # for idx, tgt_id in enumerate(target_ids):
# #     ax       = axes[idx // ncols][idx % ncols]
# #     true_dp  = true_dp_gaps[tgt_id]

# #     for aud_id in auditor_ids:
# #         if aud_id == tgt_id:
# #             continue
# #         pair_data = sorted(
# #             [r for r in budget_results
# #              if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
# #             key=lambda r: r['budget']
# #         )
# #         if not pair_data:
# #             continue
# #         budgets   = [r['budget']         for r in pair_data]
# #         mean_errs = [r['mean_abs_error'] for r in pair_data]
# #         std_errs  = [r['std_est_dp']     for r in pair_data]

# #         color = NODE_COLORS[aud_id - 1]
# #         ax.plot(budgets, mean_errs, color=color, linewidth=1.8,
# #                 marker='o', markersize=4, label=f'Auditor N{aud_id}')
# #         ax.fill_between(budgets,
# #                         np.array(mean_errs) - np.array(std_errs),
# #                         np.array(mean_errs) + np.array(std_errs),
# #                         alpha=0.1, color=color)

# #     # Full local reference lines — one dashed line per auditor
# #     # coloured to match its budgeted curve, labelled at right edge
# #     for aud_id in auditor_ids:
# #         if aud_id == tgt_id:
# #             continue
# #         full = get_full(aud_id, tgt_id)
# #         if full:
# #             ax.axhline(full['abs_error'], color=NODE_COLORS[aud_id-1],
# #                        linestyle='--', alpha=0.5, linewidth=1)
# #             # Label at right edge so it is clear which auditor it belongs to
# #             ax.text(budget_sizes[-1] * 1.05, full['abs_error'],
# #                     f'N{aud_id} full', fontsize=6,
# #                     color=NODE_COLORS[aud_id-1], va='center', alpha=0.7)

# #     # Global auditor reference line — single red dash-dot, same for all auditors
# #     glob = get_global(tgt_id)
# #     if glob:
# #         ax.axhline(glob['abs_error'], color='red',
# #                    linestyle='-.', linewidth=1.5,
# #                    label=f'Global (err={glob["abs_error"]:.3f})')
# #         ax.text(budget_sizes[-1] * 1.05, glob['abs_error'],
# #                 'global', fontsize=6, color='red', va='center')

# #     ax.set_title(f'Target: Node {tgt_id}  (true DP={true_dp:.3f})',
# #                  fontweight='bold')
# #     ax.set_xlabel('Query Budget')
# #     ax.set_ylabel('Mean Absolute Error')
# #     ax.set_xscale('log')
# #     ax.legend(fontsize=7)
# #     ax.spines[['top','right']].set_visible(False)

# #     # Legend note
# #     ax.text(0.98, 0.98,
# #             'Dashed = full local per auditor\nDash-dot = global auditor',
# #             transform=ax.transAxes, fontsize=6,
# #             ha='right', va='top', color='gray')

# # # Hide unused subplots
# # for idx in range(len(target_ids), nrows * ncols):
# #     axes[idx // ncols][idx % ncols].set_visible(False)

# # plt.tight_layout()
# # out = os.path.join(PLOT_DIR, 'step5b_per_auditee_budget.png')
# # plt.savefig(out, dpi=150, bbox_inches='tight')
# # plt.close()
# # log.info(f"  ✓ Saved → {out}")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Plot 3: Confidence intervals per (auditor, target) pair per budget
# # # ─────────────────────────────────────────────────────────────────────────────
# # log.info("  Generating Plot 3: Confidence interval plot...")

# # n_budgets = len(budget_sizes)
# # ncols     = min(3, n_budgets)
# # nrows     = (n_budgets + ncols - 1) // ncols

# # fig, axes = plt.subplots(nrows, ncols,
# #                           figsize=(6 * ncols, 4 * nrows),
# #                           squeeze=False)
# # fig.suptitle('Bootstrap 95% Confidence Intervals on Estimated DP Gap\n'
# #              'per Auditor→Target Pair at Each Budget',
# #              fontsize=13, fontweight='bold')

# # for idx, budget in enumerate(budget_sizes):
# #     ax      = axes[idx // ncols][idx % ncols]
# #     pairs   = [(aud, tgt)
# #                for aud in auditor_ids for tgt in target_ids
# #                if aud != tgt]
# #     x_ticks = []
# #     x_labels= []

# #     for x_pos, (aud_id, tgt_id) in enumerate(pairs):
# #         r = get_budget_agg(aud_id, tgt_id, budget)
# #         if r is None:
# #             continue

# #         mean_dp = r['mean_est_dp']
# #         ci_lo   = r['ci_lower']
# #         ci_hi   = r['ci_upper']
# #         true_dp = true_dp_gaps[tgt_id]
# #         color   = NODE_COLORS[aud_id - 1]

# #         # CI bar
# #         ax.plot([x_pos, x_pos], [ci_lo, ci_hi],
# #                 color=color, linewidth=2, solid_capstyle='round')
# #         # Mean point
# #         ax.scatter(x_pos, mean_dp, color=color, s=40, zorder=4)
# #         # True DP tick
# #         ax.scatter(x_pos, true_dp, color='black',
# #                    marker='_', s=80, zorder=5, linewidths=2)

# #         x_ticks.append(x_pos)
# #         x_labels.append(f'N{aud_id}→N{tgt_id}')

# #     ax.set_title(f'Budget = {budget}', fontweight='bold')
# #     ax.set_xticks(x_ticks)
# #     ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
# #     ax.set_ylabel('Estimated DP Gap')
# #     ax.spines[['top','right']].set_visible(False)

# #     # Legend
# #     ci_line   = mlines.Line2D([], [], color='gray', linewidth=2,
# #                                label='95% CI')
# #     true_mark = mlines.Line2D([], [], color='black', marker='_',
# #                                markersize=8, linewidth=0,
# #                                label='True DP gap')
# #     ax.legend(handles=[ci_line, true_mark], fontsize=7)

# # # Hide unused
# # for idx in range(n_budgets, nrows * ncols):
# #     axes[idx // ncols][idx % ncols].set_visible(False)

# # plt.tight_layout()
# # out = os.path.join(PLOT_DIR, 'step5b_confidence_intervals.png')
# # plt.savefig(out, dpi=150, bbox_inches='tight')
# # plt.close()
# # log.info(f"  ✓ Saved → {out}")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Plot 4: Bias vs variance decomposition
# # # bias  = mean(estimated DP) - true DP     (systematic over/underestimation)
# # # variance = std(estimated DP)²            (sensitivity to query sample)
# # # ─────────────────────────────────────────────────────────────────────────────
# # log.info("  Generating Plot 4: Bias vs variance decomposition...")

# # fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# # fig.suptitle('Bias vs Variance Decomposition of Audit Error\n'
# #              'bias = mean(est DP) − true DP  |  '
# #              'variance = std(est DP)²  |  '
# #              'MSE = bias² + variance',
# #              fontsize=12, fontweight='bold')

# # # Collect bias and variance per pair per budget
# # bv_data = {}   # (aud, tgt, budget) → {bias, variance, mse}
# # for r in budget_results:
# #     aud_id  = r['auditor_id']
# #     tgt_id  = r['target_id']
# #     budget  = r['budget']
# #     true_dp = true_dp_gaps[tgt_id]

# #     # Extract per-repeat estimates
# #     repeat_ests = [rep['est_dp_gap'] for rep in r['repeats']]
# #     mean_est    = np.mean(repeat_ests)
# #     std_est     = np.std(repeat_ests)

# #     bias     = float(mean_est - true_dp)   # signed: positive = overestimate
# #     variance = float(std_est ** 2)
# #     mse      = float(bias ** 2 + variance)

# #     bv_data[(aud_id, tgt_id, budget)] = {
# #         'bias': bias, 'variance': variance, 'mse': mse,
# #         'abs_bias': abs(bias)
# #     }

# # # Panel 1: |bias| vs budget, averaged across pairs
# # ax = axes[0]
# # mean_bias_per_budget = []
# # std_bias_per_budget  = []
# # for budget in budget_sizes:
# #     abs_biases = [bv_data[(a, t, budget)]['abs_bias']
# #                   for a in auditor_ids for t in target_ids
# #                   if a != t and (a, t, budget) in bv_data]
# #     mean_bias_per_budget.append(np.mean(abs_biases))
# #     std_bias_per_budget.append(np.std(abs_biases))

# # m = np.array(mean_bias_per_budget)
# # s = np.array(std_bias_per_budget)
# # ax.plot(budget_sizes, m, color='tomato', linewidth=2,
# #         marker='o', label='Mean |bias|')
# # ax.fill_between(budget_sizes, m - s, m + s, alpha=0.2, color='tomato')
# # ax.set_xlabel('Query Budget'); ax.set_ylabel('|Bias|')
# # ax.set_title('|Bias| vs Budget\n(does more data reduce systematic error?)')
# # ax.set_xscale('log')
# # ax.legend(fontsize=8)
# # ax.spines[['top','right']].set_visible(False)

# # # Panel 2: variance vs budget, averaged across pairs
# # ax = axes[1]
# # mean_var_per_budget = []
# # std_var_per_budget  = []
# # for budget in budget_sizes:
# #     variances = [bv_data[(a, t, budget)]['variance']
# #                  for a in auditor_ids for t in target_ids
# #                  if a != t and (a, t, budget) in bv_data]
# #     mean_var_per_budget.append(np.mean(variances))
# #     std_var_per_budget.append(np.std(variances))

# # m = np.array(mean_var_per_budget)
# # s = np.array(std_var_per_budget)
# # ax.plot(budget_sizes, m, color='steelblue', linewidth=2,
# #         marker='o', label='Mean variance')
# # ax.fill_between(budget_sizes, np.maximum(m - s, 0), m + s,
# #                 alpha=0.2, color='steelblue')
# # ax.set_xlabel('Query Budget'); ax.set_ylabel('Variance')
# # ax.set_title('Variance vs Budget\n(does more data reduce noise?)')
# # ax.set_xscale('log')
# # ax.legend(fontsize=8)
# # ax.spines[['top','right']].set_visible(False)

# # # Panel 3: stacked bias² + variance = MSE at each budget
# # ax = axes[2]
# # mean_bias2 = [np.mean([bv_data[(a,t,b)]['bias']**2
# #                         for a in auditor_ids for t in target_ids
# #                         if a != t and (a,t,b) in bv_data])
# #               for b in budget_sizes]
# # mean_var   = [np.mean([bv_data[(a,t,b)]['variance']
# #                         for a in auditor_ids for t in target_ids
# #                         if a != t and (a,t,b) in bv_data])
# #               for b in budget_sizes]

# # x = np.arange(len(budget_sizes))
# # w = 0.5
# # ax.bar(x, mean_bias2, w, label='Bias²',     color='tomato',    alpha=0.85)
# # ax.bar(x, mean_var,   w, bottom=mean_bias2, label='Variance',
# #        color='steelblue', alpha=0.85)
# # ax.set_xticks(x)
# # ax.set_xticklabels([str(b) for b in budget_sizes])
# # ax.set_xlabel('Query Budget'); ax.set_ylabel('MSE')
# # ax.set_title('MSE = Bias² + Variance\n(what drives the error at each budget?)')
# # ax.legend(fontsize=8)
# # ax.spines[['top','right']].set_visible(False)

# # plt.tight_layout()
# # out = os.path.join(PLOT_DIR, 'step5b_bias_variance.png')
# # plt.savefig(out, dpi=150, bbox_inches='tight')
# # plt.close()
# # log.info(f"  ✓ Saved → {out}")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Plot 5: Global + full local + budgeted on one axis per target node
# # # ─────────────────────────────────────────────────────────────────────────────
# # log.info("  Generating Plot 5: Combined all modes per target...")

# # ncols = min(3, len(target_ids))
# # nrows = (len(target_ids) + ncols - 1) // ncols
# # fig, axes = plt.subplots(nrows, ncols,
# #                           figsize=(6 * ncols, 5 * nrows),
# #                           squeeze=False)
# # fig.suptitle('All Audit Modes Combined per Target Node\n'
# #              'Global  |  Full Local  |  Budgeted (with CI)',
# #              fontsize=13, fontweight='bold')

# # for idx, tgt_id in enumerate(target_ids):
# #     ax      = axes[idx // ncols][idx % ncols]
# #     true_dp = true_dp_gaps[tgt_id]

# #     # True DP reference line
# #     ax.axhline(true_dp, color='black', linestyle='--',
# #                linewidth=1.5, label=f'True DP ({true_dp:.3f})', zorder=1)

# #     # Global estimate — horizontal band
# #     glob = get_global(tgt_id)
# #     if glob:
# #         ax.axhline(glob['est_dp_gap'], color='red', linestyle='-.',
# #                    linewidth=1.5,
# #                    label=f'Global ({glob["est_dp_gap"]:.3f})', zorder=2)

# #     # For each auditor: full local + budgeted curve
# #     for aud_id in auditor_ids:
# #         if aud_id == tgt_id:
# #             continue
# #         color = NODE_COLORS[aud_id - 1]

# #         # Full local — single point at x = max_budget * 1.3 (right side)
# #         full = get_full(aud_id, tgt_id)

# #         # Budgeted curve
# #         pair_budget = sorted(
# #             [r for r in budget_results
# #              if r['auditor_id'] == aud_id and r['target_id'] == tgt_id],
# #             key=lambda r: r['budget']
# #         )
# #         if pair_budget:
# #             budgets   = [r['budget']     for r in pair_budget]
# #             mean_ests = [r['mean_est_dp'] for r in pair_budget]
# #             ci_los    = [r['ci_lower']    for r in pair_budget]
# #             ci_his    = [r['ci_upper']    for r in pair_budget]

# #             ax.plot(budgets, mean_ests, color=color, linewidth=1.8,
# #                     marker='o', markersize=4,
# #                     label=f'N{aud_id} budgeted', zorder=3)
# #             ax.fill_between(budgets, ci_los, ci_his,
# #                             alpha=0.15, color=color)

# #         # Full local — plotted as a star at x just beyond max budget
# #         if full:
# #             x_full = budget_sizes[-1] * 1.5
# #             ax.scatter(x_full, full['est_dp_gap'],
# #                        color=color, marker='*', s=180, zorder=5)

# #     # Add a text label for the star markers
# #     ax.text(0.97, 0.04, '★ = full local',
# #             transform=ax.transAxes, fontsize=7,
# #             ha='right', va='bottom', color='gray')

# #     ax.set_title(f'Target: Node {tgt_id}', fontweight='bold')
# #     ax.set_xlabel('Query Budget (★ = full local data)')
# #     ax.set_ylabel('Estimated DP Gap')
# #     ax.set_xscale('log')
# #     ax.legend(fontsize=7, loc='upper right')
# #     ax.spines[['top','right']].set_visible(False)

# # # Hide unused subplots
# # for idx in range(len(target_ids), nrows * ncols):
# #     axes[idx // ncols][idx % ncols].set_visible(False)

# # plt.tight_layout()
# # out = os.path.join(PLOT_DIR, 'step5b_combined_all_modes.png')
# # plt.savefig(out, dpi=150, bbox_inches='tight')
# # plt.close()
# # log.info(f"  ✓ Saved → {out}")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Plot 6: Per-auditor consistency box plots
# # # For each auditor: distribution of absolute errors across all targets
# # # and all budget sizes. Shows which auditor is most/least consistent.
# # # ─────────────────────────────────────────────────────────────────────────────
# # log.info("  Generating Plot 6: Per-auditor consistency box plots...")

# # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# # fig.suptitle('Per-Auditor Consistency\n'
# #              'How much does error vary across different targets?',
# #              fontsize=13, fontweight='bold')

# # # Panel 1: full local — box per auditor, one point per target
# # ax = axes[0]
# # full_errors_per_auditor = []
# # labels = []
# # for aud_id in auditor_ids:
# #     errs = [r['abs_error'] for r in full_results
# #             if r['auditor_id'] == aud_id]
# #     full_errors_per_auditor.append(errs)
# #     labels.append(f'Node {aud_id}\n(n={len(errs)})')

# # bp = ax.boxplot(full_errors_per_auditor, patch_artist=True,
# #                 medianprops=dict(color='black', linewidth=2))
# # for patch, color in zip(bp['boxes'], NODE_COLORS):
# #     patch.set_facecolor(color)
# #     patch.set_alpha(0.7)

# # # Overlay individual points
# # for x_pos, (errs, aud_id) in enumerate(
# #         zip(full_errors_per_auditor, auditor_ids), start=1):
# #     jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(errs))
# #     ax.scatter(x_pos + jitter, errs,
# #                color=NODE_COLORS[aud_id-1], alpha=0.6, s=40, zorder=3)

# # ax.set_xticklabels(labels, fontsize=9)
# # ax.set_ylabel('Absolute Error')
# # ax.set_title('Full Local Audit\nError distribution across targets per auditor')
# # ax.spines[['top','right']].set_visible(False)

# # # Panel 2: budgeted — box per auditor at each budget,
# # # grouped by auditor with budget as hue via offset x positions
# # ax = axes[1]
# # n_auditors = len(auditor_ids)
# # n_budgets  = len(budget_sizes)
# # group_width = 0.8
# # bar_width   = group_width / n_budgets
# # budget_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_budgets))

# # for b_idx, budget in enumerate(budget_sizes):
# #     for a_idx, aud_id in enumerate(auditor_ids):
# #         errs = [r['mean_abs_error'] for r in budget_results
# #                 if r['auditor_id'] == aud_id and r['budget'] == budget]
# #         if not errs:
# #             continue
# #         x = a_idx + (b_idx - n_budgets/2 + 0.5) * bar_width
# #         ax.bar(x, np.mean(errs), bar_width * 0.9,
# #                color=budget_colors[b_idx], edgecolor='white',
# #                linewidth=0.5,
# #                label=f'Budget {budget}' if a_idx == 0 else '')

# # ax.set_xticks(range(n_auditors))
# # ax.set_xticklabels([f'Node {i}' for i in auditor_ids], fontsize=9)
# # ax.set_ylabel('Mean Absolute Error')
# # ax.set_title('Budgeted Audit\nMean error per auditor at each budget\n'
# #              '(darker = larger budget)')
# # ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
# # ax.spines[['top','right']].set_visible(False)

# # plt.tight_layout()
# # out = os.path.join(PLOT_DIR, 'step5b_auditor_consistency.png')
# # plt.savefig(out, dpi=150, bbox_inches='tight')
# # plt.close()
# # log.info(f"  ✓ Saved → {out}")


# # # ─────────────────────────────────────────────────────────────────────────────
# # # Summary
# # # ─────────────────────────────────────────────────────────────────────────────
# # log.info(f"\n{'='*70}")
# # log.info("  Step 5b Complete")
# # log.info(f"{'='*70}")
# # log.info(f"\n  All plots saved to: {PLOT_DIR}")
# # log.info(f"    step5b_budget_labelled.png")
# # log.info(f"    step5b_per_auditee_budget.png")
# # log.info(f"    step5b_confidence_intervals.png")
# # log.info(f"    step5b_bias_variance.png")
# # log.info(f"    step5b_combined_all_modes.png")
# # log.info(f"    step5b_auditor_consistency.png")
# # log.info(f"    logs/step5b.log")
# # log.info(f"{'='*70}")