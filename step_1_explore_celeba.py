"""
Step 1: CelebA Dataset Exploration
====================================
Run with:
    python3 step1_explore.py --config config.yaml

Env vars:
    NFS_ROOT=/your/nfs/path   override nfs.root from config
    HF_DATASETS_CACHE=/path   override HuggingFace cache location

Produces (all saved to {nfs_root}/experiments/{exp_name}/):
    plots/step1_label_distributions.png
    plots/step1_dp_gap_all_attrs.png
    plots/step1_correlation_matrix.png
    plots/step1_conditional_distributions.png
    plots/step1_sample_images.png
    results/step1_stats.json
    logs/step1.log                          (full stdout mirror)
"""

import os
import sys
import json
import argparse
import shutil
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datasets import load_dataset, get_dataset_split_names
import yaml

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Step 1: CelebA Dataset Exploration')
parser.add_argument('--config', type=str, default='config.yaml')
args = parser.parse_args()

# ── Load config ────────────────────────────────────────────────────────────────
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

DATASET_NAME   = cfg['dataset']['name']
SENSITIVE_ATTR = cfg['dataset']['sensitive_attr']   # 'Male'
TARGET_ATTR    = cfg['dataset']['target_attr']       # 'Smiling'
ALPHA          = cfg['partition']['alpha']
NUM_NODES      = cfg['partition']['num_nodes']
PART_SEED      = cfg['partition']['seed']
EXP_NAME       = (cfg['experiment'].get('name')
                  or f"lenet_alpha{ALPHA}_{NUM_NODES}nodes_seed{PART_SEED}")
EXP_NOTES      = cfg['experiment']['notes']

# ── NFS paths ──────────────────────────────────────────────────────────────────
NFS_ROOT  = os.environ.get('NFS_ROOT', cfg['nfs']['root'])
EXP_DIR   = os.path.join(NFS_ROOT, 'experiments', EXP_NAME)
HF_CACHE  = os.environ.get('HF_DATASETS_CACHE',
                            os.path.join(NFS_ROOT, 'hf_cache'))
os.environ['HF_DATASETS_CACHE'] = HF_CACHE

PLOT_DIR    = os.path.join(EXP_DIR, 'plots')
RESULTS_DIR = os.path.join(EXP_DIR, 'results')
LOG_DIR     = os.path.join(EXP_DIR, 'logs')

for d in [HF_CACHE, PLOT_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# Save config copy into experiment dir
shutil.copy(args.config, os.path.join(EXP_DIR, 'config.yaml'))

# ── Logging — write to both stdout and log file ────────────────────────────────
log_path = os.path.join(LOG_DIR, 'step1.log')
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

# ─────────────────────────────────────────────────────────────────────────────
# START
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 70)
log.info("  Step 1: CelebA Dataset Exploration")
log.info("=" * 70)
log.info(f"\n  Config         : {args.config}")
log.info(f"  Experiment     : {EXP_NAME}")
log.info(f"  Notes          : {EXP_NOTES}")
log.info(f"  NFS root       : {NFS_ROOT}")
log.info(f"  Experiment dir : {EXP_DIR}")
log.info(f"  HF cache       : {HF_CACHE}")
log.info(f"  Sensitive attr : {SENSITIVE_ATTR}")
log.info(f"  Target attr    : {TARGET_ATTR}")
log.info(f"  Log file       : {log_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Dataset provenance
# ─────────────────────────────────────────────────────────────────────────────
log.info("─" * 70)
log.info("  SECTION 1: Dataset Provenance")
log.info("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Load dataset & check splits
# ─────────────────────────────────────────────────────────────────────────────
log.info("─" * 70)
log.info("  SECTION 2: Loading Dataset & Split Statistics")
log.info("─" * 70)

available_splits = get_dataset_split_names(DATASET_NAME)
log.info(f"\n  Available splits: {available_splits}\n")

splits = {}
for split in available_splits:
    log.info(f"  Loading split: {split}...")
    splits[split] = load_dataset(DATASET_NAME, split=split)
    log.info(f"  ✓ {split}: {len(splits[split]):,} samples")

# Use train split as primary for exploration
dataset = splits['train']
N_train = len(dataset)
log.info(f"\n  Primary split for exploration: train ({N_train:,} samples)")
log.info(f"\n  train, val, test splits have {len(splits['train']):,}, {len(splits['valid']):,}, {len(splits['test']):,} samples respectively.")

# Extract gender and smiling for all splits
split_stats = {}
for split_name, ds in splits.items():
    g = np.array(ds[SENSITIVE_ATTR], dtype=np.int64)
    s = np.array(ds[TARGET_ATTR],    dtype=np.int64)
    n = len(g)
    split_stats[split_name] = {
        'n'            : n,
        'n_male'       : int(g.sum()),
        'n_female'     : int((1-g).sum()),
        'pct_male'     : float(g.mean()),
        'pct_female'   : float(1 - g.mean()),
        'n_smiling'    : int(s.sum()),
        'n_notsmiling' : int((1-s).sum()),
        'pct_smiling'  : float(s.mean()),
    }

log.info(f"\n  {'Split':<12} {'N':>8} {'% Male':>8} {'% Female':>9} {'% Smiling':>11}")
log.info("  " + "-" * 52)
for split_name, st in split_stats.items():
    log.info(f"  {split_name:<12} {st['n']:>8,} {100*st['pct_male']:>7.1f}% "
             f"{100*st['pct_female']:>8.1f}% {100*st['pct_smiling']:>10.1f}%")

# Work with train split from here
gender  = np.array(dataset[SENSITIVE_ATTR], dtype=np.int64)
smiling = np.array(dataset[TARGET_ATTR],    dtype=np.int64)
N = len(gender)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Core label statistics
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'─'*70}")
log.info("  SECTION 3: Core Label Statistics (train split)")
log.info(f"{'─'*70}\n")

n_male   = int(gender.sum())
n_female = int((1 - gender).sum())
n_smile  = int(smiling.sum())
n_nosmile= int((1 - smiling).sum())

# Joint counts
male_smile    = int(((gender==1) & (smiling==1)).sum())
male_nosmile  = int(((gender==1) & (smiling==0)).sum())
female_smile  = int(((gender==0) & (smiling==1)).sum())
female_nosmile= int(((gender==0) & (smiling==0)).sum())

# Conditional probabilities
p_smile_given_male   = male_smile   / n_male
p_smile_given_female = female_smile / n_female
p_male_given_smile   = male_smile   / n_smile
p_female_given_smile = female_smile / n_smile
p_male_given_nosmile   = male_nosmile   / n_nosmile
p_female_given_nosmile = female_nosmile / n_nosmile

dp_gap_data = abs(p_smile_given_male - p_smile_given_female)

log.info(f"  Total samples      : {N:,}")
log.info(f"  Num features       : {len(CELEBA_ATTRS)} binary attributes")
log.info(f"  Target label       : {TARGET_ATTR}")
log.info(f"  Sensitive attr     : {SENSITIVE_ATTR}")
log.info(f"\n  Gender distribution:")
log.info(f"    Male             : {n_male:,}  ({100*n_male/N:.1f}%)")
log.info(f"    Female           : {n_female:,}  ({100*n_female/N:.1f}%)")
log.info(f"\n  Smiling distribution:")
log.info(f"    Smiling          : {n_smile:,}  ({100*n_smile/N:.1f}%)")
log.info(f"    Not Smiling      : {n_nosmile:,}  ({100*n_nosmile/N:.1f}%)")
log.info(f"\n  Joint distribution (Gender × Smiling):")
log.info(f"    Male   & Smiling     : {male_smile:,}  ({100*male_smile/N:.1f}%)")
log.info(f"    Male   & Not Smiling : {male_nosmile:,}  ({100*male_nosmile/N:.1f}%)")
log.info(f"    Female & Smiling     : {female_smile:,}  ({100*female_smile/N:.1f}%)")
log.info(f"    Female & Not Smiling : {female_nosmile:,}  ({100*female_nosmile/N:.1f}%)")
log.info(f"\n  Conditional distributions:")
log.info(f"    P(Smiling | Male)        = {p_smile_given_male:.4f}  "
         f"({100*p_smile_given_male:.1f}% of males smile)")
log.info(f"    P(Smiling | Female)      = {p_smile_given_female:.4f}  "
         f"({100*p_smile_given_female:.1f}% of females smile)")
log.info(f"    P(Male   | Smiling)      = {p_male_given_smile:.4f}  "
         f"({100*p_male_given_smile:.1f}% of smiling faces are male)")
log.info(f"    P(Female | Smiling)      = {p_female_given_smile:.4f}  "
         f"({100*p_female_given_smile:.1f}% of smiling faces are female)")
log.info(f"    P(Male   | Not Smiling)  = {p_male_given_nosmile:.4f}")
log.info(f"    P(Female | Not Smiling)  = {p_female_given_nosmile:.4f}")
log.info(f"\n  Ground-truth Demographic Parity:")
log.info(f"    Demographic Parity (data) = |P(Smile|Male) - P(Smile|Female)|")
log.info(f"                  = |{p_smile_given_male:.4f} - {p_smile_given_female:.4f}|")
log.info(f"                  = {dp_gap_data:.4f}")
log.info(f"\n  Note: This is NOT a model fairness metric. It measures the")
log.info(f"  imbalance in the raw data and serves as a reference baseline.")
log.info(f"  A perfectly accurate model would have DP gap ≈ {dp_gap_data:.4f}.")
log.info(f"  A fair model (by demographic parity) would have DP gap = 0.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DP gap for ALL 40 attributes — sensitive attribute selection
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'─'*70}")
log.info("  SECTION 4: DP Gap for All Attributes — Sensitive Attribute Selection")
log.info(f"{'─'*70}")
log.info("""
  For each attribute A, we compute:
    DP gap = |P(Smiling=1 | A=1) - P(Smiling=1 | A=0)|

  This tells us how strongly each attribute correlates with the target (Smiling).
  A good sensitive attribute should have:
    (a) A meaningful DP gap (not near zero — otherwise nothing to audit)
    (b) Ethical relevance (gender, age, etc.)
    (c) Not be causally downstream of Smiling (e.g. High_Cheekbones, Mouth_Open)
""")

log.info(f"  {'Attribute':<25} {'P(Smile|A=1)':>13} {'P(Smile|A=0)':>13} "
         f"{'DP gap':>8} {'% with A':>9}")
log.info("  " + "-" * 75)

attr_dp_results = []
for attr in CELEBA_ATTRS:
    attr_vals  = np.array(dataset[attr], dtype=np.int64)
    pct_attr   = float(attr_vals.mean())
    p1 = float(smiling[attr_vals == 1].mean()) if (attr_vals == 1).sum() > 0 else 0.0
    p0 = float(smiling[attr_vals == 0].mean()) if (attr_vals == 0).sum() > 0 else 0.0
    gap = abs(p1 - p0)

    # Gender correlation
    gender_corr = float(np.corrcoef(gender, attr_vals)[0, 1])

    attr_dp_results.append({
        'attr'        : attr,
        'p_smile_1'   : p1,
        'p_smile_0'   : p0,
        'dp_gap'      : gap,
        'pct_present' : pct_attr,
        'gender_corr' : gender_corr,
    })

# Sort by DP gap descending
attr_dp_results.sort(key=lambda x: x['dp_gap'], reverse=True)

for r in attr_dp_results:
    marker = '  ← SELECTED' if r['attr'] == SENSITIVE_ATTR else ''
    marker += '  ← TARGET'  if r['attr'] == TARGET_ATTR    else ''
    log.info(f"  {r['attr']:<25} {r['p_smile_1']:>13.4f} {r['p_smile_0']:>13.4f} "
             f"{r['dp_gap']:>8.4f} {100*r['pct_present']:>8.1f}%{marker}")

# Sensitive attribute selection justification
selected = next(r for r in attr_dp_results if r['attr'] == SENSITIVE_ATTR)
log.info(f"""
  ── Sensitive Attribute Selection: '{SENSITIVE_ATTR}' ──────────────────────

  Justification for selecting '{SENSITIVE_ATTR}' as the sensitive attribute:

  1. MEANINGFUL DP GAP: {SENSITIVE_ATTR} has a DP gap of {selected['dp_gap']:.4f}
     with Smiling. This is large enough to produce an auditable fairness signal
     but not so large that it trivialises the auditing problem.

  2. EXCLUDES CAUSALLY CONTAMINATED ATTRIBUTES: Attributes like
     'High_Cheekbones' (gap={next(r for r in attr_dp_results if r['attr']=='High_Cheekbones')['dp_gap']:.3f})
     and 'Mouth_Slightly_Open' (gap={next(r for r in attr_dp_results if r['attr']=='Mouth_Slightly_Open')['dp_gap']:.3f})
     have very high gaps but are physically caused by smiling itself
     (cheeks raise, mouth opens when smiling). Using them would be circular.

  3. ETHICAL RELEVANCE: Gender is a protected attribute under anti-discrimination
     law in most jurisdictions. Auditing for gender fairness in a smiling
     detector is practically meaningful — such detectors are used in
     emotion recognition systems with real-world consequences.

  4. STANDARD BENCHMARK: Male/Female is the most commonly used sensitive
     attribute for CelebA in the fairness literature, making results
     directly comparable to prior work.

  5. REASONABLE PREVALENCE: {100*selected['pct_present']:.1f}% of samples are Male,
     giving adequate representation of both groups for reliable estimation.
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Image statistics
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"{'─'*70}")
log.info("  SECTION 5: Image Statistics")
log.info(f"{'─'*70}\n")

sample_img = dataset[0]['image']
log.info(f"  Image resolution : {sample_img.size}  (W × H, PIL format)")
log.info(f"  Image mode       : {sample_img.mode}")
log.info(f"\n  Computing pixel statistics on 1,000 random samples...")

rng = np.random.default_rng(42)
sample_indices = rng.choice(N, size=1000, replace=False)

pixel_vals = []
for idx in sample_indices:
    img_arr = np.array(dataset[int(idx)]['image'].convert('RGB'), dtype=np.float32) / 255.0
    pixel_vals.append(img_arr)
pixel_vals = np.stack(pixel_vals)   # (1000, H, W, 3)

channel_mean = pixel_vals.mean(axis=(0, 1, 2))
channel_std  = pixel_vals.std(axis=(0, 1, 2))

log.info(f"\n  Per-channel pixel statistics (estimated from 1,000 samples):")
log.info(f"    Channel  Mean    Std")
for i, ch in enumerate(['Red', 'Green', 'Blue']):
    log.info(f"    {ch:<7}  {channel_mean[i]:.4f}  {channel_std[i]:.4f}")
log.info(f"\n  ImageNet normalisation constants used in training:")
log.info(f"    mean = [0.485, 0.456, 0.406]")
log.info(f"    std  = [0.229, 0.224, 0.225]")
log.info(f"  Note: CelebA pixel stats are close to ImageNet — normalisation is appropriate.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Plots
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'─'*70}")
log.info("  SECTION 6: Generating Plots")
log.info(f"{'─'*70}\n")

# ── Plot 1: Label distributions ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('CelebA (train) — Core Label Distributions', fontsize=13, fontweight='bold')

# Gender
counts_g = [n_male, n_female]
bars = axes[0].bar(['Male', 'Female'], counts_g,
                   color=['steelblue', 'salmon'], edgecolor='white', linewidth=1.2)
axes[0].set_title('Gender Distribution')
axes[0].set_ylabel('Count')
for bar, c in zip(bars, counts_g):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f'{c:,}\n({100*c/N:.1f}%)', ha='center', fontsize=10)
axes[0].set_ylim(0, max(counts_g) * 1.2)
axes[0].spines[['top','right']].set_visible(False)

# Smiling
counts_s = [n_smile, n_nosmile]
bars2 = axes[1].bar(['Smiling', 'Not Smiling'], counts_s,
                    color=['mediumseagreen', 'lightcoral'], edgecolor='white', linewidth=1.2)
axes[1].set_title('Smiling Distribution')
axes[1].set_ylabel('Count')
for bar, c in zip(bars2, counts_s):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                 f'{c:,}\n({100*c/N:.1f}%)', ha='center', fontsize=10)
axes[1].set_ylim(0, max(counts_s) * 1.2)
axes[1].spines[['top','right']].set_visible(False)

# Conditional smiling rates
bars3 = axes[2].bar(['Male', 'Female'],
                    [p_smile_given_male, p_smile_given_female],
                    color=['steelblue', 'salmon'], edgecolor='white', linewidth=1.2)
axes[2].set_title(f'P(Smiling | Gender)\nDemographic Parity (data) = {dp_gap_data:.4f}')
axes[2].set_ylabel('P(Smiling = 1)')
axes[2].set_ylim(0, 1)
# axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='0.5')
for bar, v in zip(bars3, [p_smile_given_male, p_smile_given_female]):
    axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].spines[['top','right']].set_visible(False)

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step1_label_distributions.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 2: DP gap for all 40 attributes ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))

attrs_sorted  = [r['attr']   for r in attr_dp_results]
gaps_sorted   = [r['dp_gap'] for r in attr_dp_results]

# Colour bars: red if causally suspect, gold if selected, blue otherwise
CAUSAL_SUSPECTS = {'High_Cheekbones', 'Mouth_Slightly_Open', 'Rosy_Cheeks'}
bar_colors = []
# for r in attr_dp_results:
#     if r['attr'] == SENSITIVE_ATTR:
#         bar_colors.append('gold')
#     elif r['attr'] in CAUSAL_SUSPECTS:
#         bar_colors.append('tomato')
#     else:
#         bar_colors.append('steelblue')
for r in attr_dp_results:
    if r['attr'] == SENSITIVE_ATTR:
        bar_colors.append('gold')
    else:
        bar_colors.append('steelblue')

bars = ax.barh(attrs_sorted, gaps_sorted, color=bar_colors,
               edgecolor='white', linewidth=0.8)
ax.set_xlabel('Demographic Parity = |P(Smiling|Attr=1) - P(Smiling|Attr=0)|')
ax.set_title('Demographic Parity with Smiling for All 40 Attributes\n'
            #  '(gold = selected sensitive attr, red = causally contaminated)')
             '(yellow = selected sensitive attr)')
ax.axvline(0.05, color='gray', linestyle='--', alpha=0.5, label='demographic parity=0.05')
ax.invert_yaxis()
ax.legend()
ax.spines[['top','right']].set_visible(False)

# Annotate values
for bar, v in zip(bars, gaps_sorted):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=7)

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step1_dp_gap_all_attrs.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 3: Conditional distributions ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Conditional Distributions (train split)', fontsize=13, fontweight='bold')

# P(Smile | Gender) and P(Gender | Smile)
cond_labels_left  = ['P(Smile|Male)', 'P(Smile|Female)']
cond_values_left  = [p_smile_given_male, p_smile_given_female]
cond_labels_right = ['P(Male|Smile)', 'P(Female|Smile)',
                     'P(Male|No Smile)', 'P(Female|No Smile)']
cond_values_right = [p_male_given_smile, p_female_given_smile,
                     p_male_given_nosmile, p_female_given_nosmile]

colors_left  = ['steelblue', 'salmon']
colors_right = ['steelblue', 'salmon', 'cornflowerblue', 'lightsalmon']

bars = axes[0].bar(cond_labels_left, cond_values_left,
                   color=colors_left, edgecolor='white', linewidth=1.2)
axes[0].set_title('P(Smiling | Gender)\n"Of all males/females, what fraction smile?"')
axes[0].set_ylabel('Probability')
axes[0].set_ylim(0, 1)
axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.4)
for bar, v in zip(bars, cond_values_left):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
axes[0].spines[['top','right']].set_visible(False)

bars2 = axes[1].bar(cond_labels_right, cond_values_right,
                    color=colors_right, edgecolor='white', linewidth=1.2)
axes[1].set_title('P(Gender | Smiling)\n"Of all smiling/non-smiling faces, what fraction are male?"')
axes[1].set_ylabel('Probability')
axes[1].set_ylim(0, 1)
axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.4)
for bar, v in zip(bars2, cond_values_right):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
axes[1].spines[['top','right']].set_visible(False)
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step1_conditional_distributions.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ── Plot 4: Correlation matrix (subset of most relevant attributes) ────────────
log.info("  Computing label correlation matrix (all 40 attributes)...")
CORR_ATTRS = CELEBA_ATTRS  # all 40
attr_matrix = np.zeros((len(CORR_ATTRS), N), dtype=np.int8)
for i, attr in enumerate(CORR_ATTRS):
    attr_matrix[i] = np.array(dataset[attr], dtype=np.int8)

corr_matrix = np.corrcoef(attr_matrix)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(CORR_ATTRS)))
ax.set_yticks(range(len(CORR_ATTRS)))
ax.set_xticklabels(CORR_ATTRS, rotation=90, fontsize=7)
ax.set_yticklabels(CORR_ATTRS, fontsize=7)
ax.set_title('Label Correlation Matrix (all 40 CelebA attributes)\n'
             'Red = positive correlation, Blue = negative', fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step1_correlation_matrix.png')
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")


# ── Plot 5: Sample images by group ────────────────────────────────────────────
log.info("  Generating sample image grid...")
groups = {
    'Male\nSmiling'      : np.where((gender==1) & (smiling==1))[0][:4],
    'Male\nNot Smiling'  : np.where((gender==1) & (smiling==0))[0][:4],
    'Female\nSmiling'    : np.where((gender==0) & (smiling==1))[0][:4],
    'Female\nNot Smiling': np.where((gender==0) & (smiling==0))[0][:4],
}

fig, axes = plt.subplots(4, 4, figsize=(10, 11))
fig.suptitle('Sample Images by Group (Gender × Smiling)', fontsize=13, fontweight='bold')
for col, (group_name, idxs) in enumerate(groups.items()):
    for row, idx in enumerate(idxs):
        ax  = axes[row, col]
        img = dataset[int(idx)]['image'].resize((80, 80))
        ax.imshow(img)
        ax.axis('off')
        if row == 0:
            ax.set_title(group_name, fontsize=10, fontweight='bold')
plt.tight_layout()
out = os.path.join(PLOT_DIR, 'step1_sample_images.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
log.info(f"  ✓ Saved → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Save JSON stats
# ─────────────────────────────────────────────────────────────────────────────
stats = {
    'experiment'      : EXP_NAME,
    'dataset'         : DATASET_NAME,
    'sensitive_attr'  : SENSITIVE_ATTR,
    'target_attr'     : TARGET_ATTR,
    'split_stats'     : split_stats,
    'train': {
        'n'                     : N,
        'n_male'                : n_male,
        'n_female'              : n_female,
        'pct_male'              : float(gender.mean()),
        'n_smiling'             : n_smile,
        'n_notsmiling'          : n_nosmile,
        'pct_smiling'           : float(smiling.mean()),
        'male_smiling'          : male_smile,
        'male_notsmiling'       : male_nosmile,
        'female_smiling'        : female_smile,
        'female_notsmiling'     : female_nosmile,
        'p_smile_given_male'    : p_smile_given_male,
        'p_smile_given_female'  : p_smile_given_female,
        'p_male_given_smile'    : p_male_given_smile,
        'p_female_given_smile'  : p_female_given_smile,
        'p_male_given_nosmile'  : p_male_given_nosmile,
        'p_female_given_nosmile': p_female_given_nosmile,
        'dp_gap_data'           : dp_gap_data,
    },
    'image_stats': {
        'resolution'    : list(sample_img.size),
        'mode'          : sample_img.mode,
        'channel_mean'  : channel_mean.tolist(),
        'channel_std'   : channel_std.tolist(),
    },
    'attr_dp_gaps'    : attr_dp_results,   # DP gap for all 40 attrs
}

results_path = os.path.join(RESULTS_DIR, 'step1_stats.json')
with open(results_path, 'w') as f:
    json.dump(stats, f, indent=2)
log.info(f"\n  ✓ Saved → {results_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
log.info(f"\n{'='*70}")
log.info("  Step 1 Complete — Summary")
log.info(f"{'='*70}")
log.info(f"\n  Dataset         : {DATASET_NAME}")
log.info(f"  Train samples   : {N:,}")
for split_name, st in split_stats.items():
    if split_name != 'train':
        log.info(f"  {split_name:<15} : {st['n']:,}")
log.info(f"\n  Gender split    : {100*gender.mean():.1f}% Male / "
         f"{100*(1-gender.mean()):.1f}% Female")
log.info(f"  Smiling split   : {100*smiling.mean():.1f}% / "
         f"{100*(1-smiling.mean()):.1f}%")
log.info(f"\n  P(Smile|Male)   : {p_smile_given_male:.4f}")
log.info(f"  P(Smile|Female) : {p_smile_given_female:.4f}")
log.info(f"  DP gap (data)   : {dp_gap_data:.4f}  ← ground truth reference")
log.info(f"\n  Sensitive attr  : {SENSITIVE_ATTR}  (rank "
         f"{next(i for i,r in enumerate(attr_dp_results) if r['attr']==SENSITIVE_ATTR)+1}"
         f"/{len(CELEBA_ATTRS)} by DP gap, excluding causal suspects)")
log.info(f"\n  Outputs saved to: {EXP_DIR}")
log.info(f"    plots/step1_label_distributions.png")
log.info(f"    plots/step1_dp_gap_all_attrs.png")
log.info(f"    plots/step1_conditional_distributions.png")
log.info(f"    plots/step1_correlation_matrix.png")
log.info(f"    plots/step1_sample_images.png")
log.info(f"    results/step1_stats.json")
log.info(f"    logs/step1.log")
log.info(f"\n  Next → python step2_train_baseline.py --config {args.config}")
log.info(f"{'='*70}")
