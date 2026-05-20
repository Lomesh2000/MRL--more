"""
Step 13e: Anisotropy Factor α(d) Plots
α(d) = C_var(d) / C_geo(d) — the cleanest metric for the paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Load alpha results
with open('step13d_alpha_results.json', 'r') as f:
    data = json.load(f)

# ============ CONFIGURATION ============
COLORS = {
    'mpnet_standard': '#1f77b4',
    'mpnet_mrl': '#ff7f0e',
    'nomic_mrl': '#2ca02c',
    'mxbai_mrl': '#d62728',
    'qwen3_0.6b_mrl': '#9467bd',
    'qwen3_4b_mrl': '#8c564b',
}

MARKERS = {
    'mpnet_standard': 'o',
    'mpnet_mrl': 's',
    'nomic_mrl': '^',
    'mxbai_mrl': 'v',
    'qwen3_0.6b_mrl': 'D',
    'qwen3_4b_mrl': 'p',
}

LINESTYLES = {
    'mpnet_standard': '-',
    'mpnet_mrl': '--',
    'nomic_mrl': '--',
    'mxbai_mrl': '--',
    'qwen3_0.6b_mrl': '--',
    'qwen3_4b_mrl': '-.',
}

LABELS = {
    'mpnet_standard': 'MPNet-Standard',
    'mpnet_mrl': 'MPNet-MRL',
    'nomic_mrl': 'Nomic-MRL',
    'mxbai_mrl': 'mxbai-MRL',
    'qwen3_0.6b_mrl': 'Qwen3-0.6B-MRL',
    'qwen3_4b_mrl': 'Qwen3-4B-MRL',
}

# ============ PLOT 1: α(d) with CIs — the main figure ============
fig, ax = plt.subplots(figsize=(11, 7))

for model_key in data.keys():
    model_data = data[model_key]
    dims = np.array(model_data['dimensions'])
    alpha = np.array(model_data['alpha'])
    ci_lo = np.array(model_data['ci_lower'])
    ci_hi = np.array(model_data['ci_upper'])
    
    color = COLORS[model_key]
    marker = MARKERS[model_key]
    ls = LINESTYLES[model_key]
    label = LABELS[model_key]
    
    # Main curve
    ax.plot(dims, alpha, color=color, marker=marker, linestyle=ls,
            linewidth=2.8, markersize=9, label=label, zorder=3)
    
    # CI ribbon
    ax.fill_between(dims, ci_lo, ci_hi, color=color, alpha=0.12, zorder=2)

# Isotropic reference line
ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2.5, 
           label='Isotropic (α = 1.0)', zorder=1, alpha=0.8)

# Underexpression zone
ax.axhspan(0, 1.0, alpha=0.06, color='blue', zorder=0)
ax.text(900, 0.55, 'Underexpression\nzone (α < 1)', fontsize=11, 
        color='darkblue', ha='center', style='italic', fontweight='bold')

# Overexpression zone (rare)
ax.axhspan(1.0, 1.3, alpha=0.04, color='red', zorder=0)

ax.set_xlabel('Dimension d', fontsize=15, fontweight='bold')
ax.set_ylabel('Anisotropy Factor α(d)', fontsize=15, fontweight='bold')
ax.set_title('Anisotropy Factor α(d) = C_var(d) / ||v₁:d||²\n'
             'α < 1: covariance suppresses bias variance | α = 1: isotropic | α > 1: amplifies',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1100)
ax.set_ylim(0.15, 1.35)

# Annotation
textstr = ('α(d) measures whether bias variance is expressed\n'
           'isotropically (α=1) or suppressed (α<1) by covariance.\n'
           'Bootstrap CIs: n=1000 resamplings of target words.')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
ax.text(0.98, 0.15, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('alpha_curves_main_step13e.png', dpi=300, bbox_inches='tight')
print("✓ Plot 1 saved: alpha_curves_main_step13e.png")
plt.show()

# ============ PLOT 2: Zoomed early dimensions (d ≤ 512) ============
fig, ax = plt.subplots(figsize=(10, 7))

for model_key in data.keys():
    model_data = data[model_key]
    dims = np.array(model_data['dimensions'])
    alpha = np.array(model_data['alpha'])
    ci_lo = np.array(model_data['ci_lower'])
    ci_hi = np.array(model_data['ci_upper'])
    
    mask = dims <= 512
    if not mask.any():
        continue
    
    color = COLORS[model_key]
    marker = MARKERS[model_key]
    ls = LINESTYLES[model_key]
    label = LABELS[model_key]
    
    ax.plot(dims[mask], alpha[mask], color=color, marker=marker, linestyle=ls,
            linewidth=3, markersize=10, label=label, zorder=3)
    ax.fill_between(dims[mask], ci_lo[mask], ci_hi[mask], 
                    color=color, alpha=0.15, zorder=2)

ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2.5, 
           label='Isotropic (α = 1.0)', zorder=1, alpha=0.8)
ax.axhspan(0, 1.0, alpha=0.08, color='blue', zorder=0)

ax.set_xlabel('Dimension d', fontsize=15, fontweight='bold')
ax.set_ylabel('α(d) — Anisotropy Factor', fontsize=15, fontweight='bold')
ax.set_title('Early-Dimension Anisotropy (d ≤ 512)\n'
             'Where MRL Suppression is Strongest',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(30, 550)
ax.set_ylim(0.25, 1.15)

# Highlight minimum points
ax.annotate('Minimum α\n(MPNet-MRL)', xy=(128, 0.429), xytext=(200, 0.32),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5),
            fontsize=10, color='darkorange', fontweight='bold')

ax.annotate('Minimum α\n(Nomic-MRL)', xy=(128, 0.553), xytext=(250, 0.45),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            fontsize=10, color='darkgreen', fontweight='bold')

ax.annotate('Minimum α\n(Qwen3-0.6B)', xy=(64, 0.358), xytext=(150, 0.28),
            arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
            fontsize=10, color='purple', fontweight='bold')

plt.tight_layout()
plt.savefig('alpha_curves_zoomed_step13e.png', dpi=300, bbox_inches='tight')
print("✓ Plot 2 saved: alpha_curves_zoomed_step13e.png")
plt.show()

# ============ PLOT 3: Summary table as figure ============
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

# Build table data
table_data = []
table_data.append(['Model', 'α(64)', 'α(128)', 'α(256)', 'α(512)', 'Min α', 'At d', 'Verdict'])

for model_key in ['mpnet_standard', 'mpnet_mrl', 'nomic_mrl', 'mxbai_mrl', 
                   'qwen3_0.6b_mrl', 'qwen3_4b_mrl']:
    model_data = data[model_key]
    dims = np.array(model_data['dimensions'])
    alpha = np.array(model_data['alpha'])
    ci_lo = np.array(model_data['ci_lower'])
    ci_hi = np.array(model_data['ci_upper'])
    
    # Find min alpha (excluding full dim)
    non_full = dims < dims[-1]
    min_idx = np.argmin(alpha[non_full])
    min_alpha = alpha[non_full][min_idx]
    min_d = dims[non_full][min_idx]
    
    # Verdict
    all_back = all(ci_hi[i] < 1.0 for i in range(len(dims)-1))
    any_front = any(ci_lo[i] > 1.0 for i in range(len(dims)-1))
    
    if all_back:
        verdict = 'Underexpr.'
    elif any_front:
        verdict = 'Overexpr.'
    else:
        verdict = 'Mixed/Iso.'
    
    row = [
        LABELS[model_key],
        f'{alpha[0]:.3f}',
        f'{alpha[1]:.3f}',
        f'{alpha[2]:.3f}',
        f'{alpha[3]:.3f}',
        f'{min_alpha:.3f}',
        f'{int(min_d)}',
        verdict
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.18, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.14])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Color header
for j in range(len(table_data[0])):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Color rows
for i in range(1, len(table_data)):
    if table_data[i][7] == 'Underexpr.':
        color = '#FFE6E6'
    elif table_data[i][7] == 'Overexpr.':
        color = '#E6F3FF'
    else:
        color = '#FFF2E6'
    for j in range(len(table_data[0])):
        table[(i, j)].set_facecolor(color)

ax.set_title('Anisotropy Factor α(d) Summary Across Models\n'
             'α < 1: covariance suppresses bias variance expression',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('alpha_summary_table_step13e.png', dpi=300, bbox_inches='tight')
print("✓ Plot 3 saved: alpha_summary_table_step13e.png")
plt.show()

print("\n" + "="*70)
print("KEY FINDINGS FROM α(d) ANALYSIS:")
print("="*70)
print("1. MPNet-Standard: α ≈ 1.0 (isotropic) — non-MRL baseline")
print("2. MPNet-MRL: α = 0.53→0.43→0.58→0.73 (V-shape, min at 128d)")
print("3. Nomic-MRL: α = 0.66→0.55→0.61→0.72 (V-shape, min at 128d)")
print("4. mxbai-MRL: α ≈ 1.0 (isotropic) — MRL exception")
print("5. Qwen3-0.6B: α = 0.36→0.45→0.48→0.94 (monotonic recovery)")
print("6. Qwen3-4B: α = 0.85→0.79→0.77→0.82 (flat suppression)")
print("="*70)
print("α(d) = 1.0 is the unambiguous isotropic null.")
print("All MRL models except mxbai show α < 1 with bootstrap CIs excluding 1.0.")
print("This is the strongest evidence for MRL-induced underexpression.")