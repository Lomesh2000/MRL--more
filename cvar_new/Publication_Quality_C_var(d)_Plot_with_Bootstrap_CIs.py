"""
Step 13c: Publication-Quality C_var(d) Plot with Bootstrap CIs
Shows both geometric baseline (||v_{1:d}||^2) and random baseline (d/D)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============ LOAD RESULTS ============
with open('step13b_cvar_results_with_ci.json', 'r') as f:
    data = json.load(f)

# ============ CONFIGURATION ============
COLORS = {
    'mpnet_standard': '#1f77b4',   # blue
    'mpnet_mrl': '#ff7f0e',        # orange
    'nomic_mrl': '#2ca02c',        # green
    'mxbai_mrl': '#d62728',        # red
    'qwen3_0.6b_mrl': '#9467bd',   # purple
    'qwen3_4b_mrl': '#8c564b',     # brown
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

# ============ PLOT 1: C_var with CI ribbons + both baselines ============
fig, ax = plt.subplots(figsize=(12, 8))

for model_key in data.keys():
    model_data = data[model_key]
    dims = np.array(model_data['dimensions'])
    cvar = np.array(model_data['C_var_unnorm_gender'])
    ci_lo = np.array(model_data['bootstrap']['ci_unnorm_lower'])
    ci_hi = np.array(model_data['bootstrap']['ci_unnorm_upper'])
    vnorms = np.array(model_data['v_bias_norm_at_d'])
    full_dim = model_data['full_dim']
    
    color = COLORS[model_key]
    marker = MARKERS[model_key]
    ls = LINESTYLES[model_key]
    label = LABELS[model_key]
    
    # Main curve
    ax.plot(dims, cvar, color=color, marker=marker, linestyle=ls,
            linewidth=2.5, markersize=8, label=label, zorder=3)
    
    # CI ribbon (shaded)
    ax.fill_between(dims, ci_lo, ci_hi, color=color, alpha=0.15, zorder=2)
    
    # Geometric baseline ||v_{1:d}||^2
    geo_baseline = vnorms ** 2
    ax.plot(dims, geo_baseline, color=color, linestyle=':', 
            linewidth=1.5, alpha=0.6, zorder=1)
    
    # Random baseline d/D (only for first model to avoid clutter, or thin gray for all)
    if model_key == 'mpnet_standard':
        rand_baseline = dims / full_dim
        ax.plot(dims, rand_baseline, 'k--', linewidth=1.5, alpha=0.4, 
                label='Random baseline (d/D)', zorder=1)

# Formatting
ax.set_xlabel('Dimension d', fontsize=14, fontweight='bold')
ax.set_ylabel('Bias Concentration C(d)', fontsize=14, fontweight='bold')
ax.set_title('Variance-Based Bias Concentration with 95% Bootstrap CIs\n'
             'Solid = C_var(d), Shaded = CI, Dotted = Geometric baseline ||v₁:d||²',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1100)
ax.set_ylim(-0.05, 1.15)

# Annotation box explaining the three elements
textstr = ('— Solid line: C_var(d) = Var(b^(d))/Var(b^(D))\n'
           '— Shaded ribbon: 95% bootstrap CI (n=1000)\n'
           '— Dotted line: Geometric baseline ||v₁:d||²\n'
           '— Black dashed: Random baseline d/D')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('cd_curves_cvar_with_ci_step13c.png', dpi=300, bbox_inches='tight')
print("✓ Plot 1 saved: cd_curves_cvar_with_ci_step13c.png")
plt.show()

# ============ PLOT 2: Zoomed view (d ≤ 512) for clarity ============
fig, ax = plt.subplots(figsize=(10, 7))

for model_key in data.keys():
    model_data = data[model_key]
    dims = np.array(model_data['dimensions'])
    cvar = np.array(model_data['C_var_unnorm_gender'])
    ci_lo = np.array(model_data['bootstrap']['ci_unnorm_lower'])
    ci_hi = np.array(model_data['bootstrap']['ci_unnorm_upper'])
    vnorms = np.array(model_data['v_bias_norm_at_d'])
    full_dim = model_data['full_dim']
    
    # Only plot up to 512 (or full_dim if smaller)
    mask = dims <= 512
    if not mask.any():
        continue
    
    color = COLORS[model_key]
    marker = MARKERS[model_key]
    ls = LINESTYLES[model_key]
    label = LABELS[model_key]
    
    ax.plot(dims[mask], cvar[mask], color=color, marker=marker, linestyle=ls,
            linewidth=2.5, markersize=9, label=label, zorder=3)
    ax.fill_between(dims[mask], ci_lo[mask], ci_hi[mask], 
                    color=color, alpha=0.2, zorder=2)
    
    # Geometric baseline
    geo_baseline = vnorms ** 2
    ax.plot(dims[mask], geo_baseline[mask], color=color, linestyle=':', 
            linewidth=2, alpha=0.7, zorder=1)

# Reference: random baseline for 768-dim models
ref_dims = np.array([64, 128, 256, 512])
ax.plot(ref_dims, ref_dims/768, 'k--', linewidth=2, alpha=0.5, 
        label='Random (d/768)', zorder=1)

ax.set_xlabel('Dimension d', fontsize=14, fontweight='bold')
ax.set_ylabel('C(d) — Variance Concentration', fontsize=14, fontweight='bold')
ax.set_title('Early-Dimension Bias Concentration (d ≤ 512)\n'
             'C_var(d) vs. Geometric Baseline ||v₁:d||² vs. Random d/D',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(30, 550)
ax.set_ylim(-0.02, 0.45)

# Highlight the "suppression zone" where C_var < geometric
ax.axhspan(0, 0.05, alpha=0.05, color='blue', label='Suppression zone')
ax.text(400, 0.025, '← C_var < ||v₁:d||²\n  (underexpression)', 
        fontsize=9, color='darkblue', style='italic')

plt.tight_layout()
plt.savefig('cd_curves_cvar_zoomed_step13c.png', dpi=300, bbox_inches='tight')
print("✓ Plot 2 saved: cd_curves_cvar_zoomed_step13c.png")
plt.show()

# ============ PLOT 3: Deviation from geometric baseline ============
fig, ax = plt.subplots(figsize=(10, 7))

for model_key in data.keys():
    model_data = data[model_key]
    dims = np.array(model_data['dimensions'])
    cvar = np.array(model_data['C_var_unnorm_gender'])
    vnorms = np.array(model_data['v_bias_norm_at_d'])
    ci_lo = np.array(model_data['bootstrap']['ci_unnorm_lower'])
    ci_hi = np.array(model_data['bootstrap']['ci_unnorm_upper'])
    
    geo = vnorms ** 2
    deviation = cvar - geo  # negative = underexpression
    dev_lo = ci_lo - geo
    dev_hi = ci_hi - geo
    
    color = COLORS[model_key]
    marker = MARKERS[model_key]
    ls = LINESTYLES[model_key]
    label = LABELS[model_key]
    
    ax.plot(dims, deviation, color=color, marker=marker, linestyle=ls,
            linewidth=2.5, markersize=8, label=label, zorder=3)
    ax.fill_between(dims, dev_lo, dev_hi, color=color, alpha=0.15, zorder=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7, 
           label='Isotropic expectation (C_var = ||v||²)', zorder=1)

# Shade underexpression region
ax.axhspan(-0.3, 0, alpha=0.1, color='blue')
ax.text(600, -0.15, 'Underexpression\n(C_var < ||v₁:d||²)', 
        fontsize=11, color='darkblue', ha='center', fontweight='bold')

ax.set_xlabel('Dimension d', fontsize=14, fontweight='bold')
ax.set_ylabel('Deviation: C_var(d) − ||v₁:d||²', fontsize=14, fontweight='bold')
ax.set_title('Statistical Underexpression of Bias in Early Dimensions\n'
             'Negative = MRL suppresses variance below isotropic expectation',
             fontsize=13, fontweight='bold', pad=15)

ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1100)
ax.set_ylim(-0.35, 0.1)

plt.tight_layout()
plt.savefig('cd_curves_deviation_step13c.png', dpi=300, bbox_inches='tight')
print("✓ Plot 3 saved: cd_curves_deviation_step13c.png")
plt.show()

print("\n✓ All 3 plots generated. Key visual evidence:")
print("  1. Full view with CIs and both baselines")
print("  2. Zoomed early-dimension view (d ≤ 512)")
print("  3. Deviation from geometric baseline (the 'underexpression' plot)")