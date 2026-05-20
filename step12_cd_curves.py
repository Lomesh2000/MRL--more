"""
Step 12: Complete C(d) Curve Analysis for All 6 Models
Generates publication-quality plots showing bias concentration across dimensions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ============ LOAD ALL RESULTS ============
print("Loading results from all models...")

# MPNet results (Standard + MRL)
with open('./step7_all_results.json', 'r') as f:
    mpnet_data = json.load(f)

# Nomic results
with open('./nomic/step8_nomic_results.json', 'r') as f:
    nomic_data = json.load(f)

# mxbai results
with open('./mixbai_mrl/step9_mxbai_results.json', 'r') as f:
    mxbai_data = json.load(f)

# Qwen3-0.6B results
with open('./qwen_0.6_emb.py/step10_qwen3_results.json', 'r') as f:
    qwen06_data = json.load(f)

# Qwen3-4B results
with open('./qwen-3b/step11_qwen3_4b_results.json', 'r') as f:
    qwen4b_data = json.load(f)

print("All results loaded.\n")

# ============ EXTRACT C(d) DATA ============
models = {
    'MPNet-Standard': {
        'dims': mpnet_data['dimensions'],
        'C': mpnet_data['standard']['C_gender'],
        'color': '#1f77b4',  # blue
        'marker': 'o',
        'linestyle': '-',
        'full_dim': 768
    },
    'MPNet-MRL': {
        'dims': mpnet_data['dimensions'],
        'C': mpnet_data['mrl']['C_gender'],
        'color': '#ff7f0e',  # orange
        'marker': 's',
        'linestyle': '--',
        'full_dim': 768
    },
    'Nomic-MRL': {
        'dims': nomic_data['dimensions'],
        'C': nomic_data['nomic_mrl']['C_gender'],
        'color': '#2ca02c',  # green
        'marker': '^',
        'linestyle': '--',
        'full_dim': 768
    },
    'mxbai-MRL': {
        'dims': mxbai_data['dimensions'],
        'C': mxbai_data['mxbai_mrl']['C_gender'],
        'color': '#d62728',  # red
        'marker': 'v',
        'linestyle': '--',
        'full_dim': 1024
    },
    'Qwen3-0.6B-MRL': {
        'dims': qwen06_data['dimensions'],
        'C': qwen06_data['qwen3_mrl']['C_gender'],
        'color': '#9467bd',  # purple
        'marker': 'D',
        'linestyle': '--',
        'full_dim': 1024
    },
    'Qwen3-4B-MRL': {
        'dims': qwen4b_data['dimensions'],
        'C': qwen4b_data['qwen3_4b_mrl']['C_gender'],
        'color': '#8c564b',  # brown
        'marker': 'p',
        'linestyle': '-.',
        'full_dim': 2560
    }
}

# ============ PLOT 1: C(d) ABSOLUTE VALUES ============
fig, ax = plt.subplots(figsize=(10, 7))

for name, data in models.items():
    dims = data['dims']
    C_values = data['C']
    ax.plot(dims, C_values, 
            marker=data['marker'], 
            linestyle=data['linestyle'],
            color=data['color'],
            label=name,
            linewidth=2.5,
            markersize=9,
            alpha=0.85)

# Add random baseline for each full dimension
dim_ranges = {
    768: [64, 128, 256, 512, 768],
    1024: [64, 128, 256, 512, 1024],
    2560: [64, 128, 256, 512, 1024, 2560]
}

for full_dim, dims in dim_ranges.items():
    random_baseline = [d / full_dim for d in dims]
    ax.plot(dims, random_baseline, 
            'k:', 
            linewidth=1.5, 
            alpha=0.4,
            label=f'Random (d/{full_dim})' if full_dim == 768 else None)

# Formatting
ax.set_xlabel('Dimension d', fontsize=14, fontweight='bold')
ax.set_ylabel('Bias Concentration C(d)', fontsize=14, fontweight='bold')
ax.set_title('Bias Subspace Concentration Across Prefix Dimensions\n(Gender Attribute)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 2700)
ax.set_ylim(-0.05, 1.1)

# Add annotation
ax.annotate('C(d) = ||v_bias[1:d]||² / ||v_bias||²\nHigher = more bias energy in early dimensions',
            xy=(0.02, 0.02), xycoords='axes fraction',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('cd_curves_absolute_step12.png', dpi=300, bbox_inches='tight')
print("✓ Plot 1 saved: cd_curves_absolute_step12.png")
plt.show()

# ============ PLOT 2: C(d) RATIO VS RANDOM BASELINE ============
fig, ax = plt.subplots(figsize=(10, 7))

for name, data in models.items():
    dims = data['dims']
    C_values = data['C']
    full_dim = data['full_dim']
    
    # Compute ratio C(d) / (d/D)
    ratios = []
    ratio_dims = []
    for i, d in enumerate(dims):
        random_baseline = d / full_dim
        if random_baseline > 0:
            ratios.append(C_values[i] / random_baseline)
            ratio_dims.append(d)
    
    ax.plot(ratio_dims, ratios,
            marker=data['marker'],
            linestyle=data['linestyle'],
            color=data['color'],
            label=name,
            linewidth=2.5,
            markersize=9,
            alpha=0.85)

# Reference line at ratio = 1.0 (random)
ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2, 
           label='Random Baseline (ratio=1.0)', alpha=0.7)

# Formatting
ax.set_xlabel('Dimension d', fontsize=14, fontweight='bold')
ax.set_ylabel('C(d) / (d/D) — Front-Loading Ratio', fontsize=14, fontweight='bold')
ax.set_title('Bias Front-Loading Ratio vs. Random Uniform Distribution\n(>1 = front-loaded, <1 = back-loaded, =1 = uniform)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 2700)

# Add shaded regions
ax.axhspan(1.0, 3.0, alpha=0.1, color='red', label='Front-loaded region')
ax.axhspan(0, 1.0, alpha=0.1, color='blue')

# Annotations
ax.annotate('Front-loaded\n(bias concentrated in early dims)',
            xy=(2000, 2.0), fontsize=11, color='darkred', fontweight='bold',
            ha='center')
ax.annotate('Back-loaded / Uniform\n(bias distributed evenly)',
            xy=(2000, 0.6), fontsize=11, color='darkblue', fontweight='bold',
            ha='center')

plt.tight_layout()
plt.savefig('cd_curves_ratio_step12.png', dpi=300, bbox_inches='tight')
print("✓ Plot 2 saved: cd_curves_ratio_step12.png")
plt.show()

# ============ PLOT 3: NORMALIZED C(d) (C(d) / d/D) ALL ON SAME SCALE ============
fig, ax = plt.subplots(figsize=(10, 7))

for name, data in models.items():
    dims = np.array(data['dims'])
    C_values = np.array(data['C'])
    full_dim = data['full_dim']
    
    # Normalize: x-axis = d/D, y-axis = C(d) / (d/D)
    x_norm = dims / full_dim
    y_norm = C_values / (dims / full_dim)
    
    ax.plot(x_norm, y_norm,
            marker=data['marker'],
            linestyle=data['linestyle'],
            color=data['color'],
            label=name,
            linewidth=2.5,
            markersize=9,
            alpha=0.85)

# Reference line at y = 1.0
ax.axhline(y=1.0, color='black', linestyle='-', linewidth=2, alpha=0.7)

# Formatting
ax.set_xlabel('Normalized Dimension (d/D)', fontsize=14, fontweight='bold')
ax.set_ylabel('C(d) / (d/D) — Front-Loading Ratio', fontsize=14, fontweight='bold')
ax.set_title('Normalized Bias Concentration Across Models\n(Same Scale Comparison)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1.05)

plt.tight_layout()
plt.savefig('cd_curves_normalized_step12.png', dpi=300, bbox_inches='tight')
print("✓ Plot 3 saved: cd_curves_normalized_step12.png")
plt.show()

# ============ PLOT 4: AREA UNDER C(d) CURVE (SUMMARY STATISTIC) ============
fig, ax = plt.subplots(figsize=(10, 6))

# Compute AUC for each model (trapezoidal integration)
model_names = []
auc_values = []
colors = []

for name, data in models.items():
    dims = np.array(data['dims'])
    C_values = np.array(data['C'])
    full_dim = data['full_dim']
    
    # Normalize x to [0, 1]
    x_norm = dims / full_dim
    
    # Trapezoidal integration
    auc = np.trapezoid(C_values, x_norm)
    
    model_names.append(name)
    auc_values.append(auc)
    colors.append(data['color'])
    
    print(f"{name}: AUC = {auc:.4f}")

# Bar plot
bars = ax.bar(model_names, auc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, auc_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Reference line at AUC = 0.5 (random uniform distribution)
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, 
           label='Random Uniform (AUC=0.5)', alpha=0.7)

# Formatting
ax.set_ylabel('Area Under C(d) Curve (AUC)', fontsize=14, fontweight='bold')
ax.set_title('Bias Front-Loading Strength: Area Under C(d) Curve\n(Higher = More Front-Loaded)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_ylim(0, max(auc_values) * 1.15)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Rotate x labels
plt.xticks(rotation=30, ha='right')

plt.tight_layout()
plt.savefig('cd_auc_summary_step12.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot 4 saved: cd_auc_summary_step12.png")
plt.show()

# ============ SUMMARY TABLE ============
print("\n" + "=" * 80)
print("SUMMARY: C(d) Front-Loading Analysis")
print("=" * 80)
print(f"{'Model':<<20} {'C(64)':>8} {'C(128)':>8} {'C(256)':>8} {'C(512)':>8} {'C(full)':>8} {'AUC':>8}")
print("-" * 80)

for name, data in models.items():
    C = data['C']
    dims = data['dims']
    full_dim = data['full_dim']
    
    # Compute AUC
    x_norm = np.array(dims) / full_dim
    auc = np.trapezoid(np.array(C), x_norm)
    
    # Format C values
    c_strs = [f"{c:.3f}" for c in C[:4]]  # First 4
    if len(C) > 4:
        c_strs.append(f"{C[-1]:.3f}")  # Full dimension
    
    print(f"{name:<20} {c_strs[0]:>8} {c_strs[1]:>8} {c_strs[2]:>8} {c_strs[3]:>8} {c_strs[4]:>8} {auc:>8.3f}")

print("-" * 80)
print("\nInterpretation:")
print("  AUC > 0.5: Front-loaded (bias concentrated in early dimensions)")
print("  AUC = 0.5: Uniform (bias distributed evenly)")
print("  AUC < 0.5: Back-loaded (bias concentrated in late dimensions)")
print("\n✓ Step 12 complete. All C(d) curves generated.")