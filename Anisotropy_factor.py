"""
Step 13d: Compute Anisotropy Factor α(d) from existing C_var results
α(d) = C_var(d) / C_geo(d) = C_var_unnorm / (v_bias_norm_at_d)^2
"""

import json
import numpy as np

# Load existing results
with open('cvar_new/step13b_cvar_results_with_ci.json', 'r') as f:
    data = json.load(f)

print("=" * 80)
print("ANISOTROPY FACTOR α(d) = C_var(d) / ||v_{1:d}||^2")
print("=" * 80)
print("α = 1.0: isotropic (variance matches geometric energy)")
print("α < 1.0: underexpression (covariance suppresses variance)")
print("α > 1.0: overexpression (covariance amplifies variance)")
print("=" * 80)

all_alpha_results = {}

for model_key, model_data in data.items():
    dims = np.array(model_data['dimensions'])
    cvar = np.array(model_data['C_var_unnorm_gender'])
    vnorms = np.array(model_data['v_bias_norm_at_d'])
    
    # Geometric baseline
    c_geo = vnorms ** 2
    
    # Anisotropy factor
    alpha = cvar / c_geo
    
    # Bootstrap CI for alpha: since C_geo is fixed, CI scales directly
    ci_lo = np.array(model_data['bootstrap']['ci_unnorm_lower']) / c_geo
    ci_hi = np.array(model_data['bootstrap']['ci_unnorm_upper']) / c_geo
    
    # Standard error
    se = np.array(model_data['bootstrap']['se_unnorm']) / c_geo
    
    # Store
    all_alpha_results[model_key] = {
        'dimensions': dims.tolist(),
        'alpha': alpha.tolist(),
        'ci_lower': ci_lo.tolist(),
        'ci_upper': ci_hi.tolist(),
        'se': se.tolist(),
        'c_geo': c_geo.tolist(),
        'c_var': cvar.tolist()
    }
    
    print(f"\n{model_key}")
    print(f"{'d':>6} | {'C_var':>8} | {'C_geo':>8} | {'α':>8} | {'CI_lower':>10} | {'CI_upper':>10} | {'Verdict':>10}")
    print("-" * 70)
    
    for i, d in enumerate(dims):
        if ci_hi[i] < 1.0:
            verdict = "BACK"
        elif ci_lo[i] > 1.0:
            verdict = "FRONT"
        else:
            verdict = "UNIFORM"
        
        marker = " <-- FULL" if d == model_data['full_dim'] else ""
        print(f"{d:>6} | {cvar[i]:>8.4f} | {c_geo[i]:>8.4f} | {alpha[i]:>8.4f} | {ci_lo[i]:>10.4f} | {ci_hi[i]:>10.4f} | {verdict:>10}{marker}")

# Save
with open('step13d_alpha_results.json', 'w') as f:
    json.dump(all_alpha_results, f, indent=2)

print(f"\n{'='*80}")
print("SAVED: step13d_alpha_results.json")
print(f"{'='*80}")