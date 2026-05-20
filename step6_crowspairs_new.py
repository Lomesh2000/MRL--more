"""
Step 6 (Final Fixed): CrowS-Pairs Evaluation
Prints BOTH Standard and MRL results correctly.
"""

import numpy as np
import csv
import urllib.request
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ LOAD MODELS ============
print("Loading models...")
model_std = SentenceTransformer("tomaarsen/mpnet-base-nli")
model_mrl = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka")
print("Models loaded.\n")

DIMS = [64, 128, 256, 512, 768]

def encode_at_dim(model, texts, target_dim, normalize=True):
    if isinstance(texts, str):
        texts = [texts]
    full_emb = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    truncated = full_emb[:, :target_dim]
    if normalize:
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)
        truncated = truncated / norms
    return truncated

def longest_common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]

# ============ DOWNLOAD CROW'S PAIRS CSV ============
CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

print("Downloading CrowS-Pairs dataset...")
urllib.request.urlretrieve(CROWS_URL, "crows_pairs_anonymized.csv")
print("Downloaded successfully.\n")

# ============ LOAD AND PARSE CSV ============
print("Parsing CSV...")
crows_data = []
with open("crows_pairs_anonymized.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sent_more = row.get('sent_more', '').strip()
        sent_less = row.get('sent_less', '').strip()
        direction = row.get('stereo_antistereo', '').strip().lower()
        if sent_more and sent_less and direction in ['stereo', 'antistereo']:
            crows_data.append({
                'sent_more': sent_more,
                'sent_less': sent_less,
                'direction': direction,
                'context': longest_common_prefix(sent_more, sent_less)
            })

print(f"Loaded {len(crows_data)} valid sentence pairs.\n")

# ============ CROW'S PAIRS SCORE ============
def compute_crows_pairs_score(model, dim):
    css = 0
    cas = 0
    total_stereo = 0
    total_antistereo = 0
    
    for example in crows_data:
        context = example['context']
        sent_more = example['sent_more']
        sent_less = example['sent_less']
        direction = example['direction']
        
        if len(context) < 10:
            continue
        
        all_texts = [context, sent_more, sent_less]
        embs = encode_at_dim(model, all_texts, dim, normalize=True)
        
        context_emb = embs[0]
        more_emb = embs[1]
        less_emb = embs[2]
        
        sim_more = float(np.dot(context_emb, more_emb))
        sim_less = float(np.dot(context_emb, less_emb))
        
        agreement = sim_more > sim_less
        
        if direction == 'stereo':
            total_stereo += 1
            if agreement:
                css += 1
        elif direction == 'antistereo':
            total_antistereo += 1
            if agreement:
                cas += 1
    
    total = total_stereo + total_antistereo
    if total == 0:
        return 0, 0, 0, 0, 0, 0
    
    cms = (css + cas) / total * 100
    css_pct = (css / total_stereo * 100) if total_stereo > 0 else 0
    cas_pct = (cas / total_antistereo * 100) if total_antistereo > 0 else 0
    
    return css, cas, cms, css_pct, cas_pct, total

# ============ RUN EVALUATION ============
print("=" * 90)
print("CROW'S PAIRS: CSS, CAS, CMS Across Dimensions")
print("=" * 90)
print(f"{'Dim':>5} {'Model':>10} {'CSS':>8} {'CAS':>8} {'CMS':>10} {'CSS%':>8} {'CAS%':>8} {'N':>8}")
print("-" * 90)

crows_results = {'std': [], 'mrl': []}

for dim in DIMS:
    # Standard model
    css_std, cas_std, cms_std, css_pct_std, cas_pct_std, n_std = compute_crows_pairs_score(model_std, dim)
    print(f"{dim:>5} {'Standard':>10} {css_std:>8} {cas_std:>8} {cms_std:>10.2f} {css_pct_std:>7.1f}% {cas_pct_std:>7.1f}% {n_std:>8}")
    crows_results['std'].append(cms_std)
    
    # MRL model
    css_mrl, cas_mrl, cms_mrl, css_pct_mrl, cas_pct_mrl, n_mrl = compute_crows_pairs_score(model_mrl, dim)
    print(f"{dim:>5} {'MRL':>10} {css_mrl:>8} {cas_mrl:>8} {cms_mrl:>10.2f} {css_pct_mrl:>7.1f}% {cas_pct_mrl:>7.1f}% {n_mrl:>8}")
    crows_results['mrl'].append(cms_mrl)
    
    print("-" * 90)

print("\nInterpretation:")
print("  CMS = 50.0 → Unbiased")
print("  CMS > 50.0 → Stereotype preference")
print("  CMS < 50.0 → Anti-stereotype preference")

# ============ PLOT ============
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(DIMS, crows_results['std'], 'o-', label='Standard', linewidth=2, markersize=8)
ax.plot(DIMS, crows_results['mrl'], 's--', label='MRL', linewidth=2, markersize=8)
ax.axhline(y=50.0, color='black', linestyle=':', label='Unbiased Baseline (50)', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("CrowS Metric Score (CMS)")
ax.set_title("CrowS-Pairs: Stereotype Preference Across Prefix Dimensions")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(40, 70)

plt.tight_layout()
plt.savefig("crowspairs_final_step6.png", dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: crowspairs_final_step6.png")
plt.show()

print("\n✓ Step 6 complete.")