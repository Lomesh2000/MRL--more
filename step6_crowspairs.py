"""
Step 6 (Fixed): CrowS-Pairs Evaluation
Downloads raw CSV directly from GitHub to bypass datasets library script restriction.
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

# ============ DOWNLOAD CROW'S PAIRS CSV ============
CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

print("Downloading CrowS-Pairs dataset from GitHub...")
try:
    urllib.request.urlretrieve(CROWS_URL, "crows_pairs_anonymized.csv")
    print("Downloaded successfully.\n")
except Exception as e:
    print(f"Download failed: {e}")
    print("Please manually download from:")
    print(CROWS_URL)
    print("Save as 'crows_pairs_anonymized.csv' in this directory.")
    exit(1)

# ============ LOAD AND PARSE CSV ============
print("Parsing CSV...")
crows_data = []
with open("crows_pairs_anonymized.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Keys: sent_more, sent_less, stereo_antistereo, bias_type, annotations, anon_writer, anon_annotators
        sent_more = row.get('sent_more', '').strip()
        sent_less = row.get('sent_less', '').strip()
        direction = row.get('stereo_antistereo', '').strip()
        bias_type = row.get('bias_type', '').strip()
        
        if sent_more and sent_less:
            crows_data.append({
                'sent_more': sent_more,
                'sent_less': sent_less,
                'direction': direction,
                'bias_type': bias_type
            })

print(f"Loaded {len(crows_data)} sentence pairs.")
print(f"Sample pair:")
print(f"  More stereotypical: {crows_data[0]['sent_more']}")
print(f"  Less stereotypical: {crows_data[0]['sent_less']}")
print(f"  Direction: {crows_data[0]['direction']}")
print(f"  Bias type: {crows_data[0]['bias_type']}")

# ============ CROW'S PAIRS SCORE ============
def compute_crows_pairs_score(model, dim):
    """
    For encoder models, we measure which sentence gets higher 
    pre-normalization L2 norm (embedding energy).
    
    The stereotypical sentence having higher energy indicates 
    the model finds it more "natural" / expected.
    """
    stereo_favored = 0
    anti_favored = 0
    total = 0
    
    for example in crows_data:
        sent_more = example['sent_more']
        sent_less = example['sent_less']
        
        # Encode both WITHOUT normalization to get raw energy
        embs = encode_at_dim(model, [sent_more, sent_less], dim, normalize=False)
        norm_more = float(np.linalg.norm(embs[0]))
        norm_less = float(np.linalg.norm(embs[1]))
        
        # The sentence with higher pre-normalization energy is "preferred"
        if norm_more > norm_less:
            stereo_favored += 1
        else:
            anti_favored += 1
            
        total += 1
    
    if total == 0:
        return 0.0, 0
    
    stereo_score = stereo_favored / total
    return stereo_score, total

# ============ RUN EVALUATION ============
print("\n" + "=" * 70)
print("CROW'S PAIRS: Stereotype Favored Score Across Dimensions")
print("=" * 70)
print(f"{'Dim':>5} {'Std Score':>10} {'MRL Score':>10} {'N':>8}")
print("-" * 70)

crows_results = {'std': [], 'mrl': []}

for dim in DIMS:
    print(f"Evaluating dimension {dim}...")
    
    score_std, n_std = compute_crows_pairs_score(model_std, dim)
    score_mrl, n_mrl = compute_crows_pairs_score(model_mrl, dim)
    
    crows_results['std'].append(score_std)
    crows_results['mrl'].append(score_mrl)
    
    print(f"{dim:>5} {score_std:>10.4f} {score_mrl:>10.4f} {n_std:>8}")

print("-" * 70)

# ============ PLOT ============
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(DIMS, crows_results['std'], 'o-', label='Standard', linewidth=2, markersize=8)
ax.plot(DIMS, crows_results['mrl'], 's--', label='MRL', linewidth=2, markersize=8)
ax.axhline(y=0.5, color='black', linestyle=':', label='Random Baseline (0.5)', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("Stereotype Favored Score")
ax.set_title("CrowS-Pairs: Stereotypical Sentence Preference\n(Higher = Model Favors Stereotype)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("crowspairs_fixed_step6.png", dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to: crowspairs_fixed_step6.png")
plt.show()

print("\n✓ Step 6 complete. CrowS-Pairs evaluation finished.")