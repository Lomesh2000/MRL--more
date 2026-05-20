"""
Step 8: Generalization Test — Nomic Embed Text v1.5
Production MRL model, different architecture, different training data.
Tests whether the decoupling pattern generalizes beyond the MPNet ablation.
"""

import numpy as np
import json
import csv
import urllib.request
from scipy import stats
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============ LOAD NOMIC MRL MODEL ============
print("Loading Nomic Embed Text v1.5...")
model_nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
dim_nomic = model_nomic.get_embedding_dimension()
print(f"Nomic model loaded. Dimension: {dim_nomic}")

# Nomic uses different dimensions: 64, 128, 256, 512, 768
# We test the same 5 dimensions for comparability
DIMS = [64, 128, 256, 512, 768]
FULL_DIM = 768

# ============ ENCODING UTILITY ============
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

# ============ 1. WEAT COMPUTATION ============
print("\n" + "=" * 70)
print("1. COMPUTING WEAT EFFECT SIZES — NOMIC")
print("=" * 70)

MALE_WORDS = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
FEMALE_WORDS = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
CAREER_WORDS = ["executive", "management", "professional", "corporation", "salary", 
                "office", "business", "career", "manager", "administrator"]
FAMILY_WORDS = ["home", "parents", "children", "family", "cousins", "marriage", 
                "wedding", "relatives", "household", "domestic"]
MATH_WORDS = ["math", "algebra", "geometry", "calculus", "equations", "computation", 
              "numbers", "addition", "statistics", "measurement"]
ARTS_WORDS = ["poetry", "art", "dance", "literature", "novel", "symphony", 
              "drama", "sculpture", "painting", "music"]
PLEASANT_WORDS = ["caress", "freedom", "health", "love", "peace", "cheer", "friend", 
                  "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", 
                  "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family", 
                  "happy", "laughter", "paradise", "vacation"]
UNPLEASANT_WORDS = ["abuse", "crash", "filth", "murder", "sickness", "accident", "death", 
                    "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", 
                    "tragedy", "divorce", "jail", "poverty", "ugly", "cancer", "kill", 
                    "rotten", "vomit", "agony", "prison"]

def compute_weat_es(target_X, target_Y, attr_A, attr_B, model, dim):
    X_emb = encode_at_dim(model, target_X, dim)
    Y_emb = encode_at_dim(model, target_Y, dim)
    A_emb = encode_at_dim(model, attr_A, dim)
    B_emb = encode_at_dim(model, attr_B, dim)
    
    s_X = np.array([np.mean(np.dot(x, A_emb.T)) - np.mean(np.dot(x, B_emb.T)) for x in X_emb])
    s_Y = np.array([np.mean(np.dot(y, A_emb.T)) - np.mean(np.dot(y, B_emb.T)) for y in Y_emb])
    
    mean_diff = np.mean(s_X) - np.mean(s_Y)
    pooled_std = np.std(np.concatenate([s_X, s_Y]), ddof=1)
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std

weat_tests = [
    ("Gender-Career", MALE_WORDS, FEMALE_WORDS, CAREER_WORDS, FAMILY_WORDS),
    ("Gender-Math", MALE_WORDS, FEMALE_WORDS, MATH_WORDS, ARTS_WORDS),
    ("Gender-Pleasant", MALE_WORDS, FEMALE_WORDS, PLEASANT_WORDS, UNPLEASANT_WORDS),
]

weat_nomic = []

for dim in DIMS:
    es_list = []
    for name, X, Y, A, B in weat_tests:
        es = compute_weat_es(X, Y, A, B, model_nomic, dim)
        es_list.append(abs(es))
    weat_nomic.append(np.mean(es_list))
    print(f"Dim {dim}: Nomic={weat_nomic[-1]:.4f}")

# ============ 2. C(d) COMPUTATION ============
print("\n" + "=" * 70)
print("2. COMPUTING BIAS CONCENTRATION C(d) — NOMIC")
print('"=" * 70')

def compute_C_d(model, group_A, group_B, dims=DIMS):
    A_full = encode_at_dim(model, group_A, FULL_DIM, normalize=True)
    B_full = encode_at_dim(model, group_B, FULL_DIM, normalize=True)
    diffs = A_full - B_full
    mean_diff = np.mean(diffs, axis=0)
    v_bias = mean_diff / (np.linalg.norm(mean_diff) + 1e-12)
    full_norm_sq = np.sum(v_bias ** 2)
    
    concentrations = {}
    for d in dims:
        v_prefix = v_bias[:d]
        prefix_norm_sq = np.sum(v_prefix ** 2)
        concentrations[d] = prefix_norm_sq / full_norm_sq
    return concentrations

C_gender_nomic_dict = compute_C_d(model_nomic, MALE_WORDS, FEMALE_WORDS)
C_gender_nomic = [C_gender_nomic_dict[d] for d in DIMS]

print("Gender C(d) Nomic:", [f"{x:.4f}" for x in C_gender_nomic])

# ============ 3. STEREOSET COMPUTATION ============
print("\n" + "=" * 70)
print("3. COMPUTING STEREOSET SCORES — NOMIC")
print("=" * 70)

print("Loading StereoSet...")
dataset = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
print(f"Loaded {len(dataset)} examples.")

def compute_stereoset(model, dim):
    stereo_count = 0
    total = 0
    for ex in dataset:
        sentences_dict = ex.get('sentences', {})
        sentence_list = sentences_dict.get('sentence', [])
        gold_labels = sentences_dict.get('gold_label', [])
        context = ex.get('context', '')
        
        if len(sentence_list) < 2 or len(gold_labels) != len(sentence_list):
            continue
        
        sentence_list = [str(s) for s in sentence_list if s is not None]
        if len(sentence_list) < 2:
            continue
        
        all_texts = [context] + sentence_list
        embs = encode_at_dim(model, all_texts, dim, normalize=True)
        similarities = np.dot(embs[1:], embs[0])
        best_idx = int(np.argmax(similarities))
        
        if best_idx < len(gold_labels) and int(gold_labels[best_idx]) == 0:
            stereo_count += 1
        total += 1
    
    return stereo_count / total if total > 0 else 0.0

stereoset_nomic = []
for dim in DIMS:
    print(f"  StereoSet dim {dim}...")
    stereoset_nomic.append(compute_stereoset(model_nomic, dim))

print("StereoSet Nomic:", [f"{x:.4f}" for x in stereoset_nomic])

# ============ 4. CROW'S PAIRS COMPUTATION ============
print("\n" + "=" * 70)
print("4. COMPUTING CROW'S PAIRS CMS — NOMIC")
print("=" * 70)

CROWS_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
urllib.request.urlretrieve(CROWS_URL, "crows_pairs_anonymized.csv")

def longest_common_prefix(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return s1[:i]
    return s1[:min_len]

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

def compute_crows_pairs(model, dim):
    css = 0
    cas = 0
    total_stereo = 0
    total_antistereo = 0
    
    for ex in crows_data:
        if len(ex['context']) < 10:
            continue
        all_texts = [ex['context'], ex['sent_more'], ex['sent_less']]
        embs = encode_at_dim(model, all_texts, dim, normalize=True)
        sim_more = float(np.dot(embs[0], embs[1]))
        sim_less = float(np.dot(embs[0], embs[2]))
        agreement = sim_more > sim_less
        
        if ex['direction'] == 'stereo':
            total_stereo += 1
            if agreement:
                css += 1
        else:
            total_antistereo += 1
            if agreement:
                cas += 1
    
    total = total_stereo + total_antistereo
    if total == 0:
        return 50.0
    return (css + cas) / total * 100

crows_nomic = []
for dim in DIMS:
    print(f"  CrowS-Pairs dim {dim}...")
    crows_nomic.append(compute_crows_pairs(model_nomic, dim))

print("CrowS-Pairs Nomic:", [f"{x:.2f}" for x in crows_nomic])

# ============ 5. COMPARISON WITH MPNet MRL ============
print("\n" + "=" * 70)
print("5. COMPARISON: Nomic vs MPNet-MRL")
print("=" * 70)

# Load previous MPNet-MRL results from JSON
try:
    with open('step7_all_results.json', 'r') as f:
        prev_results = json.load(f)
    
    mpnet_weat = prev_results['mrl']['weat_abs_es']
    mpnet_C = prev_results['mrl']['C_gender']
    mpnet_stereo = prev_results['mrl']['stereoset_ss']
    mpnet_crows = prev_results['mrl']['crowspairs_cms']
    
    print(f"{'Dim':>5} {'Metric':>20} {'MPNet-MRL':>12} {'Nomic-MRL':>12}")
    print("-" * 60)
    for i, dim in enumerate(DIMS):
        print(f"{dim:>5} {'WEAT |ES|':>20} {mpnet_weat[i]:>12.4f} {weat_nomic[i]:>12.4f}")
        print(f"{dim:>5} {'C(d) gender':>20} {mpnet_C[i]:>12.4f} {C_gender_nomic[i]:>12.4f}")
        print(f"{dim:>5} {'StereoSet SS':>20} {mpnet_stereo[i]:>12.4f} {stereoset_nomic[i]:>12.4f}")
        print(f"{dim:>5} {'CrowS CMS':>20} {mpnet_crows[i]:>12.2f} {crows_nomic[i]:>12.2f}")
        print("-" * 60)
    
except FileNotFoundError:
    print("Previous results file not found. Printing Nomic results only:")
    for i, dim in enumerate(DIMS):
        print(f"Dim {dim}: WEAT={weat_nomic[i]:.4f}, C(d)={C_gender_nomic[i]:.4f}, "
              f"StereoSet={stereoset_nomic[i]:.4f}, CrowS={crows_nomic[i]:.2f}")

# ============ 6. CORRELATION ANALYSIS — NOMIC ============
print("\n" + "=" * 70)
print("6. RQ3: CORRELATION ANALYSIS — NOMIC")
print("=" * 70)

def compute_corr(x, y, name_x, name_y, model_name):
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    print(f"\n{model_name}: {name_x} vs {name_y}")
    print(f"  Pearson r  = {pearson_r:+.4f} (p={pearson_p:.4f})")
    print(f"  Spearman ρ = {spearman_r:+.4f} (p={spearman_p:.4f})")
    return pearson_r, spearman_r

compute_corr(weat_nomic, stereoset_nomic, "WEAT |ES|", "StereoSet SS", "Nomic-MRL")
compute_corr(weat_nomic, crows_nomic, "WEAT |ES|", "CrowS-Pairs CMS", "Nomic-MRL")
compute_corr(C_gender_nomic, stereoset_nomic, "C(d) gender", "StereoSet SS", "Nomic-MRL")
compute_corr(C_gender_nomic, crows_nomic, "C(d) gender", "CrowS-Pairs CMS", "Nomic-MRL")

# ============ 7. SAVE RESULTS ============
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    return obj

results = {
    'dimensions': DIMS,
    'nomic_mrl': {
        'weat_abs_es': weat_nomic,
        'C_gender': C_gender_nomic,
        'stereoset_ss': stereoset_nomic,
        'crowspairs_cms': crows_nomic
    }
}

with open('step8_nomic_results.json', 'w') as f:
    json.dump(convert_to_native(results), f, indent=2)
print("\n✓ Results saved to step8_nomic_results.json")

# ============ 8. PLOT COMPARISON ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: WEAT comparison
ax = axes[0, 0]
ax.plot(DIMS, mpnet_weat if 'mpnet_weat' in dir() else [0]*5, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, weat_nomic, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.set_xlabel("Dimension d")
ax.set_ylabel("WEAT |Effect Size|")
ax.set_title("Intrinsic Bias: MPNet vs Nomic")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: C(d) comparison
ax = axes[0, 1]
ax.plot(DIMS, mpnet_C if 'mpnet_C' in dir() else [0]*5, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, C_gender_nomic, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, [d/FULL_DIM for d in DIMS], 'k:', label='Random Baseline', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("C(d)")
ax.set_title("Bias Concentration: MPNet vs Nomic")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: StereoSet comparison
ax = axes[1, 0]
ax.plot(DIMS, mpnet_stereo if 'mpnet_stereo' in dir() else [0]*5, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, stereoset_nomic, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.axhline(y=1/3, color='black', linestyle=':', label='Random', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("StereoSet SS")
ax.set_title("Selection Bias: MPNet vs Nomic")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: CrowS-Pairs comparison
ax = axes[1, 1]
ax.plot(DIMS, mpnet_crows if 'mpnet_crows' in dir() else [0]*5, 's--', label='MPNet-MRL', linewidth=2, markersize=8)
ax.plot(DIMS, crows_nomic, 'o-', label='Nomic-MRL', linewidth=2, markersize=8)
ax.axhline(y=50, color='black', linestyle=':', label='Unbiased', linewidth=1.5)
ax.set_xlabel("Dimension d")
ax.set_ylabel("CrowS-Pairs CMS")
ax.set_title("Preference Bias: MPNet vs Nomic")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("generalization_nomic_step8.png", dpi=300, bbox_inches='tight')
print("\n✓ Plot saved to generalization_nomic_step8.png")
plt.show()
print("\n✓ Step 8 complete. Generalization test with Nomic finished.")